use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use reqwest::Client;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;
use voice_chatbot_client::audio::{AudioDevices, AudioIoParts};
use voice_chatbot_client::events;
use voice_chatbot_client::media::MediaPlayer;
use voice_chatbot_client::peer::PendingPeer;
use voice_chatbot_client::protocol::{exchange_offer, require_healthy, ServerEndpoints};
use voice_chatbot_client::wake::{Activity, ClientWakeGate, WakeConfig};

#[derive(Debug, Parser)]
#[command(
    name = "voice-chatbot-client",
    version,
    about = "Use native audio devices with the local voice-chatbot WebRTC server"
)]
struct Cli {
    /// Tracing filter (for example: info, debug, or voice_chatbot_client=trace).
    #[arg(long, global = true, env = "LOG_LEVEL", default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Command,
}

/// Whether to drive a speakerphone's LED ring (docs/specs/jabra-led.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum LedMode {
    /// Drive a Jabra's telephony LEDs when one is plugged in.
    Auto,
    /// Never touch the LEDs.
    Off,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List available input and output devices.
    Devices,

    /// Probe the speakerphone's LEDs: open the Jabra telephony interface and
    /// cycle off -> listening/thinking (green) -> muted -> off.
    LedTest,

    /// Start a full-duplex audio call.
    Call {
        /// Base URL of the local voice-chatbot server.
        #[arg(long, env = "SERVER_URL", default_value = "http://127.0.0.1:6210")]
        server_url: String,

        /// Input selector: default, 1-based index, stable ID, name, or unique
        /// substring. Unset, the Jabra speakerphone is used when one is
        /// plugged in; pass `default` for the system default instead.
        #[arg(long, env = "INPUT_DEVICE")]
        input_device: Option<String>,

        /// Output selector: default, 1-based index, stable ID, name, or unique
        /// substring. Unset, the Jabra speakerphone is used when one is
        /// plugged in; pass `default` for the system default instead.
        #[arg(long, env = "OUTPUT_DEVICE")]
        output_device: Option<String>,

        /// On-device wake words: a directory of openWakeWord heads
        /// (hey_<persona>.onnx), relative to the working directory. Audio is
        /// only sent while a wake session is open, and the persona that woke
        /// is reported to the server.
        #[arg(long, env = "WAKE_DIR", default_value = "models/wakeword")]
        wake_dir: String,

        /// Always-on (push) mode: send audio continuously, no wake words.
        #[arg(long, env = "NO_WAKE", conflicts_with = "wake_dir")]
        no_wake: bool,

        /// Wake probability threshold per head.
        #[arg(long, env = "WAKE_THRESHOLD", default_value_t = 0.5)]
        wake_threshold: f32,

        /// Silence (seconds) that ends a wake session.
        #[arg(long, env = "WAKE_SESSION_SECS", default_value_t = 5.0)]
        wake_session_secs: f32,

        /// Show chatbot activity on the speakerphone's LED ring.
        #[arg(long, env = "LED", value_enum, default_value_t = LedMode::Auto)]
        led: LedMode,
    },
}

/// Names this client stopped reading when the `FLOWCAT_` prefix was dropped
/// for the bare names the repo-root `.env` already uses.
const RETIRED_PREFIX: &str = "FLOWCAT_";

/// The startup error for any surviving `FLOWCAT_*`. Left set, one is simply
/// never read, and the client silently dials the built-in default instead --
/// the same trap the server's `POC_` guard exists to close.
fn retired_var_error<I: Iterator<Item = String>>(keys: I) -> Option<String> {
    let stale = voice_chatbot_env_file::names_with_prefix(keys, RETIRED_PREFIX);
    if stale.is_empty() {
        return None;
    }
    Some(format!(
        "these environment variables lost their {RETIRED_PREFIX} prefix and are no longer read: {}. \
Rename them (drop {RETIRED_PREFIX}; {RETIRED_PREFIX}URL is now SERVER_URL) \
-- leaving them set means silently running on the defaults.",
        stale.join(", ")
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    // The repo-root .env (run from the repo root; silently skipped otherwise)
    // is the same file the server reads. It has to land before Cli::parse,
    // which is when clap consults the `env` fallbacks above, and so before
    // init_tracing -- the loader's debug line for an unparsable line goes
    // unlogged here, but the server logs the same line from the same file.
    // Variables already set are never overridden, so a flag or an exported
    // variable still wins.
    voice_chatbot_env_file::load_if_unset(std::path::Path::new(".env"));
    if let Some(error) = retired_var_error(std::env::vars().map(|(k, _)| k)) {
        anyhow::bail!(error);
    }
    let cli = Cli::parse();
    init_tracing(&cli.log_level)?;

    match cli.command {
        Command::Devices => AudioDevices::new()?.print(),
        Command::LedTest => led_test()?,
        Command::Call {
            server_url,
            input_device,
            output_device,
            wake_dir,
            no_wake,
            wake_threshold,
            wake_session_secs,
            led,
        } => {
            let wake = if no_wake {
                None
            } else {
                Some(wake_config(&wake_dir, wake_threshold, wake_session_secs)?)
            };
            run_call(
                &server_url,
                input_device.as_deref(),
                output_device.as_deref(),
                wake,
                led,
            )
            .await?
        }
    }

    Ok(())
}

/// Resolve `--wake-dir` (relative to the working directory) into heads.
fn wake_config(dir: &str, threshold: f32, session_secs: f32) -> Result<WakeConfig> {
    let cwd = std::env::current_dir().context("current directory")?;
    let heads = voice_chatbot_wake::resolve_heads(&cwd, dir, "")
        .map_err(|e| anyhow::anyhow!("--wake-dir: {e}"))?;
    if !(0.0..=1.0).contains(&threshold) {
        anyhow::bail!("--wake-threshold must be in [0, 1]");
    }
    if !(session_secs > 0.0 && session_secs.is_finite()) {
        anyhow::bail!("--wake-session-secs must be positive");
    }
    Ok(WakeConfig {
        heads,
        threshold,
        session_secs,
    })
}

/// Hardware probe for docs/specs/jabra-led.md: what does each LED state
/// look (and sound) like on the attached device?
fn led_test() -> Result<()> {
    use voice_chatbot_client::led::LedSink;
    let mut leds = voice_chatbot_client::led::hid::open()?;
    println!("driving {}", leds.describe());
    // The Ring usage is intentionally never driven: it double-beeps on the
    // Speak2 40, so listening and thinking both show solid green.
    let steps: [(&str, (bool, bool, bool)); 4] = [
        ("off (asleep)", (false, false, false)),
        (
            "listening / thinking: off-hook -- expect solid green",
            (true, false, false),
        ),
        (
            "muted: off-hook + mute -- expect solid red",
            (true, false, true),
        ),
        ("off again", (false, false, false)),
    ];
    for (what, (off_hook, ring, mute)) in steps {
        println!("{what}");
        leds.set(off_hook, ring, mute)?;
        std::thread::sleep(Duration::from_secs(3));
    }
    Ok(())
}

fn init_tracing(filter: &str) -> Result<()> {
    let filter = EnvFilter::try_new(filter).context("parse --log-level filter")?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init()
        .map_err(|error| anyhow::anyhow!("initialize logging: {error}"))
}

/// Delay between reconnect attempts while the server is unreachable.
const RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// Why a session ended.
enum SessionEnd {
    /// The user pressed Ctrl-C; do not reconnect.
    HungUp,
    /// The server could not be reached (health check or offer failed).
    Unreachable(anyhow::Error),
    /// A live call was lost.
    Lost(anyhow::Error),
}

/// Keep the call up: connect, run a session, and reconnect whenever the
/// server is unavailable or the connection drops. Prints one line when the
/// server cannot be reached, one when a live connection dies, and one when
/// a connection is (re)established -- nothing while retrying in between.
async fn run_call(
    server_url: &str,
    input_selector: Option<&str>,
    output_selector: Option<&str>,
    wake: Option<WakeConfig>,
    led: LedMode,
) -> Result<()> {
    let endpoints = ServerEndpoints::new(server_url)?;
    let http = Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .build()
        .context("build the server HTTP client")?;

    // One Ctrl-C listener for the whole process; sessions and the retry
    // delay both watch it so hanging up works at any point.
    let (hangup_tx, hangup_rx) = watch::channel(false);
    tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!(%error, "failed to listen for Ctrl-C");
        }
        let _ = hangup_tx.send(true);
    });

    // Audio devices are opened once per session but only described once.
    let mut first_session = true;
    // Set after a "cannot connect"/"connection lost" line so retries stay
    // quiet and the next success is announced as a reconnect.
    let mut announced_outage = false;

    loop {
        let session = run_session(
            &http,
            &endpoints,
            server_url,
            input_selector,
            output_selector,
            wake.as_ref(),
            hangup_rx.clone(),
            first_session,
            announced_outage,
            led,
        )
        .await;

        match session {
            SessionEnd::HungUp => return Ok(()),
            SessionEnd::Unreachable(error) => {
                if !announced_outage {
                    eprintln!(
                        "cannot connect to the server at {}: {error:#}; retrying every {}s (Ctrl-C to quit)",
                        endpoints.health_url(),
                        RECONNECT_DELAY.as_secs()
                    );
                    announced_outage = true;
                } else {
                    tracing::debug!(%error, "reconnect attempt failed");
                }
            }
            SessionEnd::Lost(error) => {
                first_session = false;
                // A lost call always gets a line, even during an outage that
                // was already announced: it is new information.
                eprintln!(
                    "connection to the server lost: {error:#}; reconnecting (Ctrl-C to quit)"
                );
                announced_outage = true;
            }
        }

        if wait_or_hangup(hangup_rx.clone(), RECONNECT_DELAY).await {
            return Ok(());
        }
    }
}

/// Sleep for `delay`; true if Ctrl-C arrived first.
async fn wait_or_hangup(mut hangup: watch::Receiver<bool>, delay: Duration) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(delay) => false,
        hung_up = hangup.wait_for(|hung_up| *hung_up) => hung_up.is_ok(),
    }
}

/// One connection attempt and, if it succeeds, one call until it ends.
#[allow(clippy::too_many_arguments)]
async fn run_session(
    http: &Client,
    endpoints: &ServerEndpoints,
    server_url: &str,
    input_selector: Option<&str>,
    output_selector: Option<&str>,
    wake: Option<&WakeConfig>,
    mut hangup: watch::Receiver<bool>,
    describe_devices: bool,
    reconnecting: bool,
    led_mode: LedMode,
) -> SessionEnd {
    if *hangup.borrow() {
        return SessionEnd::HungUp;
    }
    if let Err(error) = require_healthy(http, endpoints).await {
        return SessionEnd::Unreachable(error);
    }

    let audio = match AudioDevices::new()
        .and_then(|devices| devices.open(input_selector, output_selector))
    {
        Ok(audio) => audio,
        Err(error) => return SessionEnd::Lost(error.context("open audio devices")),
    };
    let AudioIoParts {
        streams,
        input_rate,
        output_rate,
        input_device,
        output_device,
        input_rx,
        output_tx,
        media_tx,
        voice_recycle_rx,
        media_recycle_rx,
        media_gain,
    } = audio.into_parts();

    if describe_devices {
        eprintln!(
            "input:  {} ({}, {})",
            input_device.name, input_device.id, input_device.config
        );
        eprintln!(
            "output: {} ({}, {})",
            output_device.name, output_device.id, output_device.config
        );
    }

    // Bind and advertise on the interface that reaches the server, so
    // `--server-url http://<lan-ip>:6210` pairs across machines (no STUN needed
    // on a LAN); a loopback URL still yields a loopback candidate.
    let negotiate = async {
        let (host, port) = endpoints.host_port()?;
        let server_addr = tokio::net::lookup_host((host.as_str(), port))
            .await
            .with_context(|| format!("resolve server host {host}:{port}"))?
            .next()
            .ok_or_else(|| anyhow::anyhow!("server host {host} resolved to no address"))?;
        let pending = PendingPeer::create_toward(server_addr).await?;
        let response = exchange_offer(http, endpoints, pending.offer_sdp()).await?;
        let peer = pending.accept_answer(&response.sdp)?;
        anyhow::Ok((peer, response.pc_id))
    };
    let (peer, pc_id) = match negotiate.await {
        Ok(negotiated) => negotiated,
        Err(error) => return SessionEnd::Unreachable(error),
    };

    let events_url = match endpoints.events_url(&pc_id) {
        Ok(url) => url,
        Err(error) => return SessionEnd::Unreachable(error),
    };
    let (event_shutdown_tx, event_shutdown_rx) = watch::channel(false);
    // Radio, shows and sound effects mix into the call's own output stream, so
    // they play on the call's device. Without ffmpeg the call still works;
    // media commands are logged and dropped.
    let media = if MediaPlayer::is_available() {
        Some(MediaPlayer::new(
            media_tx,
            media_recycle_rx,
            media_gain,
            output_rate,
            server_url,
        ))
    } else {
        if describe_devices {
            tracing::warn!(
                "ffmpeg not found; BBC radio, shows and sound effects will not play \
                 (brew/apt install ffmpeg)"
            );
        }
        None
    };

    // Chatbot activity on the speakerphone's LED ring. A missing device or
    // missing hidraw permissions degrade to running dark, like ffmpeg above.
    let (led, led_done) = match led_mode {
        LedMode::Off => (None, None),
        LedMode::Auto => match voice_chatbot_client::led::hid::open() {
            Ok(leds) => {
                if describe_devices {
                    eprintln!("leds:   {}", leds.describe());
                }
                let (controller, done) =
                    voice_chatbot_client::led::LedController::start(Box::new(leds), wake.is_none());
                (Some(controller), Some(done))
            }
            Err(error) => {
                if describe_devices {
                    tracing::warn!(%error, "no speakerphone leds; running without");
                }
                (None, None)
            }
        },
    };

    let (outbound_tx, outbound_rx) = tokio::sync::mpsc::unbounded_channel();
    let activity = Activity::default();
    let mut event_task = tokio::spawn(events::run(
        events_url,
        event_shutdown_rx,
        media,
        outbound_rx,
        activity.clone(),
        led.clone(),
    ));

    // On-device wake: gate the capture channel before the peer sees it.
    let input_rx = match wake {
        Some(cfg) => {
            if describe_devices {
                for (path, persona) in &cfg.heads {
                    eprintln!("wake:   {} -> {persona}", path.display());
                }
            }
            let gate = match ClientWakeGate::new(cfg, input_rate, activity) {
                Ok(gate) => gate,
                Err(error) => return SessionEnd::Lost(error.context("start wake gate")),
            };
            if describe_devices {
                eprintln!("wake:   listening (audio is sent only after a wake word)");
            }
            voice_chatbot_client::wake::spawn(gate, input_rx, outbound_tx, led.clone())
        }
        None => {
            drop(outbound_tx);
            input_rx
        }
    };

    if let Err(error) = streams.start() {
        return SessionEnd::Lost(error);
    }
    if reconnecting {
        eprintln!("reconnected to the server at {}", endpoints.health_url());
    } else {
        eprintln!("audio devices started; negotiating WebRTC (press Ctrl-C to hang up)");
    }

    // Cancel the peer on Ctrl-C, or as soon as the events WebSocket ends:
    // that is the fastest sign the server went away (ICE only notices after
    // its own timeout), and the peer must not wait on ICE to close.
    let mut hung_up = false;
    let call_result = peer
        .run(
            input_rate,
            output_rate,
            input_rx,
            output_tx,
            voice_recycle_rx,
            async {
                tokio::select! {
                    result = hangup.wait_for(|hung_up| *hung_up) => hung_up = result.is_ok(),
                    _ = &mut event_task => {}
                }
            },
        )
        .await;
    let call_result = match call_result {
        Ok(()) if !hung_up => Err(anyhow::anyhow!("server event stream closed")),
        other => other,
    };

    // Stop native callbacks before waiting for the side-channel WebSocket.
    drop(streams);
    let _ = event_shutdown_tx.send(true);
    if !event_task.is_finished() {
        match tokio::time::timeout(Duration::from_secs(2), &mut event_task).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!(%error, "event task failed"),
            Err(_) => {
                tracing::warn!("event task did not stop within two seconds");
                event_task.abort();
            }
        }
    }

    // The led driver darkens the ring once every handle is gone (the events
    // and wake tasks hold the other clones); bounded so a wedged device
    // write cannot stall teardown or leave the ring lit into the next state.
    drop(led);
    if let Some(done) = led_done {
        let _ = tokio::time::timeout(Duration::from_secs(1), done).await;
    }

    match call_result {
        Ok(()) => SessionEnd::HungUp,
        Err(error) => SessionEnd::Lost(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn stale_flowcat_names_are_named_with_their_replacement() {
        let keys = ["FLOWCAT_URL", "SERVER_URL", "FLOWCAT_WAKE_DIR"]
            .into_iter()
            .map(String::from);
        let message = retired_var_error(keys).expect("stale names must be refused");
        assert!(
            message.contains("no longer read: FLOWCAT_URL, FLOWCAT_WAKE_DIR."),
            "lists the stale names and only those: {message}"
        );
        assert!(
            message.contains("FLOWCAT_URL is now SERVER_URL"),
            "points at the replacement: {message}"
        );
    }

    #[test]
    fn an_environment_without_stale_names_starts() {
        let keys = ["SERVER_URL", "WAKE_DIR"].into_iter().map(String::from);
        assert!(retired_var_error(keys).is_none());
    }

    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_device_listing() {
        let cli = Cli::try_parse_from(["client", "devices"]).unwrap();
        assert!(matches!(cli.command, Command::Devices));
        assert_eq!(cli.log_level, "info");
    }

    #[test]
    fn no_wake_selects_push_mode_and_excludes_wake_dir() {
        let cli = Cli::try_parse_from(["client", "call", "--no-wake"]).unwrap();
        match cli.command {
            Command::Call { no_wake, .. } => assert!(no_wake),
            _ => panic!("expected call command"),
        }
        assert!(Cli::try_parse_from(["client", "call", "--no-wake", "--wake-dir", "x"]).is_err());
    }

    #[test]
    fn parses_call_device_selectors_and_global_log_filter() {
        let cli = Cli::try_parse_from([
            "client",
            "call",
            "--server-url",
            "http://localhost:7000",
            "--input-device",
            "Jabra input",
            "--output-device",
            "2",
            "--log-level",
            "debug",
        ])
        .unwrap();

        assert_eq!(cli.log_level, "debug");
        match cli.command {
            Command::Call {
                server_url,
                input_device,
                output_device,
                wake_dir,
                no_wake,
                wake_threshold,
                wake_session_secs,
                led,
            } => {
                assert_eq!(server_url, "http://localhost:7000");
                assert_eq!(input_device.as_deref(), Some("Jabra input"));
                assert_eq!(output_device.as_deref(), Some("2"));
                assert_eq!(wake_dir, "models/wakeword", "wake is on by default");
                assert!(!no_wake);
                assert_eq!(wake_threshold, 0.5);
                assert_eq!(wake_session_secs, 5.0);
                assert_eq!(led, LedMode::Auto);
            }
            _ => panic!("expected call command"),
        }
    }

    #[test]
    fn parses_led_mode() {
        let cli = Cli::try_parse_from(["client", "call", "--led", "off"]).unwrap();
        match cli.command {
            Command::Call { led, .. } => assert_eq!(led, LedMode::Off),
            _ => panic!("expected call"),
        }
        let cli = Cli::try_parse_from(["client", "call"]).unwrap();
        match cli.command {
            Command::Call { led, .. } => assert_eq!(led, LedMode::Auto, "auto by default"),
            _ => panic!("expected call"),
        }
        assert!(matches!(
            Cli::try_parse_from(["client", "led-test"]).unwrap().command,
            Command::LedTest
        ));
    }
}
