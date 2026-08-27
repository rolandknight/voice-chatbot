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
    name = "flowcat-webrtc-client",
    version,
    about = "Use native audio devices with the local FlowCat WebRTC server"
)]
struct Cli {
    /// Tracing filter (for example: info, debug, or voice_chatbot_client=trace).
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List available input and output devices.
    Devices,

    /// Start a full-duplex audio call.
    Call {
        /// Base URL of the local FlowCat server.
        #[arg(long, env = "FLOWCAT_URL", default_value = "http://127.0.0.1:6210")]
        server_url: String,

        /// Input selector: default, 1-based index, stable ID, name, or unique substring.
        #[arg(long, env = "FLOWCAT_INPUT_DEVICE")]
        input_device: Option<String>,

        /// Output selector: default, 1-based index, stable ID, name, or unique substring.
        #[arg(long, env = "FLOWCAT_OUTPUT_DEVICE")]
        output_device: Option<String>,

        /// On-device wake words: a directory of openWakeWord heads
        /// (hey_<persona>.onnx), relative to the working directory. Audio is
        /// only sent while a wake session is open, and the persona that woke
        /// is reported to the server.
        #[arg(long, env = "FLOWCAT_WAKE_DIR", default_value = "models/wakeword")]
        wake_dir: String,

        /// Always-on (push) mode: send audio continuously, no wake words.
        #[arg(long, env = "FLOWCAT_NO_WAKE", conflicts_with = "wake_dir")]
        no_wake: bool,

        /// Wake probability threshold per head.
        #[arg(long, env = "FLOWCAT_WAKE_THRESHOLD", default_value_t = 0.5)]
        wake_threshold: f32,

        /// Silence (seconds) that ends a wake session.
        #[arg(long, env = "FLOWCAT_WAKE_SESSION_SECS", default_value_t = 15.0)]
        wake_session_secs: f32,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level)?;

    match cli.command {
        Command::Devices => AudioDevices::new()?.print(),
        Command::Call {
            server_url,
            input_device,
            output_device,
            wake_dir,
            no_wake,
            wake_threshold,
            wake_session_secs,
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

fn init_tracing(filter: &str) -> Result<()> {
    let filter = EnvFilter::try_new(filter).context("parse --log-level filter")?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init()
        .map_err(|error| anyhow::anyhow!("initialize logging: {error}"))
}

async fn run_call(
    server_url: &str,
    input_selector: Option<&str>,
    output_selector: Option<&str>,
    wake: Option<WakeConfig>,
) -> Result<()> {
    let endpoints = ServerEndpoints::new(server_url)?;
    let http = Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .build()
        .context("build FlowCat HTTP client")?;

    require_healthy(&http, &endpoints).await.with_context(|| {
        format!(
            "FlowCat is not ready at {}; start it with `make poc-up`",
            endpoints.health_url()
        )
    })?;

    let audio = AudioDevices::new()?.open(input_selector, output_selector)?;
    let AudioIoParts {
        streams,
        input_rate,
        output_rate,
        input_device,
        output_device,
        input_rx,
        output_tx,
    } = audio.into_parts();

    eprintln!(
        "input:  {} ({}, {})",
        input_device.name, input_device.id, input_device.config
    );
    eprintln!(
        "output: {} ({}, {})",
        output_device.name, output_device.id, output_device.config
    );

    // Bind and advertise on the interface that reaches the server, so
    // `--server-url http://<lan-ip>:6210` pairs across machines (no STUN needed
    // on a LAN); a loopback URL still yields a loopback candidate.
    let (host, port) = endpoints.host_port()?;
    let server_addr = tokio::net::lookup_host((host.as_str(), port))
        .await
        .with_context(|| format!("resolve server host {host}:{port}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("server host {host} resolved to no address"))?;
    let pending = PendingPeer::create_toward(server_addr).await?;
    let response = exchange_offer(&http, &endpoints, pending.offer_sdp()).await?;
    let peer = pending.accept_answer(&response.sdp)?;

    let events_url = endpoints.events_url(&response.pc_id)?;
    let (event_shutdown_tx, event_shutdown_rx) = watch::channel(false);
    // Radio/shows/sound effects the server's skills start play here via mpv,
    // on the call's output device. Without mpv the call still works; media
    // commands are logged and dropped.
    let media = if MediaPlayer::is_available() {
        Some(MediaPlayer::new(Some(&output_device.name), server_url))
    } else {
        tracing::warn!("mpv not found; BBC radio, shows and sound effects will not play (brew/apt install mpv)");
        None
    };
    let (outbound_tx, outbound_rx) = tokio::sync::mpsc::unbounded_channel();
    let activity = Activity::default();
    let event_task = tokio::spawn(events::run(
        events_url,
        event_shutdown_rx,
        media,
        outbound_rx,
        activity.clone(),
    ));

    // On-device wake: gate the capture channel before the peer sees it.
    let input_rx = match &wake {
        Some(cfg) => {
            for (path, persona) in &cfg.heads {
                eprintln!("wake:   {} -> {persona}", path.display());
            }
            let gate = ClientWakeGate::new(cfg, input_rate, activity)?;
            eprintln!("wake:   listening (audio is sent only after a wake word)");
            voice_chatbot_client::wake::spawn(gate, input_rx, outbound_tx)
        }
        None => {
            drop(outbound_tx);
            input_rx
        }
    };

    streams.start()?;
    eprintln!("audio devices started; negotiating WebRTC (press Ctrl-C to hang up)");

    let call_result = peer
        .run(input_rate, output_rate, input_rx, output_tx, async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::warn!(%error, "failed to listen for Ctrl-C");
            }
        })
        .await;

    // Stop native callbacks before waiting for the side-channel WebSocket.
    drop(streams);
    let _ = event_shutdown_tx.send(true);
    match tokio::time::timeout(Duration::from_secs(2), event_task).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::warn!(%error, "event task failed"),
        Err(_) => tracing::warn!("event task did not stop within two seconds"),
    }

    call_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

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
            Command::Devices => panic!("expected call command"),
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
            } => {
                assert_eq!(server_url, "http://localhost:7000");
                assert_eq!(input_device.as_deref(), Some("Jabra input"));
                assert_eq!(output_device.as_deref(), Some("2"));
                assert_eq!(wake_dir, "models/wakeword", "wake is on by default");
                assert!(!no_wake);
                assert_eq!(wake_threshold, 0.5);
                assert_eq!(wake_session_secs, 15.0);
            }
            Command::Devices => panic!("expected call command"),
        }
    }
}
