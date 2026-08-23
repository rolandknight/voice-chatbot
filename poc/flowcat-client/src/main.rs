use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flowcat_webrtc_client::audio::{AudioDevices, AudioIoParts};
use flowcat_webrtc_client::events;
use flowcat_webrtc_client::peer::PendingPeer;
use flowcat_webrtc_client::protocol::{exchange_offer, require_healthy, ServerEndpoints};
use reqwest::Client;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "flowcat-webrtc-client",
    version,
    about = "Use native audio devices with the local FlowCat WebRTC server"
)]
struct Cli {
    /// Tracing filter (for example: info, debug, or flowcat_webrtc_client=trace).
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
        } => {
            run_call(
                &server_url,
                input_device.as_deref(),
                output_device.as_deref(),
            )
            .await?
        }
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

async fn run_call(
    server_url: &str,
    input_selector: Option<&str>,
    output_selector: Option<&str>,
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

    let pending = PendingPeer::create().await?;
    let response = exchange_offer(&http, &endpoints, pending.offer_sdp()).await?;
    let peer = pending.accept_answer(&response.sdp)?;

    let events_url = endpoints.events_url(&response.pc_id)?;
    let (event_shutdown_tx, event_shutdown_rx) = watch::channel(false);
    let event_task = tokio::spawn(events::run(events_url, event_shutdown_rx));

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
            } => {
                assert_eq!(server_url, "http://localhost:7000");
                assert_eq!(input_device.as_deref(), Some("Jabra input"));
                assert_eq!(output_device.as_deref(), Some("2"));
            }
            Command::Devices => panic!("expected call command"),
        }
    }
}
