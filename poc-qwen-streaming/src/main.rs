//! poc-qwen-streaming: Rust server embedding poc-qwen's mlx-audio engine.
//!
//!   poc-qwen-streaming [--config config.yaml] serve   # default
//!   poc-qwen-streaming info                          # print model_info() and exit
//!   poc-qwen-streaming bench                         # headless TTFA bench

mod bench;
mod config;
mod engine;
mod pcm;
mod server;

use anyhow::Result;
use std::path::PathBuf;

use config::Config;
use engine::Engine;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=warn".into()))
        .init();

    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut config_path = PathBuf::from("config.yaml");
    if let Some(i) = args.iter().position(|a| a == "--config") {
        config_path = PathBuf::from(args.get(i + 1).cloned().unwrap_or_default());
        args.drain(i..=i + 1);
    }
    let cmd = args.first().map(String::as_str).unwrap_or("serve").to_string();
    let cfg = Config::load(&config_path)?;

    // Python is started on its own thread before the runtime; the runtime only
    // ever talks to it through channels.
    let engine = Engine::start(&cfg)?;

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        match cmd.as_str() {
            "info" => {
                println!("{}", serde_json::to_string_pretty(&engine.info().await?)?);
                Ok(())
            }
            "bench" => bench::run(&cfg, &engine).await,
            "serve" => serve(&cfg, engine).await,
            other => anyhow::bail!("unknown command {other:?} (serve | info | bench)"),
        }
    })
}

async fn serve(cfg: &Config, engine: Engine) -> Result<()> {
    let app = server::router(cfg, engine.clone());
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("serving http://{addr}  (ui: {})", cfg.ui_dir().display());
    // Preload after bind so the UI is reachable at once; generations issued
    // meanwhile simply queue behind it on the python thread. /api/info
    // reports progress under "preload".
    let pre = engine.clone();
    tokio::spawn(async move {
        if let Err(e) = pre.preload().await {
            tracing::warn!("preload: {e}");
        }
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    // Exit without unwinding into the interpreter: the Python thread is parked
    // in rx.recv() and MLX's Metal state is cheaper to drop with the process.
    std::process::exit(0);
}
