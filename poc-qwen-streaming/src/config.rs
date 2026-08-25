//! The Rust-side view of config.yaml. The engine sections (models:,
//! generation:, voices:, transcribe:) are read by poc-qwen's Python loader
//! from the same file; only what the server needs is deserialised here.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: Server,
    pub python: Python,
    #[serde(default)]
    pub bench: Bench,
    #[serde(skip)]
    pub base_dir: PathBuf,
    #[serde(skip)]
    pub path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Server {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Python {
    /// Prepended to sys.path, relative to the config file.
    #[serde(default)]
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Bench {
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_repeats")]
    pub repeats: usize,
}

impl Default for Bench {
    fn default() -> Self {
        Self { voice: default_voice(), size: default_size(), repeats: default_repeats() }
    }
}

fn default_host() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    8008
}
fn default_voice() -> String {
    "one-one".into()
}
fn default_size() -> String {
    "1.7B".into()
}
fn default_repeats() -> usize {
    2
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut cfg: Config = serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cfg.path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        cfg.base_dir = cfg.path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        if let Ok(host) = std::env::var("HOST") {
            if !host.trim().is_empty() {
                cfg.server.host = host;
            }
        }
        if let Ok(port) = std::env::var("POC_QWEN_SERVER_PORT") {
            if let Ok(p) = port.trim().parse() {
                cfg.server.port = p;
            }
        }
        Ok(cfg)
    }

    pub fn python_paths(&self) -> Vec<PathBuf> {
        self.python.paths.iter().map(|p| self.base_dir.join(p)).collect()
    }

    pub fn ui_dir(&self) -> PathBuf {
        self.base_dir.join("ui")
    }

    pub fn reports_dir(&self) -> PathBuf {
        self.base_dir.join("reports")
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.base_dir.join("uploads")
    }
}
