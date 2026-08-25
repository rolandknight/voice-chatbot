//! Headless TTFA bench: the three poc-tts bench sentences, cloned from the
//! configured preset voice, streamed through the same engine path the
//! WebSocket uses. Appends to reports/rs_runs.jsonl with `bench: true`;
//! the last take of each sentence is written to reports/bench_<i>.wav.

use anyhow::{anyhow, Result};
use serde_json::json;

use crate::config::Config;
use crate::engine::{Engine, StreamEvent};
use crate::pcm::write_wav;
use crate::server::record_run;

pub async fn run(cfg: &Config, engine: &Engine) -> Result<()> {
    let sentences = engine.bench_sentences().await?;
    let catalog = engine.catalog().await?;
    let voice = catalog["voices"]
        .as_array()
        .and_then(|vs| vs.iter().find(|v| v["name"] == cfg.bench.voice))
        .cloned()
        .ok_or_else(|| anyhow!("bench voice {:?} not found in voices/", cfg.bench.voice))?;
    let reports = cfg.reports_dir();
    tokio::fs::create_dir_all(&reports).await?;
    let runs = reports.join("rs_runs.jsonl");

    println!("{:<8} {:<8} {:>4} {:>7} {:>7} {:>7} {:>6} {:>6}", "size", "sentence", "rep", "ttfa_s", "gen_s", "audio_s", "rtf", "chunks");
    for (name, sentence) in sentences.iter() {
        for rep in 0..cfg.bench.repeats {
            let params = json!({
                "text": sentence,
                "ref_audio": voice["path"],
                "ref_text": voice["transcript"],
                "language": "English",
                "size": cfg.bench.size,
            });
            let mut rx = engine.generate("clone", params)?;
            let mut samples: Vec<i16> = Vec::new();
            let mut sr = 24_000;
            let mut timings = None;
            while let Some(ev) = rx.recv().await {
                match ev {
                    StreamEvent::Start { sample_rate, .. } => sr = sample_rate,
                    StreamEvent::Audio { samples: s, .. } => samples.extend_from_slice(&s),
                    StreamEvent::Done { timings: t } => timings = Some(t),
                    StreamEvent::Error(e) => return Err(anyhow!("bench generation failed: {e}")),
                }
            }
            let mut t = timings.ok_or_else(|| anyhow!("no Done event"))?;
            t["bench"] = json!(true);
            t["sentence"] = json!(name);
            t["cold"] = json!(rep == 0);
            t["size"] = json!(cfg.bench.size);
            println!(
                "{:<8} {:<8} {:>4} {:>7.3} {:>7.3} {:>7.3} {:>6.3} {:>6}",
                cfg.bench.size,
                name,
                rep,
                t["ttfa_s"].as_f64().unwrap_or(0.0),
                t["gen_s"].as_f64().unwrap_or(0.0),
                t["audio_s"].as_f64().unwrap_or(0.0),
                t["rtf"].as_f64().unwrap_or(0.0),
                t["chunks"].as_u64().unwrap_or(0),
            );
            record_run(&runs, &t).await;
            write_wav(&reports.join(format!("bench_{name}.wav")), &samples, sr)?;
        }
    }
    println!("rows appended to {}", runs.display());
    Ok(())
}
