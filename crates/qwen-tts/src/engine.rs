//! The one thread that talks to Python.
//!
//! MLX keeps per-thread Metal state, and the Qwen PoC learned (commit faca18a)
//! that touching it from short-lived pool threads segfaults. So: a single
//! dedicated OS thread attaches to the interpreter, builds the bridge, and
//! serves commands from a channel for the life of the process. tokio never
//! sees a Python object. The engine's own `mlx-worker` daemon thread stays in
//! charge of Metal; the Python queue waits it blocks on release the GIL, so
//! the two threads hand off cleanly.

use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

use crate::config::Config;
use crate::pcm::f32_to_i16;

/// What the WebSocket (or the bench) receives while a generation runs.
#[derive(Debug)]
pub enum StreamEvent {
    Start { sample_rate: u32, model: String },
    Audio { samples: Vec<i16> },
    Done { timings: Value },
    Error(String),
}

pub enum Cmd {
    Info(oneshot::Sender<Result<Value>>),
    Catalog(oneshot::Sender<Result<Value>>),
    VoicePath(String, oneshot::Sender<Result<Option<String>>>),
    Transcribe(String, oneshot::Sender<Result<String>>),
    Unload(oneshot::Sender<Result<()>>),
    /// Run Python's exit handlers and end the Python thread. Replies once
    /// they have run; the engine is unusable afterwards.
    Shutdown(oneshot::Sender<Result<()>>),
    BenchSentences(oneshot::Sender<Result<Vec<(String, String)>>>),
    /// Queue load + warm of configured models and preset ICL-cache priming on the
    /// MLX worker; replies at once with the initial status (progress via Info).
    Preload(oneshot::Sender<Result<Value>>),
    Generate {
        tab: String,
        params: Value,
        tx: mpsc::Sender<StreamEvent>,
    },
}

#[derive(Clone)]
pub struct Engine {
    tx: std_mpsc::Sender<Cmd>,
}

impl Engine {
    /// Spawn the Python thread; returns once the bridge has been constructed
    /// (or failed to), so start-up errors surface immediately.
    pub fn start(cfg: &Config) -> Result<Self> {
        let (tx, rx) = std_mpsc::channel::<Cmd>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();
        let paths = cfg.python_paths();
        let config_path = cfg.path.clone();
        std::thread::Builder::new()
            .name("python".into())
            .spawn(move || python_thread(paths, config_path, rx, ready_tx))
            .context("spawning python thread")?;
        ready_rx
            .recv()
            .context("python thread died during start-up")??;
        Ok(Self { tx })
    }

    fn send(&self, cmd: Cmd) -> Result<()> {
        self.tx
            .send(cmd)
            .map_err(|_| anyhow!("python thread is gone"))
    }

    pub async fn info(&self) -> Result<Value> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::Info(tx))?;
        rx.await?
    }

    pub async fn catalog(&self) -> Result<Value> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::Catalog(tx))?;
        rx.await?
    }

    pub async fn voice_path(&self, name: &str) -> Result<Option<String>> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::VoicePath(name.to_string(), tx))?;
        rx.await?
    }

    pub async fn transcribe(&self, path: &str) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::Transcribe(path.to_string(), tx))?;
        rx.await?
    }

    pub async fn unload(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::Unload(tx))?;
        rx.await?
    }

    /// Run Python's `atexit` handlers and end the Python thread.
    ///
    /// The interpreter is embedded, so nothing finalizes it when the process
    /// exits: `multiprocessing`'s exit handler — the one that unlinks the
    /// semaphores libraries create at import and unregisters them with its
    /// `resource_tracker` — never runs, and the tracker prints "leaked
    /// semaphore objects" as it dies. Running the handlers here keeps the exit
    /// clean. Later commands fail with "python thread is gone".
    pub async fn shutdown(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::Shutdown(tx))?;
        rx.await?
    }

    /// poc-tts's (name, text) bench sentences, shared by every TTS PoC in the repo.
    pub async fn bench_sentences(&self) -> Result<Vec<(String, String)>> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::BenchSentences(tx))?;
        rx.await?
    }

    pub async fn preload(&self) -> Result<Value> {
        let (tx, rx) = oneshot::channel();
        self.send(Cmd::Preload(tx))?;
        rx.await?
    }

    /// Start a generation; events arrive on the returned receiver. Dropping
    /// the receiver cancels the generation after the current chunk.
    pub fn generate(&self, tab: &str, params: Value) -> Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel(64);
        self.send(Cmd::Generate {
            tab: tab.to_string(),
            params,
            tx,
        })?;
        Ok(rx)
    }
}

// ---------------------------------------------------------------------------

fn python_thread(
    paths: Vec<PathBuf>,
    config_path: PathBuf,
    rx: std_mpsc::Receiver<Cmd>,
    ready: std_mpsc::Sender<Result<()>>,
) {
    Python::attach(|py| {
        let bridge = match init_bridge(py, &paths, &config_path) {
            Ok(b) => {
                let _ = ready.send(Ok(()));
                b
            }
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        // Block for commands with the GIL released so the mlx-worker thread
        // and Python's own signal handling are never starved by an idle server.
        // (Mutex only because `detach` needs a Send closure; nothing else holds it.)
        let rx = std::sync::Mutex::new(rx);
        while let Ok(cmd) = py.detach(|| rx.lock().unwrap().recv()) {
            if let Cmd::Shutdown(reply) = cmd {
                let _ = reply.send(run_exit_handlers(py));
                return;
            }
            handle(py, &bridge, cmd);
        }
    });
}

fn init_bridge<'py>(
    py: Python<'py>,
    paths: &[PathBuf],
    config_path: &Path,
) -> Result<Bound<'py, PyAny>> {
    let sys = py.import("sys")?;
    let sys_path = sys
        .getattr("path")?
        .cast_into::<PyList>()
        .map_err(|e| anyhow!("{e}"))?;
    for p in paths.iter().rev() {
        sys_path.insert(0, p.to_string_lossy().as_ref())?;
    }
    // An embedded interpreter has an empty argv and reports this binary as
    // sys.executable; libraries that spawn `sys.executable -c ...` would run
    // the server. Point both at sane values (the interpreter we linked).
    sys.setattr("argv", PyList::new(py, ["qwen-tts"])?)?;
    let python = env!("POC_PYTHON");
    if std::path::Path::new(python).exists() {
        sys.setattr("executable", python)?;
    }
    let module = py.import("qwen_tts.bridge").map_err(|e| {
        let tb = e
            .traceback(py)
            .and_then(|t| t.format().ok())
            .unwrap_or_default();
        anyhow!(
            "importing qwen_tts.bridge failed: {e}\n{tb}\nsys.path={:?}",
            paths
        )
    })?;
    let bridge = module
        .getattr("Bridge")?
        .call1((config_path.to_string_lossy().as_ref(),))?;
    tracing::info!(
        "python bridge ready ({})",
        sys.getattr("version")?
            .extract::<String>()?
            .lines()
            .next()
            .unwrap_or("")
    );
    Ok(bridge)
}

/// `atexit._run_exitfuncs()`: what `Py_FinalizeEx` would run first, without
/// tearing down an interpreter whose Metal state belongs to another thread.
/// Handlers are removed as they run, so a second call is a no-op.
///
/// A library in the engine's stack installs a Python SIGINT handler, so a
/// ctrl-c reaches both tokio (shutdown) and Python, where it sits pending
/// and would surface as a `KeyboardInterrupt` on the next Python call —
/// including the ones below. Drain it first (`PyErr_CheckSignals` raises
/// and clears it; the error is the signal we already acted on), then ignore
/// SIGINT so a second ctrl-c during the handlers is dropped too.
fn run_exit_handlers(py: Python<'_>) -> Result<()> {
    let _ = py.check_signals();
    let signal = py.import("signal")?;
    signal.call_method1(
        "signal",
        (signal.getattr("SIGINT")?, signal.getattr("SIG_IGN")?),
    )?;
    py.import("atexit")?
        .call_method0("_run_exitfuncs")
        .map(|_| ())
        .map_err(pyerr(py))
}

fn handle(py: Python<'_>, bridge: &Bound<'_, PyAny>, cmd: Cmd) {
    match cmd {
        Cmd::Info(reply) => {
            let _ = reply.send(
                bridge
                    .call_method0("model_info")
                    .and_then(|o| to_json(py, &o))
                    .map_err(pyerr(py)),
            );
        }
        Cmd::Catalog(reply) => {
            let _ = reply.send(catalog(py, bridge).map_err(pyerr(py)));
        }
        Cmd::VoicePath(name, reply) => {
            let _ = reply.send(
                bridge
                    .call_method1("voice_path", (name,))
                    .and_then(|o| o.extract::<Option<String>>())
                    .map_err(pyerr(py)),
            );
        }
        Cmd::Transcribe(path, reply) => {
            let _ = reply.send(
                bridge
                    .call_method1("transcribe", (path,))
                    .and_then(|o| o.extract::<String>())
                    .map_err(pyerr(py)),
            );
        }
        Cmd::Shutdown(reply) => {
            // Handled in the thread loop; unreachable here.
            let _ = reply.send(Ok(()));
        }
        Cmd::Unload(reply) => {
            let _ = reply.send(bridge.call_method0("unload").map(|_| ()).map_err(pyerr(py)));
        }
        Cmd::BenchSentences(reply) => {
            let r = py
                .import("qwen_tts.bench")
                .and_then(|m| m.getattr("SENTENCES"))
                .and_then(|s| s.extract::<Vec<(String, String)>>())
                .map_err(pyerr(py));
            let _ = reply.send(r);
        }
        Cmd::Preload(reply) => {
            let r = bridge
                .call_method0("preload")
                .and_then(|o| to_json(py, &o))
                .map_err(pyerr(py));
            match &r {
                Ok(v) => tracing::info!("preload queued: {}", v["pending"]),
                Err(e) => tracing::warn!("preload failed: {e}"),
            }
            let _ = reply.send(r);
        }
        Cmd::Generate { tab, params, tx } => {
            if let Err(e) = generate(py, bridge, &tab, &params, &tx) {
                let msg = e.to_string();
                tracing::warn!("generate failed: {msg}");
                let _ = tx.blocking_send(StreamEvent::Error(msg));
            }
        }
    }
}

fn pyerr(py: Python<'_>) -> impl Fn(PyErr) -> anyhow::Error + '_ {
    move |e| {
        let tb = e
            .traceback(py)
            .and_then(|t| t.format().ok())
            .unwrap_or_default();
        if tb.is_empty() {
            anyhow!("{e}")
        } else {
            anyhow!("{e}\n{tb}")
        }
    }
}

fn catalog(py: Python<'_>, bridge: &Bound<'_, PyAny>) -> PyResult<Value> {
    Ok(json!({
        "voices": to_json(py, &bridge.call_method0("voices")?)?,
        "speakers": bridge.call_method0("speakers")?.extract::<Vec<String>>()?,
        "languages": bridge.call_method0("languages")?.extract::<Vec<String>>()?,
        "sizes": bridge.call_method0("sizes")?.extract::<Vec<String>>()?,
    }))
}

fn to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = py.import("json")?;
    let kw = PyDict::new(py);
    kw.set_item("default", py.eval(pyo3::ffi::c_str!("str"), None, None)?)?;
    let s: String = json.call_method("dumps", (obj,), Some(&kw))?.extract()?;
    serde_json::from_str(&s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

fn json_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => pyo3::types::PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(0.0).into_pyobject(py)?.into_any()
            }
        }
        Value::String(s) => s.into_pyobject(py)?.into_any(),
        Value::Array(a) => {
            let items: Vec<Bound<'py, PyAny>> = a
                .iter()
                .map(|x| json_to_py(py, x))
                .collect::<PyResult<_>>()?;
            PyList::new(py, items)?.into_any()
        }
        Value::Object(o) => {
            let d = PyDict::new(py);
            for (k, x) in o {
                d.set_item(k, json_to_py(py, x)?)?;
            }
            d.into_any()
        }
    })
}

/// Drive `Bridge.stream()` and forward chunks. Sets the bridge's stop event
/// when the receiver is gone so the model stops after the current chunk.
fn generate(
    py: Python<'_>,
    bridge: &Bound<'_, PyAny>,
    tab: &str,
    params: &Value,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<()> {
    let t0 = Instant::now();
    let stop = py.import("threading")?.call_method0("Event")?;
    let py_params = json_to_py(py, params)?;
    let model: String = bridge
        .call_method1("model_for", (tab, &py_params))
        .and_then(|o| o.extract())
        .unwrap_or_default();
    let iter = bridge
        .call_method1("stream", (tab, &py_params, &stop))
        .map_err(pyerr(py))?;
    let mut iter = iter.try_iter().map_err(pyerr(py))?;

    let mut started = false;
    let mut sample_rate = 24_000u32;
    let mut total_samples = 0usize;
    let mut ttfa_s: Option<f64> = None;
    let mut chunks = 0usize;
    let mut cancelled = false;

    loop {
        // `next()` blocks inside Python's queue.get(), which releases the GIL itself.
        let item = match iter.next() {
            None => break,
            Some(Ok(item)) => item,
            Some(Err(e)) => return Err(pyerr(py)(e)),
        };
        let sr: u32 = item.getattr("sample_rate")?.extract()?;
        let audio = item.getattr("audio")?;
        let arr = audio
            .extract::<numpy::PyReadonlyArray1<f32>>()
            .map_err(|e| anyhow!("chunk is not a float32 array: {e}"))?;
        let samples = f32_to_i16(arr.as_slice()?);
        drop(arr);
        if !started {
            started = true;
            sample_rate = sr;
            ttfa_s = Some(t0.elapsed().as_secs_f64());
            if tx
                .blocking_send(StreamEvent::Start {
                    sample_rate: sr,
                    model: model.clone(),
                })
                .is_err()
            {
                cancelled = true;
            }
        }
        total_samples += samples.len();
        chunks += 1;
        if !cancelled
            && py
                .detach(|| tx.blocking_send(StreamEvent::Audio { samples }))
                .is_err()
        {
            cancelled = true;
        }
        if cancelled {
            stop.call_method0("set")?;
        }
    }
    let gen_s = t0.elapsed().as_secs_f64();
    let audio_s = total_samples as f64 / sample_rate as f64;
    let timings = json!({
        "model": model,
        "tab": tab,
        "chars": params.get("text").and_then(|t| t.as_str()).map(|t| t.len()).unwrap_or(0),
        "ttfa_s": ttfa_s.map(round3),
        "gen_s": round3(gen_s),
        "audio_s": round3(audio_s),
        "rtf": if audio_s > 0.0 { Some(round3(gen_s / audio_s)) } else { None },
        "chunks": chunks,
        "cancelled": cancelled,
    });
    if !cancelled {
        let _ = tx.blocking_send(StreamEvent::Done { timings });
    }
    Ok(())
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}
