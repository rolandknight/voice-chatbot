//! qwen-tts's build script adds libpython's directory to *its* link line, but
//! `cargo:rustc-link-arg` does not reach a dependent binary, so this binary
//! adds the rpath itself (same interpreter: PYO3_PYTHON, the crate's venv).

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .args([
            "-c",
            "import sysconfig;print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output();
    if let Ok(out) = out {
        let libdir = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !libdir.is_empty() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
        }
    }
}
