//! Add libpython's directory to the binary's rpath so the embedded interpreter
//! resolves at run time without DYLD_LIBRARY_PATH. pyo3 already links against
//! the interpreter named by PYO3_PYTHON (the Makefile points it at this crate's
//! .venv, mise Python 3.12); it only emits the link search path, not an rpath.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".into());
    // The interpreter path is also needed at run time: an embedded interpreter
    // reports the host binary as sys.executable, and libraries that spawn
    // `sys.executable -c ...` (tokenizers, multiprocessing) would run *us*.
    println!("cargo:rustc-env=POC_PYTHON={python}");
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
