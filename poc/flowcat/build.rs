use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=MOONSHINE_LIB_DIR");
    qwen_tts_rpath();
    nemotron_native_link();

    // Keep the existing Whisper-only build independent of Moonshine's native
    // artifacts. Cargo exposes enabled features to build scripts this way.
    if env::var_os("CARGO_FEATURE_MOONSHINE").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let lib_dir = env::var_os("MOONSHINE_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../.deps/moonshine/v0.1.3/lib"));
    let lib_dir = lib_dir.canonicalize().unwrap_or_else(|error| {
        panic!(
            "Moonshine feature enabled but native library directory {} is unavailable: {error}. \
             Run the PoC setup or set MOONSHINE_LIB_DIR.",
            lib_dir.display()
        )
    });

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target OS set by Cargo");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("target arch set by Cargo");
    if !matches!(target_arch.as_str(), "x86_64" | "aarch64") {
        panic!("Moonshine v0.1.3 has no supported native artifact for {target_os}/{target_arch}");
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    match target_os.as_str() {
        "linux" => {
            require_file(&lib_dir, "libmoonshine.so");
            // libmoonshine.so has a $ORIGIN runpath for this co-located SONAME.
            require_file(&lib_dir, "libonnxruntime.so.1");
            println!("cargo:rustc-link-lib=dylib=moonshine");
            // Use an absolute development rpath so `make poc-up` can launch the
            // Cargo output directly. Packaged binaries should copy both shared
            // libraries beside the executable and use an $ORIGIN rpath instead.
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
        "macos" => {
            require_file(&lib_dir, "libmoonshine.a");
            println!("cargo:rustc-link-lib=static=moonshine");
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        _ => panic!("Moonshine PoC integration supports Linux and macOS, not {target_os}"),
    }
}

fn require_file(dir: &Path, name: &str) {
    let path = dir.join(name);
    if !path.is_file() {
        panic!(
            "Moonshine feature enabled but {} is missing. Run the PoC setup or set \
             MOONSHINE_LIB_DIR to the extracted v0.1.3 release lib directory.",
            path.display()
        );
    }
    println!("cargo:rerun-if-changed={}", path.display());
}

/// `qwen-tts`: the binary embeds Python (poc-qwen-streaming's PyO3 engine).
/// pyo3 links against the interpreter named by PYO3_PYTHON but emits no rpath,
/// and a dependency's `cargo:rustc-link-arg` is not transitive, so add
/// libpython's directory to this package's link line (binary and tests) here —
/// mirrors poc-qwen-streaming/build.rs. Other builds emit nothing.
fn qwen_tts_rpath() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    if env::var_os("CARGO_FEATURE_QWEN_TTS").is_none() {
        return;
    }
    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = std::process::Command::new(&python)
        .args(["-c", "import sysconfig;print(sysconfig.get_config_var('LIBDIR') or '')"])
        .output();
    if let Ok(out) = out {
        let libdir = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !libdir.is_empty() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
        }
    }
}

/// `nemotron-native`: link NeMo-Speech.cpp's C library (prebuilt dylibs from
/// setup_nemotron.sh) and add its directory to the rpath — the ggml backend
/// dylibs it depends on live beside it.
fn nemotron_native_link() {
    println!("cargo:rerun-if-env-changed=NEMO_SPEECH_LIB_DIR");
    if env::var_os("CARGO_FEATURE_NEMOTRON_NATIVE").is_none() {
        return;
    }
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let lib_dir = env::var_os("NEMO_SPEECH_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../.deps/nemo-speech/v0.1.0/lib"));
    let lib_dir = lib_dir.canonicalize().unwrap_or_else(|error| {
        panic!(
            "nemotron-native enabled but {} is unavailable: {error}. Run ./scripts/setup_nemotron.sh or set NEMO_SPEECH_LIB_DIR.",
            lib_dir.display()
        )
    });
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target OS set by Cargo");
    let lib_name = match target_os.as_str() {
        "macos" => "libnemo_speech_asr_c.dylib",
        "linux" => "libnemo_speech_asr_c.so",
        other => panic!("nemotron-native supports macOS and Linux, not {other}"),
    };
    require_file(&lib_dir, lib_name);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=nemo_speech_asr_c");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
