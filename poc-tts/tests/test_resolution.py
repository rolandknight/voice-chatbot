import pytest
import torch

import poc_tts.engine_flash as engine_flash
from poc_tts.engine_flash import (
    UnsupportedDtypeError,
    _bf16_supported,
    _flashinfer_available,
    resolve_backend,
    resolve_device,
    resolve_dtype,
)


def test_device_auto_prefers_cuda_when_available():
    assert resolve_device("auto", cuda_available=True) == "cuda"


def test_device_auto_falls_back_to_cpu():
    assert resolve_device("auto", cuda_available=False) == "cpu"


def test_device_explicit_cuda_without_cuda_is_an_error():
    with pytest.raises(ValueError, match="CUDA requested but not available"):
        resolve_device("cuda", cuda_available=False)


def test_dtype_auto_uses_bf16_when_gpu_supports_it():
    got = resolve_dtype("auto", device="cuda", bf16_supported=True)
    assert got is torch.bfloat16


def test_dtype_auto_falls_back_to_fp16_on_sm75():
    """The RTX 2060 is sm_75: bf16 is unsupported.

    chatterbox-flash defaults to bfloat16 unconditionally, which on Turing
    means emulated speeds or an outright failure. Auto must never pick it
    when the hardware says no. This is the single most likely way this PoC
    silently runs slow.
    """
    got = resolve_dtype("auto", device="cuda", bf16_supported=False)
    assert got is torch.float16


def test_dtype_auto_uses_fp32_on_cpu():
    got = resolve_dtype("auto", device="cpu", bf16_supported=False)
    assert got is torch.float32


def test_dtype_explicit_bf16_on_unsupported_gpu_raises():
    with pytest.raises(UnsupportedDtypeError, match="bfloat16"):
        resolve_dtype("bfloat16", device="cuda", bf16_supported=False)


def test_dtype_explicit_fp16_is_honoured():
    got = resolve_dtype("float16", device="cuda", bf16_supported=True)
    assert got is torch.float16


def test_backend_auto_prefers_flashinfer():
    assert resolve_backend("auto", flashinfer_available=True) == "flashinfer"


def test_backend_auto_falls_back_to_torch():
    assert resolve_backend("auto", flashinfer_available=False) == "torch"


def test_backend_auto_never_selects_mlx():
    assert resolve_backend("auto", flashinfer_available=False) != "mlx"


def test_backend_explicit_flashinfer_when_absent_raises():
    with pytest.raises(ValueError, match="flashinfer requested but not installed"):
        resolve_backend("flashinfer", flashinfer_available=False)


def test_backend_auto_on_cpu_never_selects_flashinfer():
    """flashinfer is CUDA-only. On this project's sm_75 box it IS importable,
    so an auto resolution that ignored device would hand a CPU engine a CUDA
    backend and fail at generation time."""
    got = resolve_backend("auto", flashinfer_available=True, device="cpu")
    assert got == "torch"


def test_backend_explicit_flashinfer_on_cpu_raises():
    with pytest.raises(ValueError, match="flashinfer requires a CUDA device"):
        resolve_backend("flashinfer", flashinfer_available=True, device="cpu")


def test_bf16_supported_rejects_sm75_emulation(monkeypatch):
    """torch.cuda.is_bf16_supported() defaults to including_emulation=True and
    returns True on sm_75, where bf16 is emulated and slow -- which silently
    defeated the auto-dtype guard. The honest call passes
    including_emulation=False explicitly, which real sm_75 hardware reports
    as False. This mock simulates exactly that split so the test runs on any
    machine, not just the sm_75 box."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported",
        lambda including_emulation=True: including_emulation,
    )
    assert _bf16_supported() is False


def test_flashinfer_unavailable_without_nvcc(monkeypatch):
    """flashinfer JIT-compiles its kernels on first use and needs nvcc. A
    package that imports cleanly but has no CUDA toolkit is not actually
    usable -- fail closed to torch SDPA rather than crash mid-generation."""
    monkeypatch.setattr(
        engine_flash.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(engine_flash.shutil, "which", lambda cmd: None)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(engine_flash.Path, "exists", lambda self: False)
    assert _flashinfer_available() is False


def test_flashinfer_unavailable_on_sm75_even_with_nvcc(monkeypatch):
    """flashinfer's prebuilt kernels select fp16 QK accumulation on compute
    capability < 8.0 (Turing and older) and the wheel is not built with
    FP16_QK_REDUCTION_SUPPORTED, so the JIT compile itself static-asserts and
    fails -- verified on an RTX 2060 with CUDA 12.4: nvcc ran, the JIT
    proceeded, and prefill.cuh failed with a static assertion demanding
    boost_math. A working toolkit is not enough; the card must be Ampere or
    newer."""
    monkeypatch.setattr(
        engine_flash.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(engine_flash.shutil, "which", lambda cmd: "/usr/bin/nvcc")
    assert _flashinfer_available() is False


def test_flashinfer_available_on_sm80_with_nvcc(monkeypatch):
    """The compute-capability guard must not simply disable flashinfer
    everywhere -- Ampere and newer (sm_80+) can run it."""
    monkeypatch.setattr(
        engine_flash.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(engine_flash.shutil, "which", lambda cmd: "/usr/bin/nvcc")
    assert _flashinfer_available() is True
