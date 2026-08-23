import pytest
import torch

from poc_tts.engine_flash import (
    UnsupportedDtypeError,
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
