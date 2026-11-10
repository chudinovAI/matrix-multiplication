"""Precision contracts, operand validation and independent numerical references.

Numerical libraries are imported lazily so the CLI can configure BLAS first.
"""

from contextlib import contextmanager

PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
INPUT_DTYPES = {"fp32": "float32", "tf32": "float32", "fp16": "float16", "bf16": "bfloat16"}
PRECISION_IDS = {name: index for index, name in enumerate(PRECISIONS)}
CUDA_BACKENDS = ("tiled", "register", "warp", "warp_large", "pipeline", "wmma",
                 "cublas", "torch", "triton", "jax")
METAL_BACKENDS = ("metal_naive", "metal_tiled", "metal_simd", "metal_tensor",
                  "metal_tensor_morton", "mps", "torch_mps")
CPU_BACKENDS = ("numpy", "cpu_neon", "accelerate", "torch_cpu")
BACKEND_PRECISIONS = {
    "numpy": ("fp32",), "tiled": ("fp32",), "register": ("fp32",),
    "warp": ("fp32",), "warp_large": ("fp32",), "pipeline": ("fp32",),
    "wmma": ("tf32", "fp16", "bf16"),
    "cublas": PRECISIONS, "torch": PRECISIONS, "jax": PRECISIONS, "triton": PRECISIONS,
    "cpu_neon": ("fp32",), "accelerate": ("fp32",), "torch_cpu": ("fp32",),
    "metal_naive": ("fp32", "fp16"), "metal_tiled": ("fp32", "fp16"),
    "metal_simd": ("fp32", "fp16"), "mps": ("fp32", "fp16"), "torch_mps": ("fp32",),
    "metal_tensor": ("fp32", "fp16", "bf16"), "metal_tensor_morton": ("fp32", "fp16", "bf16"),
}
RTOL = 5e-4
ATOL = 5e-4


def unsupported_reason(backend: str, precision: str, capability: tuple[int, int] | None,
                       *, target: str = "cuda") -> str | None:
    import platform
    import sys

    if backend in CUDA_BACKENDS and target != "cuda":
        return f"{backend} is a CUDA backend"
    if backend in METAL_BACKENDS and target != "metal":
        return f"{backend} is a Metal backend"
    if backend in ("cpu_neon", "accelerate") and (sys.platform != "darwin" or platform.machine() != "arm64"):
        return f"{backend} requires macOS on Apple Silicon"
    if precision not in BACKEND_PRECISIONS[backend]:
        return f"{backend} supports {', '.join(BACKEND_PRECISIONS[backend])}, not {precision}"
    if capability is not None and backend in CUDA_BACKENDS:
        if backend in ("triton", "pipeline", "wmma") and capability < (8, 0):
            return f"{backend} requires NVIDIA compute capability 8.0 or newer"
        if precision in ("tf32", "bf16") and capability < (8, 0):
            return f"{precision} requires NVIDIA compute capability 8.0 or newer"
        if precision == "fp16" and capability < (5, 3):
            return "FP16 requires NVIDIA compute capability 5.3 or newer"
    return None


def quantize_inputs(values, precision: str, *, tf32_ties: str = "even"):
    """Return FP32 representations of rounded inputs; TF32 also supports RNA."""
    import numpy as np

    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision: {precision}")
    if tf32_ties not in ("even", "away") or (precision != "tf32" and tf32_ties != "even"):
        raise ValueError("TF32 tie policy must be even or away; other formats use even")
    values = np.asarray(values, dtype=np.float32)
    if precision == "fp32":
        return values
    if precision == "fp16":
        with np.errstate(over="ignore", invalid="ignore"):
            return values.astype(np.float16).astype(np.float32)
    drop = 13 if precision == "tf32" else 16
    bits = values.view(np.uint32)
    # Adding the retained low bit implements ties-to-even, for either sign.
    bias = np.uint32((1 << (drop - 1)) - 1) + ((bits >> drop) & np.uint32(1))
    if precision == "tf32" and tf32_ties == "away":
        bias = np.uint32(1 << (drop - 1))
    rounded = (bits + bias) & np.uint32(0xFFFFFFFF ^ ((1 << drop) - 1))
    # Keep infinities and NaNs (including their payloads) rather than rounding them.
    rounded = np.where((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000), bits, rounded)
    return rounded.astype(np.uint32, copy=False).view(np.float32)


def reference_product(a, b, precision: str, *, tf32_ties: str = "even"):
    import numpy as np

    left = quantize_inputs(a, precision, tf32_ties=tf32_ties)
    right = quantize_inputs(b, precision, tf32_ties=tf32_ties)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"Input quantization to {precision} produced nonfinite values")
    return left.astype(np.float64) @ right.astype(np.float64)


def _relative_error(difference, reference) -> float | None:
    import numpy as np

    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(difference))
    if denominator == 0:
        return 0.0 if numerator == 0 else None  # Relative error is undefined here.
    return numerator / denominator


class GemmReference:
    """Cache legal FP64 input models and an FP32 accumulation-error bound."""

    def __init__(self, a, b, precision: str, *, original=None):
        import math
        import numpy as np

        left = quantize_inputs(a, precision).astype(np.float64)
        right = quantize_inputs(b, precision).astype(np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Input quantization to {precision} produced nonfinite values")
        self.precision = precision
        self.original = original if original is not None else reference_product(a, b, "fp32")
        self.rounded = self.original if precision == "fp32" else left @ right
        self._references = {"rounded_inputs": self.rounded}
        magnitude_left, magnitude_right = np.abs(left), np.abs(right)
        if precision == "tf32":
            away_left = quantize_inputs(a, precision, tf32_ties="away").astype(np.float64)
            away_right = quantize_inputs(b, precision, tf32_ties="away").astype(np.float64)
            self._references["tf32_ties_away"] = away_left @ away_right
            self._references["original_fp32"] = self.original
            # One common upper bound covers all permitted TF32 input models.
            magnitude_left = np.maximum(np.maximum(magnitude_left, np.abs(away_left)), np.abs(a))
            magnitude_right = np.maximum(np.maximum(magnitude_right, np.abs(away_right)), np.abs(b))
        # A dot product with K FP32 accumulation steps has forward error bounded
        # by gamma_K * |A| |B|. Unlike a fixed atol, this handles cancellation.
        k = left.shape[1]
        # Full FP32 epsilon is conservative for hardware MMA accumulation,
        # whose internal rounding need not be nearest-even.
        unit_roundoff = float(np.finfo(np.float32).eps)
        ku = k * unit_roundoff
        gamma = ku / (1 - ku) if ku < 1 else math.expm1(k * math.log1p(unit_roundoff))
        self._roundoff_bound = gamma * (magnitude_left @ magnitude_right)
        self._bound_max = float(self._roundoff_bound.max())

    def verify(self, actual, label: str, *, allow_tf32_library: bool = False) -> dict:
        import numpy as np

        if actual.shape != self.rounded.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise RuntimeError(f"{label}: expected finite FP32 output of shape {self.rounded.shape}")
        result = actual.astype(np.float64)
        original_difference = result - self.original
        candidates = self._references.items() if allow_tf32_library and self.precision == "tf32" else (
            ("rounded_inputs", self.rounded),)
        best_error = float("inf")
        for name, reference in candidates:
            difference = result - reference
            allowance = ATOL + RTOL * np.abs(reference) + self._roundoff_bound
            scaled_error = float(np.max(np.abs(difference) / allowance))
            if scaled_error < best_error:
                best_error = scaled_error
                reference_name = name
                arithmetic_reference = reference
                arithmetic_difference = difference
        if best_error > 1:
            raise RuntimeError(f"{label}: incorrect GEMM (roundoff-aware tolerance ratio {best_error:.3g})")
        return {"max_abs_error": float(np.max(np.abs(original_difference))),
                "arithmetic_max_abs_error": float(np.max(np.abs(arithmetic_difference))),
                "max_tolerance_ratio": best_error, "arithmetic_reference": reference_name,
                "roundoff_bound_max": self._bound_max,
                "relative_frobenius_error": _relative_error(original_difference, self.original),
                "arithmetic_relative_error": _relative_error(arithmetic_difference, arithmetic_reference),
                "quantization_relative_error": _relative_error(arithmetic_reference - self.original, self.original)}


def validate_operands(a, b, out, precision: str, device: int) -> tuple[int, int, int]:
    """Validate the shared native/Triton row-major CUDA tensor contract once."""
    import torch

    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision: {precision}")
    input_dtype = getattr(torch, INPUT_DTYPES[precision])
    for name, tensor, dtype in (("a", a, input_dtype), ("b", b, input_dtype), ("out", out, torch.float32)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if tensor.device.type != "cuda" or tensor.device.index != device:
            raise ValueError(f"{name} must be on cuda:{device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or tuple(out.shape) != (m, n):
        raise ValueError("expected a=(m, k), b=(k, n), and out=(m, n)")
    if any(size <= 0 or size > 2**31 - 1 for size in (m, k, n)):
        raise ValueError("matrix dimensions must be positive and fit in a C int")
    begin = out.data_ptr()
    end = begin + out.numel() * out.element_size()
    for name, tensor in (("a", a), ("b", b)):
        address = tensor.data_ptr()
        if begin < address + tensor.numel() * tensor.element_size() and address < end:
            raise ValueError(f"out must not overlap {name}")
    return m, k, n


def validate_host_operands(a, b, precision: str, allowed_precisions=("fp32",)) -> tuple[int, int, int]:
    """Native CPU/Metal plans consume unchanged, contiguous FP32 host inputs."""
    import numpy as np

    if precision not in allowed_precisions:
        raise ValueError(f"Supported precisions: {', '.join(allowed_precisions)}")
    for name, value in (("a", a), ("b", b)):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise TypeError(f"{name} must be a float32 NumPy array")
        if value.ndim != 2 or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be a contiguous two-dimensional matrix")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("expected a=(m, k) and b=(k, n)")
    if any(size <= 0 or size > 2**31 - 1 for size in (m, k, n)):
        raise ValueError("matrix dimensions must be positive and fit in a C int")
    return m, k, n


@contextmanager
def torch_precision(torch, precision: str):
    """Restore the caller's global policy, including on failed validation/timing."""
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision: {precision}")
    backend = torch.backends.cuda.matmul
    previous = backend.fp32_precision
    fp16_reduction = backend.allow_fp16_reduced_precision_reduction
    bf16_reduction = backend.allow_bf16_reduced_precision_reduction
    try:
        backend.fp32_precision = "tf32" if precision == "tf32" else "ieee"
        backend.allow_fp16_reduced_precision_reduction = False
        backend.allow_bf16_reduced_precision_reduction = False
        yield
    finally:
        backend.fp32_precision = previous
        backend.allow_fp16_reduced_precision_reduction = fp16_reduction
        backend.allow_bf16_reduced_precision_reduction = bf16_reduction
