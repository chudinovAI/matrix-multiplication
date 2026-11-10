"""Autotuned, row-major CUDA GEMM with frozen per-plan launch configurations."""

import threading

import torch
import triton
import triton.language as tl

from precision import PRECISION_IDS, unsupported_reason, validate_operands


def _config(m, n, k, warps, stages):
    return triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k},
                         num_warps=warps, num_stages=stages)


# IEEE FP32 uses SIMT arithmetic rather than tensor cores; large accumulator
# tiles that are useful for tensor cores can spill badly in that mode.
_SIMT_CONFIGS = [
    _config(16, 32, 32, 4, 2),
    _config(32, 32, 32, 4, 3),
    _config(32, 64, 16, 4, 2),
    _config(32, 64, 32, 8, 3),
]
_TENSOR_CONFIGS = [
    _config(32, 32, 32, 4, 2),
    _config(64, 64, 32, 4, 3),
    _config(128, 64, 32, 4, 3),
    _config(128, 128, 32, 8, 3),
    _config(128, 256, 32, 8, 3),
    _config(64, 128, 64, 8, 4),
]


def _prune_configs(configs, named_args, **kwargs):
    del configs, kwargs
    return _SIMT_CONFIGS if named_args["PRECISION"] == PRECISION_IDS["fp32"] else _TENSOR_CONFIGS


@triton.jit
def _round_tf32_rne(value):
    # cvt.rn.tf32.f32 requires sm90; integer RNE works on Ampere as well.
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    special = (bits & 0x7F800000) == 0x7F800000
    return tl.where(special, bits, rounded).to(tl.float32, bitcast=True)


@triton.autotune(
    configs=_SIMT_CONFIGS + _TENSOR_CONFIGS,
    key=["M", "N", "K", "PRECISION", "DEVICE"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit(do_not_specialize_on_alignment=["A", "B", "C"])
def _matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            PRECISION: tl.constexpr, DEVICE: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Group output tiles by rows to reuse B tiles in L2.
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_size = 8 * num_n
    group_id = pid // group_size
    first_m = group_id * 8
    group_m = tl.minimum(num_m - first_m, 8)
    pid_m = first_m + (pid % group_size) % group_m
    pid_n = (pid % group_size) // group_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    # Dimensions fit C ints, but their products and pointer offsets need not.
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)
    accumulator = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for start in range(0, tl.cdiv(K, BLOCK_K)):
        ks = start * BLOCK_K + inner
        ks64 = ks.to(tl.int64)
        a = tl.load(A + rows64[:, None] * K + ks64[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(B + ks64[:, None] * N + cols64[None, :],
                    mask=(ks[:, None] < K) & (cols[None, :] < N), other=0)
        if PRECISION == 0:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        elif PRECISION == 1:
            # Explicit RNE makes the subsequent TF32 conversion idempotent.
            a = _round_tf32_rne(a)
            b = _round_tf32_rne(b)
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        else:
            accumulator = tl.dot(a, b, accumulator)
    tl.store(C + rows64[:, None] * N + cols64[None, :], accumulator,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


# Triton's autotuner exposes the last winner through mutable shared state.
# Serialize setup and copy the winner before another prepare can replace it.
_tuning_lock = threading.Lock()


class _Plan:
    def __init__(self, a, b, out, precision, device, shape, config):
        self.out = out
        self._precision = precision
        self._device = device
        self._shape = shape
        self._config = (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"],
                        config.kwargs["BLOCK_K"], config.num_stages, config.num_warps)
        self._run = self.bind(a, b, out)

    @property
    def config(self):
        m, n, k, stages, warps = self._config
        return {"block_m": m, "block_n": n, "block_k": k,
                "num_stages": stages, "num_warps": warps}

    def bind(self, a, b, out):
        shape = validate_operands(a, b, out, self._precision, self._device)
        if shape != self._shape:
            raise ValueError(f"bound shape {shape} does not match prepared shape {self._shape}")
        stream = torch.cuda.current_stream(self._device)
        for tensor in (a, b, out):
            tensor.record_stream(stream)
        m, k, n = shape
        block_m, block_n, block_k, stages, warps = self._config
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        launch = _matmul.fn[grid]
        precision_id = PRECISION_IDS[self._precision]
        device = self._device

        def run():
            # The closure retains the storage and stream. Triton JIT launches on
            # the current stream, so restore the stream captured at binding.
            with torch.cuda.stream(stream):
                launch(a, b, out, m, n, k, precision_id, device,
                                 BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                                 num_warps=warps, num_ctas=1, num_stages=stages)
            return out

        return run

    def run(self):
        return self._run()


def prepare(a, b, out, precision: str):
    """Tune and execute once, then return a plan bound to the current stream."""
    if not isinstance(a, torch.Tensor):
        raise TypeError("a must be a torch.Tensor")
    if a.device.type != "cuda":
        raise ValueError("a must be a CUDA tensor")
    device = a.device.index
    shape = validate_operands(a, b, out, precision, device)
    reason = unsupported_reason("triton", precision, torch.cuda.get_device_capability(device))
    if reason:
        raise ValueError(reason)
    m, k, n = shape
    stream = torch.cuda.current_stream(device)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    with _tuning_lock, torch.cuda.device(device), torch.cuda.stream(stream):
        _matmul[grid](a, b, out, m, n, k, PRECISION_IDS[precision], device)
        config = _matmul.best_config
        # Ensure setup includes the selected first execution, and that no
        # autotuning work remains when callers begin CUDA graph capture.
        stream.synchronize()
        return _Plan(a, b, out, precision, device, shape, config)
