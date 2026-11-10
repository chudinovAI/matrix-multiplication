"""Compare custom GEMM kernels and libraries on CUDA, Apple Metal or CPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from benchmark_backends import BackendRuntime, DEFAULT_BACKENDS, DEVICE_BATCH, mode_supported, resolve_target
from precision import (ATOL, RTOL, CPU_BACKENDS, CUDA_BACKENDS, METAL_BACKENDS,
                       GemmReference, INPUT_DTYPES, PRECISIONS, reference_product, unsupported_reason)

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = (*CPU_BACKENDS, *CUDA_BACKENDS, *METAL_BACKENDS)
BACKEND_WIDTH = max(map(len, BACKENDS))
MODES = ("device", "resident", "end_to_end")
PLOT_FILES = {"device": "runtime.png", "resident": "resident.png", "end_to_end": "end_to_end.png"}


def shape_arg(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
        if len(shape) != 3 or any(n < 1 or n > 2**31 - 1 for n in shape):
            raise ValueError
        return shape
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected positive MxKxN, e.g. 31x33x65") from error


def positive_int(value: str) -> int:
    number = int(value)
    if not 0 < number <= 2**31 - 1:
        raise argparse.ArgumentTypeError("Expected a positive 32-bit integer")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument("--sizes", nargs="+", type=positive_int, help="Square matrix sizes")
    shapes.add_argument("--shapes", nargs="+", type=shape_arg, help="Rectangular MxKxN problems")
    parser.add_argument("--platform", dest="target", choices=("auto", "cuda", "metal", "cpu"), default="auto")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, help="Default: platform-specific custom kernels and baselines")
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=["fp32"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=21)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=positive_int, default=1)
    parser.add_argument("--device", type=int, default=0, help="CUDA or native Metal device ordinal")
    parser.add_argument("--library", type=Path, help="Override CUDA/Metal library, or CPU library with --platform cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "result")
    parser.add_argument("--no-plot", action="store_true", help="Write JSON/CSV without matplotlib")
    args = parser.parse_args()
    if args.warmup < 0 or args.device < 0 or args.seed < 0:
        parser.error("warmup, device and seed must be nonnegative")
    args.shapes = args.shapes or [(n, n, n) for n in (args.sizes or [64, 128, 256, 512, 1024, 2048, 4096])]
    for name in ("shapes", "precisions", "modes"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    return args


def configure_environment(threads: int) -> None:
    # Set before importing numerical libraries, including PyTorch's CPU backend.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["PYTORCH_MPS_FAST_MATH"] = "0"


def measure_wall(call, synchronize, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        call()
        synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter_ns()
        call()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def result_row(backend, precision, mode, shape, samples, errors, setup_ms, case) -> dict:
    import numpy as np

    m, k, n = shape
    if not samples or not all(np.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError(f"{backend}/{mode}: invalid timing samples")
    median = statistics.median(samples)
    p10, p90 = np.percentile(samples, [10, 90])
    return {"backend": backend, "execution_device": case.execution_device, "precision": precision,
            "input_dtype": INPUT_DTYPES[precision], "output_dtype": "float32", "mode": mode,
            "m": m, "k": k, "n": n, "median_ms": median, "p10_ms": float(p10), "p90_ms": float(p90),
            "tflops": 2 * m * k * n / (median * 1e9), "setup_ms": setup_ms,
            "kernel_config": case.config, **errors, "samples_ms": samples}


def plot_results(report: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows, metadata = report["measurements"], report["metadata"]
    shapes = metadata["shapes_mkn"]
    square = all(m == k == n for m, k, n in shapes)
    names = [name for name in metadata["backends"] if any(row["backend"] == name for row in rows)]
    palette = ("#94a3b8", "#fb923c", "#c084fc", "#34d399", "#60a5fa", "#facc15", "#f87171",
               "#22d3ee", "#e879f9", "#a3e635", "#fda4af", "#818cf8", "#2dd4bf", "#fdba74", "#e2e8f0")
    colors = {name: palette[index % len(palette)] for index, name in enumerate(names)}
    device_title = "CUDA graph" if metadata["target"] == "cuda" else "Metal command-buffer GPU timestamps"
    titles = {"device": f"Device execution · {device_title}", "resident": "Resident inputs · synchronized call",
              "end_to_end": "End-to-end · original host inputs & host output"}
    precision_titles = {"fp32": "FP32 · IEEE", "tf32": "TF32 · FP32 inputs", "fp16": "FP16 inputs → FP32", "bf16": "BF16 inputs → FP32"}
    style = {"figure.facecolor": "#101827", "axes.facecolor": "#101827", "text.color": "#e2e8f0",
             "axes.labelcolor": "#cbd5e1", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
             "axes.edgecolor": "#334155", "font.size": 10, "axes.titlesize": 12,
             "grid.color": "#334155", "grid.alpha": 0.55}
    with plt.rc_context(style):
        for mode in MODES:
            precisions = [p for p in PRECISIONS if any(r["mode"] == mode and r["precision"] == p for r in rows)]
            if not precisions:
                continue
            columns = min(2, len(precisions))
            count_rows = (len(precisions) + columns - 1) // columns
            mode_names = [name for name in names if any(r["mode"] == mode and r["backend"] == name for r in rows)]
            legend_columns = min(4 if columns == 1 else 6, len(mode_names))
            legend_rows = (len(mode_names) + legend_columns - 1) // legend_columns
            footer = 0.09 + 0.035 * legend_rows
            fig, axes = plt.subplots(count_rows, columns, figsize=(6.4 * columns, 4.7 * count_rows + 1.0 + legend_rows * 0.2), squeeze=False)
            handles = {}
            for ax, precision in zip(axes.flat, precisions):
                for name in names:
                    points = [r for r in rows if r["backend"] == name and r["mode"] == mode and r["precision"] == precision]
                    if not points:
                        continue
                    if square:
                        points.sort(key=lambda r: r["n"])
                        x = [r["n"] for r in points]
                    else:
                        points.sort(key=lambda r: shapes.index([r["m"], r["k"], r["n"]]))
                        x = [shapes.index([r["m"], r["k"], r["n"]]) for r in points]
                    line, = ax.plot(x, [r["median_ms"] for r in points], marker="o", markersize=4,
                                    linewidth=1.8, color=colors[name], label=name)
                    ax.fill_between(x, [r["p10_ms"] for r in points], [r["p90_ms"] for r in points], color=colors[name], alpha=0.12)
                    handles[name] = line
                if square:
                    ax.set_xscale("log", base=2)
                    ticks = sorted({n for _, _, n in shapes})
                    ax.set_xticks(ticks, [str(n) for n in ticks], rotation=35)
                    ax.set_xlabel("Matrix size N (N × N)")
                else:
                    ax.set_xticks(np.arange(len(shapes)), [f"{m}×{k}×{n}" for m, k, n in shapes], rotation=35, ha="right")
                    ax.set_xlabel("Problem shape M × K × N")
                ax.set_yscale("log")
                ax.set_ylabel("Latency (ms) · lower is better")
                ax.set_title(precision_titles[precision], pad=12)
                ax.grid(True, which="major", linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
            for ax in list(axes.flat)[len(precisions):]:
                ax.set_visible(False)
            hardware = metadata.get("gpu", {}).get("name", metadata["cpu"] or "CPU")
            fig.suptitle(titles[mode], fontsize=14, y=0.99)
            fig.text(0.5, 0.925 if count_rows == 1 else 0.955,
                     f"{hardware} · CPU threads requested: {metadata['cpu_threads']}\nFP32 output · median & p10–p90 · {metadata['repeats']} samples · setup excluded",
                     ha="center", va="top", color="#94a3b8", fontsize=10)
            ordered = [name for name in names if name in handles]
            fig.legend([handles[name] for name in ordered], ordered, loc="lower center", ncol=legend_columns,
                       frameon=False, bbox_to_anchor=(0.5, 0.005))
            fig.tight_layout(rect=(0, footer if count_rows == 1 else footer * 0.6, 1, 0.85 if count_rows == 1 else 0.92))
            fig.savefig(output / PLOT_FILES[mode], dpi=170, facecolor=fig.get_facecolor())
            plt.close(fig)


def write_report(report: dict, output: Path, no_plot: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = report["measurements"]
    with (output / "results.csv").open("w", newline="") as file:
        columns = [name for name in rows[0] if name != "samples_ms"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], sort_keys=True) if isinstance(row[key], dict) else row[key] for key in columns})
    if not no_plot:
        plot_results(report, output)
    print(f"\nResults: {output.resolve()}")


def run(args: argparse.Namespace) -> None:
    args.target = resolve_target(args.target, args.backends)
    args.backends = list(dict.fromkeys(args.backends or DEFAULT_BACKENDS[args.target]))
    import numpy as np

    with BackendRuntime(args) as runtime:
        supported, unsupported = set(), []
        for name in args.backends:
            for precision in args.precisions:
                reason = (unsupported_reason(name, precision, runtime.capability, target=args.target)
                          or runtime.unavailable_reason(name))
                for mode in args.modes:
                    why = reason or (None if mode_supported(name, mode) else f"{name} has no device timestamp timing")
                    if why:
                        unsupported.append({"backend": name, "precision": precision, "mode": mode, "reason": why})
                    else:
                        supported.add((name, precision, mode))
        if not supported:
            raise RuntimeError("No supported combinations on this host: " + "; ".join(sorted({r["reason"] for r in unsupported})))
        runtime.initialize(supported)
        metadata = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                    "target": args.target, "cpu": platform.processor(), "numpy": np.__version__, "cpu_threads": args.cpu_threads,
                    "blas": np.show_config(mode="dicts"), "precisions": args.precisions,
                    "input_dtypes": {p: INPUT_DTYPES[p] for p in args.precisions}, "output_dtype": "float32",
                    "rounding": "FP16/BF16 and custom/Triton TF32 use RNE; library TF32 may use ties-away or IEEE FP32",
                    "reduced_precision_reduction": False, "mps_fast_math": False,
                    "reference": "NumPy FP64 of precision-rounded inputs; total error separately uses original FP32 inputs",
                    "tf32_policy": "Library validation accepts RNE, ties-away, or higher-precision IEEE FP32; custom WMMA and Triton use RNE",
                    "rtol": RTOL, "atol": ATOL, "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
                    "arithmetic_bound": "atol + rtol*abs(reference) + gamma_K*(abs(A)@abs(B)); conservative FP32 u=2^-23; TF32 uses upper envelope of legal inputs",
                    "shapes_mkn": [list(shape) for shape in args.shapes], "backends": args.backends,
                    "modes": args.modes, "device_batch": DEVICE_BATCH,
                    "setup_timing": "wall time for input placement/conversion/packing, output allocation, preparation, first dispatch and completion; includes JIT/autotuning, excludes reusable backend initialization and validation readback",
                    "timing": {"device": "CUDA graph events or native Metal command-buffer GPU timestamps, averaged over device_batch dispatches; no host transfers or Python enqueue gaps",
                               "resident": "synchronized wall time with converted/packed resident inputs and reused output; includes dispatch",
                               "end_to_end": "wall time from original FP32 host inputs to owned FP32 host output; matrix allocations, casts, packing, placement/transfers and readback included; reusable backend context/pipelines retained"},
                    **runtime.metadata}
        report = {"schema_version": 3, "metadata": metadata, "measurements": [], "unsupported_cases": unsupported}
        hardware = metadata.get("gpu", {}).get("name", "CPU only")
        print(f"GEMM / FP32 output | {args.target} | {hardware} | CPU threads requested: {args.cpu_threads}")
        for name, precision, reason in sorted({(r['backend'], r['precision'], r['reason']) for r in unsupported}):
            print(f"skip {name}/{precision}: {reason}")
        print(f"{'backend':<{BACKEND_WIDTH}} {'prec':<5} {'M×K×N':<19} {'mode':<12} {'median ms':>11} {'p10':>10} {'p90':>10} {'TFLOP/s':>9} {'rel. error':>12}")
        rng = np.random.default_rng(args.seed)
        for shape in args.shapes:
            m, k, n = shape
            a = rng.standard_normal((m, k), dtype=np.float32)
            b = rng.standard_normal((k, n), dtype=np.float32)
            original_reference = reference_product(a, b, "fp32")
            for precision in args.precisions:
                active = [name for name in args.backends if any((name, precision, mode) in supported for mode in args.modes)]
                if not active:
                    continue
                reference = GemmReference(a, b, precision, original=original_reference)
                for name in active:
                    label = f"{name}/{precision} {m}×{k}×{n}"
                    with runtime.policy(name, precision):
                        start = time.perf_counter_ns()
                        with runtime.prepare(name, precision, a, b) as case:
                            case.run()
                            case.synchronize()
                            setup_ms = (time.perf_counter_ns() - start) / 1e6
                            allow_library = precision == "tf32" and name in ("cublas", "torch", "jax")
                            errors = reference.verify(case.output(), label, allow_tf32_library=allow_library)
                            reference.verify(case.end_to_end(), label + "/end_to_end", allow_tf32_library=allow_library)
                            suffix = "" if case.config is None else f" · config {json.dumps(case.config, sort_keys=True)}"
                            print(f"setup {label}: {setup_ms:.3f} ms{suffix}", flush=True)
                            for mode in args.modes:
                                if (name, precision, mode) not in supported:
                                    continue
                                samples = (case.device_samples(args.warmup, args.repeats) if mode == "device" else
                                           measure_wall(case.run if mode == "resident" else case.end_to_end,
                                                        case.synchronize, args.warmup, args.repeats))
                                row = result_row(name, precision, mode, shape, samples, errors, setup_ms, case)
                                report["measurements"].append(row)
                                relative = "undefined" if row["relative_frobenius_error"] is None else f"{row['relative_frobenius_error']:.2e}"
                                problem = f"{m}×{k}×{n}"
                                print(f"{name:<{BACKEND_WIDTH}} {precision:<5} {problem:<19} {mode:<12} {row['median_ms']:11.4f} {row['p10_ms']:10.4f} {row['p90_ms']:10.4f} {row['tflops']:9.3f} {relative:>12}", flush=True)
                        del case
        write_report(report, args.output, args.no_plot)


def main() -> int:
    args = arguments()
    configure_environment(args.cpu_threads)
    try:
        run(args)
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
