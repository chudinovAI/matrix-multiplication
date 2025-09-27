import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
from gpumat import GPUMatrix
import mplcyberpunk


def benchmark(N: int):
    # print(N, N)

    A_np = np.random.rand(N, N).astype(np.float32)
    B_np = np.random.rand(N, N).astype(np.float32)

    start = time.perf_counter()
    A_np @ B_np
    end = time.perf_counter()
    t_numpy = end - start
    # print(f"numpy: {t_numpy:.4f} sec")

    GA, GB = GPUMatrix(A_np), GPUMatrix(B_np)
    torch.cuda.synchronize()
    start = time.perf_counter()
    GA @ GB
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_kernel = end - start
    # print(f"kernel: {t_kernel:.4f} sec")

    A_torch, B_torch = torch.from_numpy(A_np).cuda(), torch.from_numpy(B_np).cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    A_torch @ B_torch
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_torch = end - start
    # print(f"torch: {t_torch:.4f} sec")
    # print(f"Time diff: {t_torch - t_kernel:.4f} sec")

    return t_numpy, t_kernel, t_torch


if __name__ == "__main__":
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8196]
    times_numpy = []
    times_kernel = []
    times_torch = []

    for n in sizes:
        t_np, t_kernel, t_torch = benchmark(n)
        times_numpy.append(t_np)
        times_kernel.append(t_kernel)
        times_torch.append(t_torch)

    plt.style.use("cyberpunk")
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times_numpy, marker="o", label="numpy")
    plt.plot(sizes, times_kernel, marker="o", label="kernel")
    plt.plot(sizes, times_torch, marker="o", label="torch")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime scaling")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()

    mplcyberpunk.add_glow_effects()
    plt.tight_layout()

    output_dir = Path("result")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "runtime.png")
