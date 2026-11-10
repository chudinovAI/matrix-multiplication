import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import sys
import unittest
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpumat import CpuMatmul


LIBRARY = Path(__file__).resolve().parents[1] / "build" / "cpumat.dylib"
SUPPORTED = platform.system() == "Darwin" and platform.machine() == "arm64" and LIBRARY.is_file()


def unaligned_matrix(shape, rng):
    size = int(np.prod(shape))
    storage = np.empty(size * 4 + 1, dtype=np.uint8)
    matrix = np.ndarray(shape, dtype=np.float32, buffer=storage, offset=1)
    matrix[:] = rng.standard_normal(shape)
    return matrix


@unittest.skipUnless(SUPPORTED, "requires Apple Silicon and `make cpu`")
class CpuMatmulTests(unittest.TestCase):
    def test_rectangular_edges_long_k_and_threads(self):
        rng = np.random.default_rng(627)
        shapes = [(1, 1, 1), (3, 17, 7), (5, 9, 11), (129, 257, 131),
                  (261, 513, 17), (11, 1025, 9)]
        for threads in (1, 3, 1000):
            with CpuMatmul(threads=threads) as cpu:
                for implementation in ("cpu_neon", "accelerate"):
                    for m, k, n in shapes:
                        with self.subTest(threads=threads, implementation=implementation, shape=(m, k, n)):
                            a = rng.standard_normal((m, k)).astype(np.float32)
                            b = rng.standard_normal((k, n)).astype(np.float32)
                            expected = a.astype(np.float64) @ b.astype(np.float64)
                            with cpu.prepare(implementation, a, b) as plan:
                                for _ in range(2):
                                    plan.run()
                                    plan.synchronize()
                                    np.testing.assert_allclose(plan.numpy(), expected, rtol=5e-4, atol=5e-4)

    def test_unaligned_contiguous_inputs(self):
        rng = np.random.default_rng(45)
        a, b = unaligned_matrix((7, 19), rng), unaligned_matrix((19, 13), rng)
        self.assertTrue(a.flags.c_contiguous and b.flags.c_contiguous)
        self.assertNotEqual(a.ctypes.data % 4, 0)
        expected = a.astype(np.float64) @ b.astype(np.float64)
        with CpuMatmul() as cpu:
            for implementation in ("cpu_neon", "accelerate"):
                with self.subTest(implementation=implementation), cpu.prepare(implementation, a, b) as plan:
                    plan.run()
                    np.testing.assert_allclose(plan.numpy(), expected, rtol=5e-4, atol=5e-4)

    def test_shared_pool_across_concurrent_and_recreated_plans(self):
        rng = np.random.default_rng(33)
        inputs = [(rng.standard_normal((m, 23)).astype(np.float32),
                   rng.standard_normal((23, 17)).astype(np.float32)) for m in (129, 385)]
        expected = [a.astype(np.float64) @ b.astype(np.float64) for a, b in inputs]
        with CpuMatmul(threads=4) as cpu:
            for _ in range(3):
                with cpu.prepare("cpu_neon", *inputs[0]) as first, cpu.prepare("cpu_neon", *inputs[1]) as second:
                    def run_many(plan):
                        for _ in range(4):
                            plan.run()
                        return plan.numpy()
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        results = list(executor.map(run_many, (first, second)))
                    for result, reference in zip(results, expected):
                        np.testing.assert_allclose(result, reference, rtol=5e-4, atol=5e-4)

    def test_retention_owned_output_and_context_lifetime(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.full((3, 4), 2, dtype=np.float32)
        a_ref, b_ref = weakref.ref(a), weakref.ref(b)
        cpu = CpuMatmul()
        plan = cpu.prepare("accelerate", a, b)
        with self.assertRaises(RuntimeError):
            plan.numpy()
        del a, b
        gc.collect()
        self.assertIsNotNone(a_ref())
        self.assertIsNotNone(b_ref())
        plan.run()
        result = plan.numpy()
        self.assertTrue(result.flags.owndata and result.flags.c_contiguous)
        cpu.close()
        np.testing.assert_array_equal(result, np.full((2, 4), 6, dtype=np.float32))
        for operation in (plan.run, plan.synchronize, plan.numpy, plan.__enter__):
            with self.assertRaises(RuntimeError):
                operation()
        with self.assertRaises(RuntimeError):
            cpu.prepare("cpu_neon", np.ones((1, 1), np.float32), np.ones((1, 1), np.float32))
        plan.close()
        cpu.close()
        gc.collect()
        self.assertIsNone(a_ref())
        self.assertIsNone(b_ref())

    def test_invalid_inputs_and_closed_plan(self):
        a = np.ones((3, 4), dtype=np.float32)
        b = np.ones((4, 5), dtype=np.float32)
        with CpuMatmul() as cpu:
            invalid = [(a.astype(np.float64), b), (a.tolist(), b),
                       (a[:, ::2], b), (a.reshape(12), b),
                       (a, np.ones((3, 5), dtype=np.float32)),
                       (np.empty((0, 4), np.float32), b)]
            for x, y in invalid:
                with self.subTest(shape=getattr(x, "shape", None)), self.assertRaises((ValueError, TypeError)):
                    cpu.prepare("cpu_neon", x, y)
            with self.assertRaises(ValueError):
                cpu.prepare("cpu_neon", a, b, precision="tf32")
            with self.assertRaises(ValueError):
                cpu.prepare("unknown", a, b)
            plan = cpu.prepare("cpu_neon", a, b)
            plan.close()
            plan.close()
            for operation in (plan.run, plan.synchronize, plan.numpy):
                with self.assertRaises(RuntimeError):
                    operation()
        for threads in (0, -1, True, 2**31, 1.5):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                CpuMatmul(threads=threads)


if __name__ == "__main__":
    unittest.main()
