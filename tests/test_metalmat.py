import math
from pathlib import Path
import platform
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metalmat import MetalMatmul
from precision import GemmReference, reference_product


ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = platform.system() == "Darwin" and (ROOT / "build" / "metal_matmul.dylib").is_file()


@unittest.skipUnless(AVAILABLE, "requires macOS and the Metal library built with make metal")
class MetalMatmulTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = MetalMatmul()
        cls.implementations = ("metal_naive", "metal_tiled", "metal_simd", "mps")
        if cls.context.info["tensor_ops"]:
            cls.implementations += ("metal_tensor", "metal_tensor_morton")

    @classmethod
    def tearDownClass(cls):
        cls.context.close()

    def test_all_kernels_against_independent_fp64_reference(self):
        rng = np.random.default_rng(8704)
        for m, k, n in ((1, 1, 1), (7, 13, 11), (35, 37, 67), (17, 65, 19),
                        (65, 17, 65), (513, 129, 577), (3, 2049, 5)):
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n)).astype(np.float32)
            for impl in self.implementations:
                precisions = (("fp32", "fp16", "bf16") if impl.startswith("metal_tensor") else
                              ("fp32", "fp16"))
                for precision in precisions:
                    with self.subTest(shape=(m, k, n), implementation=impl, precision=precision):
                        with self.context.prepare(impl, a, b, precision) as plan:
                            plan.run()
                            plan.synchronize()
                            actual = plan.numpy()
                            GemmReference(a, b, precision).verify(actual, impl)

    def test_half_round_to_nearest_even(self):
        # Consecutive FP16 values straddle halfway points with even/odd low bits.
        a = np.array([[1.0 + 2**-11, 1.0 + 3 * 2**-11, -1.0 - 2**-11,
                       -1.0 - 3 * 2**-11, 2**-24, 2**-25]], dtype=np.float32)
        b = np.eye(a.shape[1], dtype=np.float32)
        expected = reference_product(a, b, "fp16").astype(np.float32)
        for impl in self.implementations:
            with self.subTest(implementation=impl), self.context.prepare(impl, a, b, "fp16") as plan:
                plan.run()
                np.testing.assert_array_equal(plan.numpy(), expected)

    def test_repeated_async_runs_and_owned_lifetimes(self):
        a = np.arange(21, dtype=np.float32).reshape(3, 7)
        b = np.arange(35, dtype=np.float32).reshape(7, 5)
        expected = reference_product(a, b, "fp32")
        for impl in self.implementations:
            with self.subTest(implementation=impl):
                context = MetalMatmul()
                plan = context.prepare(impl, a, b)
                context.close()
                for _ in range(5):
                    plan.run()
                plan.synchronize()
                first = plan.numpy()
                plan.run()
                second = plan.numpy()
                plan.close()
                self.assertTrue(first.flags.owndata and first.flags.c_contiguous)
                np.testing.assert_allclose(first, expected)
                np.testing.assert_array_equal(second, first)
                first.fill(-123)
                np.testing.assert_allclose(second, expected)
                with self.assertRaises(RuntimeError):
                    plan.run()
                with self.assertRaises(RuntimeError):
                    plan.numpy()

    def test_invalid_inputs_precisions_and_closed_context(self):
        a = np.ones((3, 4), dtype=np.float32)
        b = np.ones((4, 5), dtype=np.float32)
        cases = ((a.astype(np.float64), b, TypeError),
                 (a, b.astype(np.float16), TypeError),
                 (a[:, ::2], b, ValueError),
                 (a, b.reshape(-1), ValueError),
                 (a, b[:3], ValueError),
                 (np.empty((0, 4), np.float32), b, ValueError))
        for left, right, error in cases:
            with self.subTest(left=left.shape, right=right.shape, error=error):
                with self.assertRaises(error):
                    self.context.prepare("metal_naive", left, right)
        for impl, precision in (("mps", "bf16"), ("metal_simd", "tf32"), ("metal_tiled", "bf16")):
            with self.assertRaises(ValueError):
                self.context.prepare(impl, a, b, precision)
        with self.assertRaises(ValueError):
            self.context.prepare("unknown", a, b)
        context = MetalMatmul()
        context.close()
        with self.assertRaises(RuntimeError):
            context.prepare("metal_naive", a, b)

    def test_bfloat16_ties_use_rne_with_fp32_output(self):
        if not self.context.info["tensor_ops"]:
            self.skipTest("requires Metal 4")
        a = np.array([[1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8]], dtype=np.float32)
        b = np.eye(4, dtype=np.float32)
        expected = reference_product(a, b, "bf16").astype(np.float32)
        for impl in ("metal_tensor", "metal_tensor_morton"):
            with self.subTest(implementation=impl), self.context.prepare(impl, a, b, "bf16") as plan:
                plan.run()
                np.testing.assert_array_equal(plan.numpy(), expected)

    def test_device_timestamps_and_mps_padded_rows(self):
        rng = np.random.default_rng(498)
        a = rng.normal(size=(9, 13)).astype(np.float32)
        b = rng.normal(size=(13, 7)).astype(np.float32)
        for impl in self.implementations:
            with self.subTest(implementation=impl), self.context.prepare(impl, a, b) as plan:
                samples = plan.device_samples(warmup=2, repeats=3, batch=4)
                self.assertEqual(len(samples), 3)
                self.assertTrue(all(math.isfinite(sample) and sample > 0 for sample in samples))
                GemmReference(a, b, "fp32").verify(plan.numpy(), impl)
                for args in ((-1, 1, 1), (0, 0, 1), (0, 1, 0), (True, 1, 1)):
                    with self.assertRaises(ValueError):
                        plan.device_samples(*args)

if __name__ == "__main__":
    unittest.main()
