from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precision import GemmReference, quantize_inputs


class PrecisionTests(unittest.TestCase):
    def test_tf32_and_bf16_signed_ties_round_to_even(self):
        for precision, fraction_bits in (("tf32", 10), ("bf16", 7)):
            step = 2.0 ** -fraction_bits
            source = np.array([1 + step / 2, 1 + 3 * step / 2,
                               -1 - step / 2, -1 - 3 * step / 2], dtype=np.float32)
            expected = np.array([1, 1 + 2 * step, -1, -1 - 2 * step], dtype=np.float32)
            with self.subTest(precision=precision):
                np.testing.assert_array_equal(quantize_inputs(source, precision), expected)

    def test_signed_zero_and_nonfinite_values_are_not_reinterpreted(self):
        source = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
        for precision in ("tf32", "fp16", "bf16"):
            with self.subTest(precision=precision):
                rounded = quantize_inputs(source, precision)
                np.testing.assert_array_equal(rounded[:4].view(np.uint32), source[:4].view(np.uint32))
                self.assertTrue(np.isnan(rounded[4]))

    def test_fp16_subnormal_rounding_and_overflow(self):
        source = np.array([2.0**-24, 2.0**-25, -2.0**-25], dtype=np.float32)
        expected = np.array([2.0**-24, 0.0, -0.0], dtype=np.float32)
        np.testing.assert_array_equal(quantize_inputs(source, "fp16").view(np.uint32), expected.view(np.uint32))
        with self.assertRaises(ValueError):
            GemmReference(np.array([[1e6]], dtype=np.float32), np.ones((1, 1), dtype=np.float32), "fp16")

    def test_quantization_is_reported_separately_from_wrong_arithmetic(self):
        a = np.array([[1.003, -2.007], [0.125, 1.111]], dtype=np.float32)
        b = np.array([[0.333, 2.111], [-1.007, 0.125]], dtype=np.float32)
        reference = GemmReference(a, b, "bf16")
        correct = reference.rounded.astype(np.float32)
        metrics = reference.verify(correct, "bf16")
        self.assertGreater(metrics["quantization_relative_error"], 0)
        self.assertGreater(metrics["relative_frobenius_error"], metrics["arithmetic_relative_error"])
        damaged = correct.copy()
        damaged[0, 0] += 0.1
        with self.assertRaises(RuntimeError):
            reference.verify(damaged, "bf16")

    def test_tf32_allows_ieee_only_when_the_backend_contract_permits_it(self):
        a = np.array([[1.0004, -1.0]], dtype=np.float32)
        b = np.array([[1.0004], [1.0]], dtype=np.float32)
        reference = GemmReference(a, b, "tf32")
        ieee_result = reference.original.astype(np.float32)
        with self.assertRaises(RuntimeError):
            reference.verify(ieee_result, "forced TF32")
        metrics = reference.verify(ieee_result, "TF32 allowed", allow_tf32_library=True)
        self.assertEqual(metrics["arithmetic_reference"], "original_fp32")
        self.assertEqual(metrics["quantization_relative_error"], 0)
        metrics = reference.verify(reference.rounded.astype(np.float32), "TF32 allowed", allow_tf32_library=True)
        self.assertEqual(metrics["arithmetic_reference"], "rounded_inputs")
        with self.assertRaises(RuntimeError):
            reference.verify(ieee_result + 0.1, "wrong TF32", allow_tf32_library=True)

    def test_tf32_ties_away_is_a_distinct_strict_reference(self):
        values = np.array([1 + 2.0**-11, -1 - 2.0**-11], dtype=np.float32)
        away_values = quantize_inputs(values, "tf32", tf32_ties="away")
        np.testing.assert_array_equal(away_values,
                                      np.array([1 + 2.0**-10, -1 - 2.0**-10], dtype=np.float32))
        a = np.array([[values[0], -1.0]], dtype=np.float32)
        b = np.array([[4.0], [4.0]], dtype=np.float32)
        reference = GemmReference(a, b, "tf32")
        actual = np.array([[2.0**-8]], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            reference.verify(actual, "forced RNE")
        metrics = reference.verify(actual, "RNA", allow_tf32_library=True)
        self.assertEqual(metrics["arithmetic_reference"], "tf32_ties_away")
        with self.assertRaises(RuntimeError):
            reference.verify(actual + 0.1, "wrong RNA", allow_tf32_library=True)

    def test_cancellation_respects_fp32_roundoff_but_rejects_wrong_results(self):
        a = np.array([[1e8, 1.0, -1e8]], dtype=np.float32)
        b = np.ones((3, 1), dtype=np.float32)
        reference = GemmReference(a, b, "fp32")
        # A valid sequential FP32 accumulation loses the middle term.
        actual = np.cumsum(a * b.T, axis=1, dtype=np.float32)[:, -1:]
        metrics = reference.verify(actual, "cancellation")
        self.assertEqual(metrics["max_abs_error"], 1)
        self.assertEqual(metrics["relative_frobenius_error"], 1)
        with self.assertRaises(RuntimeError):
            reference.verify(np.array([[1000.0]], dtype=np.float32), "wrong cancellation")

    def test_zero_reference_and_invalid_outputs(self):
        zero = np.zeros((2, 2), dtype=np.float32)
        reference = GemmReference(zero, np.ones((2, 2), dtype=np.float32), "fp32")
        metrics = reference.verify(zero, "zero")
        self.assertEqual(metrics["relative_frobenius_error"], 0)
        metrics = reference.verify(np.full((2, 2), 1e-7, dtype=np.float32), "near zero")
        self.assertIsNone(metrics["relative_frobenius_error"])
        for invalid in (np.zeros((2, 3), dtype=np.float32), zero.astype(np.float64),
                        np.full((2, 2), np.nan, dtype=np.float32)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(RuntimeError):
                reference.verify(invalid, "invalid")


if __name__ == "__main__":
    unittest.main()
