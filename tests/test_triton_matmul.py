import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precision import INPUT_DTYPES, PRECISIONS, quantize_inputs, reference_product


TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
GPU_AVAILABLE = (
    TRITON_AVAILABLE and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 0)
)

if GPU_AVAILABLE:
    from triton_matmul import prepare


@unittest.skipUnless(GPU_AVAILABLE, "requires Triton and a CUDA GPU with compute capability 8.0+")
class TritonMatmulTests(unittest.TestCase):
    def test_precisions_and_masked_shapes_against_float64_reference(self):
        rng = np.random.default_rng(3927)
        for m, k, n in ((1, 1, 1), (7, 13, 11), (35, 37, 67), (17, 65, 19)):
            left = rng.standard_normal((m, k)).astype(np.float32)
            right = rng.standard_normal((k, n)).astype(np.float32)
            for precision in PRECISIONS:
                with self.subTest(shape=(m, k, n), precision=precision):
                    dtype = getattr(torch, INPUT_DTYPES[precision])
                    a = torch.tensor(left, device="cuda", dtype=dtype)
                    b = torch.tensor(right, device="cuda", dtype=dtype)
                    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
                    plan = prepare(a, b, out, precision)
                    self.assertIs(plan.out, out)
                    np.testing.assert_allclose(out.cpu().numpy(), reference_product(left, right, precision),
                                               rtol=5e-4, atol=5e-4)
                    out.fill_(float("nan"))
                    self.assertIs(plan.run(), out)
                    np.testing.assert_allclose(out.cpu().numpy(), reference_product(left, right, precision),
                                               rtol=5e-4, atol=5e-4)

    def test_tf32_signed_ties_and_rounding(self):
        bits = np.array([0x3f801000, 0x3f803000, 0x3f800fff, 0x3f801001,
                         0xbf801000, 0xbf803000, 0xbf800fff, 0xbf801001], dtype=np.uint32)
        values = bits.view(np.float32).reshape(1, -1)
        a = torch.tensor(values, device="cuda")
        b = torch.eye(values.size, device="cuda")
        out = torch.empty((1, values.size), device="cuda")
        prepare(a, b, out, "tf32")
        np.testing.assert_array_equal(out.cpu().numpy(), quantize_inputs(values, "tf32"))

    def test_rebinding_and_independent_prepared_plans(self):
        a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")
        b = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device="cuda")
        out = torch.empty((2, 2), device="cuda")
        plan = prepare(a, b, out, "fp32")
        original_config = plan.config
        # Mutating the report dictionary must not affect the frozen launch.
        plan.config["block_m"] = 1
        a2 = torch.tensor([[3., 1., -1.], [0., 2., 5.]], device="cuda")
        b2 = torch.tensor([[2., -1.], [1., 3.], [-2., 4.]], device="cuda")
        out2 = torch.empty_like(out)
        rebound = plan.bind(a2, b2, out2)
        out.fill_(-99)
        out2.fill_(-99)
        self.assertIs(rebound(), out2)
        np.testing.assert_array_equal(out2.cpu().numpy(), [[9., -4.], [-8., 26.]])
        np.testing.assert_array_equal(out.cpu().numpy(), np.full((2, 2), -99, dtype=np.float32))

        other_a = torch.ones((3, 5), device="cuda", dtype=torch.float16)
        other_b = torch.ones((5, 4), device="cuda", dtype=torch.float16)
        other_out = torch.empty((3, 4), device="cuda")
        prepare(other_a, other_b, other_out, "fp16")
        self.assertEqual(plan.config, original_config)
        self.assertIs(plan.run(), out)
        np.testing.assert_array_equal(out.cpu().numpy(), [[22., 28.], [49., 64.]])
        out2.fill_(0)
        rebound()
        np.testing.assert_array_equal(out2.cpu().numpy(), [[9., -4.], [-8., 26.]])

    def test_bind_rejects_shape_dtype_and_overlap(self):
        a = torch.ones((2, 3), device="cuda")
        b = torch.ones((3, 2), device="cuda")
        out = torch.empty((2, 2), device="cuda")
        plan = prepare(a, b, out, "fp32")
        with self.assertRaises(ValueError):
            plan.bind(torch.ones((1, 3), device="cuda"), b, torch.empty((1, 2), device="cuda"))
        with self.assertRaises(TypeError):
            plan.bind(a.half(), b, out)
        with self.assertRaises(ValueError):
            plan.bind(a, b, a.reshape(-1)[:4].view(2, 2))

    def test_bound_stream_and_cuda_graph(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
            b = torch.ones((4, 5), device="cuda")
            out = torch.empty((3, 5), device="cuda")
            plan = prepare(a, b, out, "fp32")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan.run()
            out.fill_(-1)
            graph.replay()
        stream.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), np.repeat([[6.], [22.], [38.]], 5, axis=1))


if __name__ == "__main__":
    unittest.main()
