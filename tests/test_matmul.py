import gc
from pathlib import Path
import sys
import unittest
import weakref

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpumat import NativeMatmul
from precision import GemmReference, INPUT_DTYPES, PRECISIONS, reference_product, unsupported_reason


LIBRARY = Path(__file__).resolve().parents[1] / "build" / "matmul.so"
GPU_AVAILABLE = torch.cuda.is_available() and LIBRARY.is_file()
ALGORITHMS = ("tiled", "register", "cublas", "warp", "warp_large", "pipeline", "wmma")




@unittest.skipUnless(GPU_AVAILABLE, "requires CUDA and build/matmul.so (run make cuda)")
class NativeMatmulTests(unittest.TestCase):
    def setUp(self):
        self.native = NativeMatmul(LIBRARY)
        self.addCleanup(self.native.close)

    def test_all_implementations_against_float64_reference(self):
        rng = np.random.default_rng(923)
        capability = torch.cuda.get_device_capability(0)
        for m, k, n in ((1, 1, 1), (7, 13, 11), (35, 37, 67), (129, 35, 133), (2, 2049, 3)):
            a_numpy = rng.standard_normal((m, k)).astype(np.float32)
            b_numpy = rng.standard_normal((k, n)).astype(np.float32)
            for implementation in ALGORITHMS:
                for precision in PRECISIONS:
                    if unsupported_reason(implementation, precision, capability):
                        continue
                    with self.subTest(shape=(m, k, n), implementation=implementation, precision=precision):
                        reference = GemmReference(a_numpy, b_numpy, precision)
                        dtype = getattr(torch, INPUT_DTYPES[precision])
                        a = torch.empty(m * k + 1, dtype=dtype, device="cuda")[1:].view(m, k)
                        b = torch.empty(k * n + 1, dtype=dtype, device="cuda")[1:].view(k, n)
                        out = torch.empty(m * n + 1, dtype=torch.float32, device="cuda")[1:].view(m, n)
                        a.copy_(torch.from_numpy(a_numpy))
                        b.copy_(torch.from_numpy(b_numpy))
                        run = self.native.prepare(implementation, a, b, out, precision=precision)
                        for _ in range(2):
                            run()
                        torch.cuda.current_stream().synchronize()
                        reference.verify(out.cpu().numpy(), implementation,
                                         allow_tf32_library=implementation == "cublas" and precision == "tf32")

    def test_cublas_alternating_precisions_on_one_context(self):
        capability = torch.cuda.get_device_capability(0)
        a_numpy = np.array([[1.0004, -0.5], [0.25, -1.0004]], dtype=np.float32)
        b_numpy = np.array([[1.0004, -0.25], [0.5, 1.0004]], dtype=np.float32)
        for precision in ("fp32", "tf32", "fp32", "fp16", "fp32", "bf16", "fp32", "tf32", "fp32"):
            if unsupported_reason("cublas", precision, capability):
                continue
            with self.subTest(precision=precision):
                dtype = getattr(torch, INPUT_DTYPES[precision])
                a = torch.from_numpy(a_numpy).to(device="cuda", dtype=dtype)
                b = torch.from_numpy(b_numpy).to(device="cuda", dtype=dtype)
                out = torch.empty((2, 2), dtype=torch.float32, device="cuda")
                self.native.prepare("cublas", a, b, out, precision=precision)()
                expected = reference_product(a_numpy, b_numpy, precision)
                if precision == "tf32":
                    GemmReference(a_numpy, b_numpy, precision).verify(out.cpu().numpy(), "cublas/tf32",
                                                                    allow_tf32_library=True)
                else:
                    tolerance = 1e-6 if precision == "fp32" else 5e-4
                    np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=tolerance, atol=tolerance)

    def test_prepared_nondefault_stream_and_graph(self):
        capability = torch.cuda.get_device_capability(0)
        rng = np.random.default_rng(14)
        a_numpy = rng.standard_normal((7, 13)).astype(np.float32)
        b_numpy = rng.standard_normal((13, 11)).astype(np.float32)
        stream = torch.cuda.Stream()
        for implementation in ALGORITHMS:
            for precision in PRECISIONS:
                if unsupported_reason(implementation, precision, capability):
                    continue
                with self.subTest(implementation=implementation, precision=precision), torch.cuda.stream(stream):
                    dtype = getattr(torch, INPUT_DTYPES[precision])
                    a = torch.from_numpy(a_numpy).to(device="cuda", dtype=dtype)
                    b = torch.from_numpy(b_numpy).to(device="cuda", dtype=dtype)
                    out = torch.empty((7, 11), dtype=torch.float32, device="cuda")
                    run = self.native.prepare(implementation, a, b, out, precision=precision)
                    run()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        run()
                    graph.replay()
                    stream.synchronize()
                    GemmReference(a_numpy, b_numpy, precision).verify(
                        out.cpu().numpy(), implementation + "/graph",
                        allow_tf32_library=implementation == "cublas" and precision == "tf32")

    def test_wmma_tf32_rounds_halfway_to_even(self):
        if unsupported_reason("wmma", "tf32", torch.cuda.get_device_capability(0)):
            self.skipTest("WMMA requires compute capability 8.0")
        bits = np.array([0x3f801000, 0x3f803000, 0xbf801000, 0xbf803000], dtype=np.uint32)
        values = bits.view(np.float32).reshape(1, 4)
        a = torch.from_numpy(values).cuda()
        b = torch.eye(4, dtype=torch.float32, device="cuda")
        out = torch.empty((1, 4), dtype=torch.float32, device="cuda")
        self.native.prepare("wmma", a, b, out, precision="tf32")()
        expected = reference_product(values, np.eye(4, dtype=np.float32), "tf32")
        np.testing.assert_array_equal(out.cpu().numpy(), expected)

    def test_invalid_tensor_contracts(self):
        a = torch.ones((2, 3), dtype=torch.float32, device="cuda")
        b = torch.ones((3, 4), dtype=torch.float32, device="cuda")
        out = torch.empty((2, 4), dtype=torch.float32, device="cuda")
        cases = [
            ("unknown", a, b, out, ValueError),
            ("tiled", None, b, out, TypeError),
            ("tiled", a.flatten(), b, out, ValueError),
            ("tiled", a.double(), b, out, TypeError),
            ("tiled", a.cpu(), b, out, ValueError),
            ("tiled", torch.ones((3, 2), device="cuda").T, b, out, ValueError),
            ("tiled", a, b[:2], out, ValueError),
            ("tiled", a, b, out[:, :3].contiguous(), ValueError),
            ("tiled", a[:0], b, out[:0], ValueError),
        ]
        for index, (algorithm, left, right, result, error) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(error):
                self.native.prepare(algorithm, left, right, result)
        if torch.cuda.device_count() > 1:
            with self.assertRaises(ValueError):
                self.native.prepare("tiled", a.to("cuda:1"), b, out)

    def test_precision_contracts(self):
        a = torch.ones((2, 3), dtype=torch.float32, device="cuda")
        b = torch.ones((3, 4), dtype=torch.float32, device="cuda")
        out = torch.empty((2, 4), dtype=torch.float32, device="cuda")
        with self.assertRaises(ValueError):
            self.native.prepare("cublas", a, b, out, precision="invalid")
        for implementation in ("tiled", "register", "warp", "warp_large", "pipeline"):
            for precision in ("tf32", "fp16", "bf16"):
                dtype = getattr(torch, INPUT_DTYPES[precision])
                with self.subTest(implementation=implementation, precision=precision), self.assertRaises(ValueError):
                    self.native.prepare(implementation, a.to(dtype), b.to(dtype), out, precision=precision)
        with self.assertRaises(ValueError):
            self.native.prepare("wmma", a, b, out, precision="fp32")
        for precision in ("fp16", "bf16"):
            with self.subTest(precision=precision), self.assertRaises(TypeError):
                self.native.prepare("cublas", a, b, out, precision=precision)
        with self.assertRaises(TypeError):
            self.native.prepare("cublas", a, b, out.half(), precision="fp32")

    def test_partial_overlap_rejected_but_disjoint_storage_works(self):
        storage = torch.arange(12, dtype=torch.float32, device="cuda")
        first = storage[:4].view(2, 2)
        overlapping = storage[1:5].view(2, 2)
        disjoint = storage[4:8].view(2, 2)
        independent = torch.ones((2, 2), dtype=torch.float32, device="cuda")
        with self.assertRaises(ValueError):
            self.native.prepare("tiled", first, independent, overlapping)
        with self.assertRaises(ValueError):
            self.native.prepare("tiled", independent, first, overlapping)
        self.native.prepare("tiled", first, independent, disjoint)()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(disjoint.cpu().numpy(), [[1, 1], [5, 5]])

    def test_prepared_operation_retains_tensors_and_rejects_closed_context(self):
        a = torch.ones((2, 2), dtype=torch.float32, device="cuda")
        b = torch.ones((2, 2), dtype=torch.float32, device="cuda")
        out = torch.empty((2, 2), dtype=torch.float32, device="cuda")
        references = [weakref.ref(tensor) for tensor in (a, b, out)]
        run = self.native.prepare("tiled", a, b, out)
        del a, b, out
        gc.collect()
        run()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(references[2]().cpu().numpy(), [[2, 2], [2, 2]])
        self.native.close()
        self.native.close()
        with self.assertRaises(RuntimeError):
            run()
        with self.assertRaises(RuntimeError):
            self.native.prepare("tiled", None, None, None)
        with self.assertRaises(RuntimeError):
            self.native.configuration("tiled")
        del run
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_invalid_device_does_not_poison_a_later_valid_launch(self):
        with self.assertRaises(RuntimeError):
            NativeMatmul(LIBRARY, device=torch.cuda.device_count())
        a = torch.tensor([[2.0, -3.0]], device="cuda")
        b = torch.tensor([[4.0], [5.0]], device="cuda")
        out = torch.empty((1, 1), device="cuda")
        self.native.prepare("register", a, b, out)()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), [[-7.0]])


if __name__ == "__main__":
    unittest.main()
