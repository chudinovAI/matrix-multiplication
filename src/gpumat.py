import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary("./matmul.so")
lib.matmul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


class GPUMatrix:
    def __init__(self, data: np.ndarray):
        assert data.dtype == np.float32
        assert data.flags["C_CONTIGUOUS"]
        self.data = data

    def __matmul__(self, other: "GPUMatrix") -> "GPUMatrix":
        A = self.data
        B = other.data
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Несовпадение размерностей!"

        C = np.zeros((M, N), dtype=np.float32)
        lib.matmul(A, B, C, M, K, N)
        return GPUMatrix(C)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self):
        return f"GPUMatrix(shape={self.data.shape})"
