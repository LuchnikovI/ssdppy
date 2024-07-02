from typing import Union, Any
import numpy as np
from numpy.typing import NDArray, DTypeLike
from scipy.linalg.blas import get_blas_funcs  # type: ignore


def _fortran_random_normal_matrix(
    nrows: int,
    ncols: int,
    dtype: DTypeLike = np.float64,
) -> NDArray:
    buff = np.random.normal(size=(nrows * ncols,)).astype(dtype)
    sizeof = buff.itemsize
    return np.lib.stride_tricks.as_strided(
        buff, (nrows, ncols), (sizeof, sizeof * nrows)
    )


def _fortran_zero_matrix(
    nrows: int,
    ncols: int,
    dtype: DTypeLike = np.float64,
) -> NDArray:
    buff = np.zeros((nrows * ncols,), dtype=dtype)
    sizeof = buff.itemsize
    return np.lib.stride_tricks.as_strided(
        buff, (nrows, ncols), (sizeof, sizeof * nrows)
    )


def _rank1_update_noalloc(
    dst: NDArray,
    lhs: NDArray,
    rhs: NDArray,
    eta: Union[NDArray, float],
) -> NDArray:
    assert dst.flags.f_contiguous
    assert lhs.flags.f_contiguous
    assert rhs.flags.f_contiguous
    dst *= 1 - eta
    xger = get_blas_funcs("ger", (dst, lhs, rhs))
    return xger(eta, lhs, rhs, a=dst, overwrite_a=True)


def _matvec_noalloc(
    matrix: NDArray,
    vec: NDArray,
    dst: NDArray,
    alpha: Union[float, NDArray],
    beta: Union[float, NDArray],
    trans: bool,
) -> NDArray:
    assert matrix.flags.f_contiguous
    assert dst.flags.f_contiguous
    assert vec.flags.f_contiguous
    xgemv = get_blas_funcs("gemv", (matrix, vec, dst))
    return xgemv(alpha, matrix, vec, beta, dst, trans=trans, overwrite_y=True)


def _sum_noalloc(
    dst: NDArray,
    src: NDArray,
    alpha: Union[NDArray, float],
) -> NDArray:
    assert dst.flags.f_contiguous
    assert src.flags.f_contiguous
    strides = dst.strides
    shape = dst.shape
    sizeof = dst.itemsize
    dst = np.lib.stride_tricks.as_strided(dst, (dst.size,), (sizeof,))
    src = np.lib.stride_tricks.as_strided(src, (src.size,), (sizeof,))
    xaxpy = get_blas_funcs("axpy", (dst, src))
    result = xaxpy(src, dst, src.size, alpha)
    return np.lib.stride_tricks.as_strided(result, shape, strides)


def _get_ptr(arr: NDArray) -> int:
    return arr.__array_interface__["data"][0]
