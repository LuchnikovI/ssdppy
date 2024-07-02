from typing import Callable, Tuple, Union
import numpy as np
from numpy.typing import NDArray, DTypeLike
from numpy.linalg import norm
from scipy.linalg import eigh_tridiagonal  # type: ignore
from ..low_level import _sum_noalloc


def lanczos(
    operator: Callable[[NDArray, NDArray], None],
    variables_number: int,
    q: int,
    acc: Union[float, NDArray] = 1e-8,
    dtype: DTypeLike = np.float64,
) -> Tuple[NDArray, NDArray]:
    iter_number = max(min(variables_number - 1, q), 20)
    omega_vec = np.zeros((iter_number,), dtype=dtype)
    rho_vec = np.zeros((iter_number,), dtype=dtype)
    rng_state = np.random.get_state()
    v0 = np.random.normal(size=(variables_number,)).astype(dtype)
    v0 /= norm(v0)
    buffers = [
        np.zeros((variables_number,), dtype=dtype),
        v0,
        np.zeros((variables_number,), dtype=dtype),
    ]
    for i in range(iter_number):
        v_i_minus_1 = buffers[i % 3]
        v_i = buffers[(i + 1) % 3]
        v_i_plus_1 = buffers[(i + 2) % 3]
        operator(v_i, v_i_plus_1)
        omega_vec[i] = v_i.dot(v_i_plus_1)
        _sum_noalloc(v_i_plus_1, v_i, -omega_vec[i])
        if i != 0:
            _sum_noalloc(v_i_plus_1, v_i_minus_1, -rho_vec[i - 1])
        rho_vec[i] = norm(v_i_plus_1)
        if rho_vec[i] < acc:
            break
        v_i_plus_1 /= rho_vec[i]
    lmbd, v = eigh_tridiagonal(
        omega_vec[: i + 1],
        rho_vec[:i],
        tol=acc,
        select="i",
        select_range=(0, 0),
    )
    # reuse a buffer (will be filled in the loop below)
    eigvec = buffers[1]
    eigvec[:] = 0.0
    # rerun the main loop aggregating vectors
    np.random.set_state(rng_state)
    buffers[1] = np.random.normal(size=(variables_number,)).astype(dtype)
    buffers[1] /= norm(buffers[1])
    for i in range(iter_number):
        v_i_minus_1 = buffers[i % 3]
        v_i = buffers[(i + 1) % 3]
        v_i_plus_1 = buffers[(i + 2) % 3]
        _sum_noalloc(eigvec, v_i, v[i, 0])
        operator(v_i, v_i_plus_1)
        omega_vec[i] = v_i.dot(v_i_plus_1)
        _sum_noalloc(v_i_plus_1, v_i, -omega_vec[i])
        if i != 0:
            _sum_noalloc(v_i_plus_1, v_i_minus_1, -rho_vec[i - 1])
        rho_vec[i] = norm(v_i_plus_1)
        if rho_vec[i] < acc:
            break
        v_i_plus_1 /= rho_vec[i]
    return lmbd, eigvec
