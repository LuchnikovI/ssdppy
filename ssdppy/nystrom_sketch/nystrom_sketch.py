from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray, DTypeLike
from numpy.linalg import norm, cholesky
from scipy.linalg import solve_triangular, svd  # type: ignore
from ..low_level import (
    _fortran_random_normal_matrix,
    _fortran_zero_matrix,
    _rank1_update_noalloc,
    _matvec_noalloc,
    _sum_noalloc,
)


class NystromSketch:

    def __init__(
        self,
        variables_number: int,
        sketch_size: int,
        dtype: DTypeLike = np.float64,
    ):
        self._dtype = dtype
        self._variables_number = variables_number
        self._sketch_size = sketch_size
        self._sketch = _fortran_zero_matrix(variables_number, sketch_size, dtype)
        self._omega = _fortran_random_normal_matrix(
            variables_number, sketch_size, dtype
        )
        self.__vec = np.empty(shape=(sketch_size,), dtype=dtype)

    def update(self, v: NDArray, eta: Union[NDArray, float]) -> None:
        self.__vec = _matvec_noalloc(self._omega, v, self.__vec, 1.0, 0.0, True)
        self._sketch = _rank1_update_noalloc(self._sketch, v, self.__vec, eta)

    def reconstruct(self) -> Tuple[NDArray, NDArray]:
        # TODO: must be a better way to regularize cholesky, try to improve
        sigma = (
            np.sqrt(self._variables_number)
            * 1e-14
            * np.max(norm(self._sketch, axis=0))  # allocates only a small buffer
        )
        self._sketch = _sum_noalloc(self._sketch, self._omega, sigma)
        # matrices below are small, can go high level
        b = self._omega.T @ self._sketch
        b = 0.5 * (b + b.T)
        lt = cholesky(b)
        # -------------------------------------------
        # TODO: solve_triangular and svd allocate extra 'big' memory, try to resolve
        self._sketch = solve_triangular(
            lt, self._sketch.T, lower=True, overwrite_b=True
        ).T
        u, s, _ = svd(self._sketch, full_matrices=False, overwrite_a=True)
        s = np.maximum(s**2 - sigma * np.ones(s.shape[0]), 0.0)
        return u, s
