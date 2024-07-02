import numpy as np
from ..low_level import (
    _matvec_noalloc,
    _fortran_random_normal_matrix,
    _rank1_update_noalloc,
    _sum_noalloc,
)


"""Tests vector matrix product subroutine that does not allocate memory"""


def test_vecmat_noalloc():
    alpha = 0.65
    beta = 1.23
    vec = np.random.normal(size=(10,))
    matrix = _fortran_random_normal_matrix(10, 25)
    dst = np.random.normal(size=(25,))
    true_result = dst * beta + np.tensordot(vec, matrix, axes=1) * alpha
    dst = _matvec_noalloc(matrix, vec, dst, alpha, beta, True)
    assert np.isclose(true_result, dst).all()


"""Tests rank 1 update subroutine  that does not allocate memory"""


def test_rank1_update_noalloc():
    eta = 0.81
    lhs = np.random.normal(size=(15,))
    rhs = np.random.normal(size=(35,))
    matrix = _fortran_random_normal_matrix(15, 35)
    true_result = eta * np.outer(lhs, rhs) + (1 - eta) * matrix
    matrix = _rank1_update_noalloc(matrix, lhs, rhs, eta)
    assert np.isclose(matrix, true_result).all()


"""Tests summation of matrices that does not allocate memory"""


def test_sum_noalloc():
    alpha = 0.42
    dst = _fortran_random_normal_matrix(17, 33)
    src = _fortran_random_normal_matrix(17, 33)
    true_result = dst + src * alpha
    dst = _sum_noalloc(dst, src, alpha)
    assert np.isclose(true_result, dst).all()
