#!/usr/bin/env python3

from typing import Union, Type
import numpy as np
from numpy.typing import DTypeLike
from sparse_sdp import SDPBuilderF64, SDPBuilderF32


def _test_subroutines(
    builder_type: Union[Type[SDPBuilderF32], Type[SDPBuilderF64]],
    dtype: DTypeLike,
    var_num: int,
    constr_num: int,
    elem_num_approx: int,
    alpha: float,
    beta: float,
):
    if dtype is np.float64:
        atol = 1e-10
    elif dtype is np.float32:
        atol = 1e-5
    random_pos = np.random.randint(0, var_num, (elem_num_approx, 2))
    unique_pos = set()
    for pos in random_pos:
        unique_pos.add((pos[0], pos[1]))
    builder = builder_type(constr_num, var_num)
    c = np.zeros((var_num, var_num), dtype)
    for row_idx, col_idx in unique_pos:
        elem = float(np.random.normal(size=()))
        builder.add_element_to_objective_matrix(row_idx, col_idx, elem)
        c[row_idx, col_idx] = elem
    a = np.zeros((constr_num, var_num, var_num), dtype)
    for constr_idx in range(constr_num):
        random_pos = np.random.randint(0, var_num, (elem_num_approx, 2))
        unique_pos = set()
        for pos in random_pos:
            unique_pos.add((pos[0], pos[1]))
        for row_idx, col_idx in unique_pos:
            elem = float(np.random.normal(size=()))
            builder.add_element_to_constraint_matrix(constr_idx, row_idx, col_idx, elem)
            a[constr_idx, row_idx, col_idx] = elem
    sdp = builder.build()
    vec = np.random.normal(size=(var_num,)).astype(dtype)
    z = np.random.normal(size=(constr_num,)).astype(dtype)
    cvec = np.random.normal(size=(var_num,)).astype(dtype)
    true_cvec = alpha * c.dot(vec) + beta * cvec
    sdp._apply_objective_matrix(vec, cvec, alpha, beta)
    assert np.isclose(true_cvec, cvec, atol=atol).all(), f"{true_cvec}, {cvec}"
    weighted_constraints = np.random.normal(size=(var_num,)).astype(dtype)
    true_weighted_constraints = (
        alpha * np.einsum("i,ijk,k->j", z, a, vec) + beta * weighted_constraints
    )
    sdp._apply_weighted_constraints(vec, z, weighted_constraints, alpha, beta)
    assert np.isclose(
        true_weighted_constraints, weighted_constraints, atol=atol
    ).all(), f"{true_weighted_constraints}, {weighted_constraints}"
    brackets = np.random.normal(size=(constr_num,)).astype(dtype)
    true_brackets = alpha * np.einsum("ikj,k,j->i", a, vec, vec) + beta * brackets
    sdp._compute_brackets(vec, brackets, alpha, beta)
    assert np.isclose(
        true_brackets, brackets, atol=atol
    ).all(), f"{true_brackets}, {brackets}"


_test_subroutines(
    SDPBuilderF64,
    np.float64,
    1,
    0,
    1,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF64,
    np.float64,
    15,
    0,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF64,
    np.float64,
    15,
    5,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF64,
    np.float64,
    15,
    10,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF64,
    np.float64,
    15,
    25,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF32,
    np.float32,
    1,
    0,
    1,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF32,
    np.float32,
    15,
    0,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF32,
    np.float32,
    15,
    5,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF32,
    np.float32,
    15,
    10,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)
_test_subroutines(
    SDPBuilderF32,
    np.float32,
    15,
    25,
    60,
    float(np.random.uniform(size=())),
    float(np.random.uniform(size=())),
)

print("Subroutines: OK")
