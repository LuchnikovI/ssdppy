#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
from ssdppy import SDPBuilderF64, SketchyCGALSolver


def main():

    variables_number = 10
    constraints_number = 7
    sketch_size = 3

    np.random.seed(42)

    # matrix defining the objective function (with normalization)
    c = np.random.randn(variables_number, variables_number)
    c = (c + c.T) / 2
    c /= np.linalg.norm(c)
    # constraints (with normalization)
    b = np.random.randn(constraints_number)
    a = np.random.randn(constraints_number, variables_number, variables_number)
    a = (a + a.transpose(0, 2, 1)) / 2
    if constraints_number != 0:
        for i in range(constraints_number):
            f_norm = np.linalg.norm(a[i])
            a[i] /= f_norm
            b[i] /= f_norm
        op_norm = np.linalg.norm(
            a.transpose((1, 2, 0)).reshape((variables_number**2, constraints_number)), 2
        )
        a /= op_norm
        b /= op_norm

    # configure sketchy sgal solver
    sparse_sdp_builder = SDPBuilderF64(constraints_number, variables_number)
    for row_idx, row in enumerate(c):
        for col_idx, value in enumerate(row):
            sparse_sdp_builder.add_element_to_objective_matrix(row_idx, col_idx, value)
    for constr_num, constr in enumerate(a):
        sparse_sdp_builder.add_element_to_b_vector(constr_num, b[constr_num])
        for row_idx, row in enumerate(constr):
            for col_idx, value in enumerate(row):
                sparse_sdp_builder.add_element_to_constraint_matrix(
                    constr_num, row_idx, col_idx, value
                )
    sparse_sdp = sparse_sdp_builder.build()
    scgal_solver = SketchyCGALSolver(sparse_sdp, sketch_size)

    # solving sdp using sketchy sgal
    u, s = scgal_solver.solve()
    cgal_x = u @ (s * u).T

    # solving sdp with cvx
    x = cp.Variable((variables_number, variables_number), symmetric=True)
    constraints = [x >> 0, cp.trace(x) == 1]
    constraints += [cp.trace(a[i] @ x) == b[i] for i in range(constraints_number)]
    prob = cp.Problem(cp.Minimize(cp.trace(c @ x)), constraints)
    prob.solve()

    print(
        f"Relative error between solutions {np.linalg.norm(cgal_x - x.value) / np.linalg.norm(cgal_x)}"
    )


if __name__ == "__main__":
    main()
