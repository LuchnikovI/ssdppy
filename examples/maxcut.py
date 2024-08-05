#!/usr/bin/env python3
import sys
import time
import urllib.request as ur
from typing import Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from ssdppy import SDPBuilderF64, SketchyCGALSolver

# list with best known cut values for gset https://medium.com/toshiba-sbm/benchmarking-the-max-cut-problem-on-the-simulated-bifurcation-machine-e26e1127c0b0
best_cut_values = [
    11624,
    11620,
    11622,
    11646,
    11631,
    2178,
    2006,
    2005,
    2054,
    2000,
    564,
    556,
    582,
    3064,
    3050,
    3052,
    3047,
    992,
    906,
    941,
    931,
    13359,
    13344,
    13337,
    13340,
    13328,
    3341,
    3298,
    3405,
    3413,
    3310,
    1410,
    1382,
    1384,
    7687,
    7680,
    7691,
    7688,
    2408,
    2400,
    2405,
    2481,
    6660,
    6650,
    6654,
    6649,
    6657,
    6000,
    6000,
    5880,
    3848,
    3851,
    3850,
    3852,
    10299,
    4017,
    3494,
    19293,
    6086,
    14188,
    5796,
    4870,
    27045,
    8751,
    5562,
    6364,
    6950,
    None,
    None,
    9591,
    None,
    7006,
]


def main():

    # parameters
    gset_graph_id = int(sys.argv[1])
    sketch_size = 150
    randomized_rounding_attempts_number = 1000
    cgal_iterations_number = 3000
    seed = 42

    print(f"G{gset_graph_id} is running...")

    np.random.seed(seed)

    # download and parse a graph
    with ur.urlopen(f"https://web.stanford.edu/~yyye/yyye/Gset/G{gset_graph_id}") as f:
        graph_str = f.read().decode("utf-8")
    lines = graph_str.split("\n")
    variables_number = int(lines[0].split()[0])
    constraints_number = variables_number

    # create an SDP builder
    sdp_builder = SDPBuilderF64(constraints_number, variables_number)
    # add maxcut constraints
    sdp_builder.add_maxcut_constraints()
    degrees = np.zeros((variables_number,))
    graph = {}

    def parse_line(x: str) -> Tuple[int, int, float]:
        i, j, v = x.split()
        return int(i) - 1, int(j) - 1, float(v)

    # populate sdp builder
    for i, j, val in map(parse_line, filter(lambda x: len(x) != 0, lines[1:])):
        sdp_builder.add_element_to_objective_matrix(i, j, 0.25 * val)
        sdp_builder.add_element_to_objective_matrix(j, i, 0.25 * val)
        degrees[i] += 0.25 * val
        degrees[j] += 0.25 * val
    for i, d in enumerate(degrees):
        sdp_builder.add_element_to_objective_matrix(i, i, -float(d))
    # normalize -laplacian / 4
    sdp_builder.normalize_objective_matrix()
    # build an sdp instance
    sdp = sdp_builder.build()

    # following two closures are useful to evaluate the solution
    """This closure takes argument in a decomposed form and compute the objective function value."""
    def objective_function_value(u: NDArray, s: NDArray) -> NDArray:
        return -sdp.compute_objective_value(u, s) * sdp.variables_number

    """This closure computes the cut value."""
    def cut_value(cut: NDArray) -> NDArray:
        return -sdp.compute_objective_value(cut.copy().reshape((-1, 1)), np.array([1.]))

    # solve an sdp
    solver = SketchyCGALSolver(sdp, T=cgal_iterations_number, sketch_size=sketch_size)
    start_time = time.time()
    u, s, info = (
        solver.solve()
    )  # this returns the truncated eigen decomposition of the solution matrix
    end_time = time.time()
    print(f"Solver runtime: {end_time - start_time} seconds")

    print(f"SDP relaxation objective function value: {objective_function_value(u, s)}")

    # randomized rounding
    tau = np.random.randn(randomized_rounding_attempts_number, sketch_size)
    tau /= np.linalg.norm(tau, axis=1, keepdims=True)
    tau *= np.sqrt(s)
    cuts = np.sign(np.tensordot(u, tau, axes=[[1], [1]]))
    maxcut = 0.0
    for cut in cuts.T:
        maxcut = max(maxcut, cut_value(cut))
    print(f"Best known cut: {best_cut_values[gset_graph_id - 1]}")
    print(
        f"Goemans-Williamson cut estimation: {0.878 * best_cut_values[gset_graph_id - 1]}"
    )
    print(f"Infeasibility of the result (|<AX> - b|_F / |b|_F): {info.infeasibility}")
    print(
        f"Randomized rounding with {randomized_rounding_attempts_number} attempts gives cut value: {maxcut}"
    )

    # rounding from https://arxiv.org/pdf/1912.02949
    maxcut = 0.0
    cuts = np.sign(u)
    for cut in cuts.T:
        maxcut = max(maxcut, cut_value(cut))
    print(f"Rounding from SketchyCGAL paper gives cut value: {maxcut}")


if __name__ == "__main__":
    main()
