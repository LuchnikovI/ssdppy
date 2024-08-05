from typing import Union, Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray, DTypeLike
from sparse_sdp import SDPF64, SDPF32
from ssdppy.nystrom_sketch.nystrom_sketch import NystromSketch
from ssdppy.rand_lanczos.rand_lanczos import lanczos

Float = Union[float, NDArray]


def _find_gamma(z: NDArray, b: NDArray, t: int, beta0: Float, eta: Float) -> NDArray:
    return np.minimum(
        4 * beta0 * np.sqrt(t + 2) * (eta**2) / (np.linalg.norm(z - b) ** 2), beta0
    )


"""The information about optimization process. It includes:
    infeasibility: |Tr(AX) - b| / |b|
    TODO: an estimation of the duality gap
"""


@dataclass
class Info:
    infeasibility: float


class SketchyCGALSolver:
    """A sketchy CGAL solver from https://arxiv.org/pdf/1912.02949.
    Args:
        sdp: either SDPF64 or SDPF32 task that can be build from SDPBuilderF32, SDPBuilderF64;
        sketch_size: a size of Nystrom sketch that is can be also viewed as the rank of the
            resulting matrix;
        T: number of iterations;
        beta0: a parameter from https://arxiv.org/pdf/1912.02949;
        acc: accuracy parameter of some internal subroutines."""

    def __init__(
        self,
        sdp: Union[SDPF32, SDPF64],
        sketch_size: int,
        T: int = 1000,
        beta0: Float = 1.0,
        acc: Float = 1e-8,
    ):
        variables_number = sdp.variables_number
        if variables_number <= 0:
            raise ValueError("Number of variables must be > 0")
        if sketch_size <= 0:
            raise ValueError("Sketch size must be > 0")
        if variables_number < sketch_size:
            raise ValueError("Sketch size must be <= variables number")
        self._sdp = sdp
        self._variables_number = variables_number
        self._sketch_size = sketch_size
        self._T = T
        self._beta0 = beta0
        self._acc = acc
        self._dtype: DTypeLike = sdp._get_b().dtype

    """Solves SDP problem.
    Returns:
        truncated eigen decomposition of the resulting matrix, i.e.
        a matrix U with eigenvectors and a vector S with eigenvalues, and
        information about result."""

    def solve(self) -> Tuple[NDArray, NDArray, Info]:
        sketch = NystromSketch(self._variables_number, self._sketch_size, self._dtype)
        z = np.zeros(self._sdp._get_b().shape, self._dtype)
        y = np.zeros(self._sdp._get_b().shape, self._dtype)
        for t in range(1, self._T + 1):
            beta = self._beta0 * np.sqrt(t + 1)
            eta = 2.0 / (t + 1)
            q = int(np.ceil((t**0.25) * np.log(self._variables_number)))

            def lanczos_input(src: NDArray, dst: NDArray):
                self._sdp._apply_weighted_constraints(
                    src, y + beta * (z - self._sdp._get_b()), dst, 1.0, 0.0
                )
                self._sdp._apply_objective_matrix(src, dst, 1.0, 1.0)

            _, eigvec = lanczos(
                lanczos_input,
                self._variables_number,
                q,
                self._acc,
                self._dtype,
            )

            self._sdp._compute_brackets(eigvec, z, eta, 1.0 - eta)
            gamma = _find_gamma(z, self._sdp._get_b(), t, self._beta0, eta)
            y = y + gamma * (z - self._sdp._get_b())
            sketch.update(eigvec, eta)
        u, s = sketch.reconstruct()
        s = s + ((1 - s.sum()) / self._sketch_size)
        infeasibility = self._sdp._compute_infeasibility(u, s)
        info = Info(infeasibility)
        return u, s, info
