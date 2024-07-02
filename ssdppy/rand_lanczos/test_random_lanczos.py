import numpy as np
from numpy.typing import NDArray
from .rand_lanczos import lanczos


def test_random_lanczos():
    matrix = np.random.normal(size=(100, 100))
    matrix = 0.5 * (matrix + matrix.T)

    def op(src: NDArray, dst: NDArray):
        dst[:] = matrix.dot(src)

    lmbd, eigvec = lanczos(
        op,
        100,
        50,
    )
    assert np.isclose(matrix.dot(eigvec) / eigvec, lmbd).all()
    assert np.isclose(np.linalg.eigvalsh(matrix)[0], lmbd)
