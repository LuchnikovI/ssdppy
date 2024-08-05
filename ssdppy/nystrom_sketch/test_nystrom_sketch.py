import numpy as np
from .nystrom_sketch import NystromSketch


def test_nystrom_sketch():
    eta = 0.1
    sketch_size = 10
    variables_number = 100
    sketched_matrix = np.zeros((variables_number, variables_number))
    sketch = NystromSketch(variables_number, sketch_size)
    for _ in range(sketch_size):
        v = np.random.normal(size=(variables_number,))
        sketch.update(v, eta)
        sketched_matrix = (1 - eta) * sketched_matrix + eta * np.outer(v, v)
    (u, s) = sketch.reconstruct()
    reconstructed_sketch = u @ (s[:, np.newaxis] * u.T.conj())
    assert np.isclose(
        reconstructed_sketch, sketched_matrix
    ).all(), f"{reconstructed_sketch}\n{sketched_matrix}"
