from numpy.typing import NDArray

class SDPF32:
    @property
    def constraints_number(self) -> int: ...
    @property
    def variables_number(self) -> int: ...
    def _get_b(self) -> NDArray: ...
    def _apply_objective_matrix(
        self, src: NDArray, dst: NDArray, alpha: float, beta: float
    ) -> None: ...
    def _apply_weighted_constraints(
        self, src: NDArray, weights: NDArray, dst: NDArray, alpha: float, beta: float
    ) -> None: ...
    def _compute_brackets(
        self, src: NDArray, dst: NDArray, alpha: float, beta: float
    ) -> None: ...

class SDPBuilderF32:
    """A single-precision builder class for the following semidefinite
    programming (SDP) problem:
        min. <C, X>
            s.t. <A[i], X> = b[i];
                         tr(X) = 1;
                         X >= 0,
    constraints and objective function matrix (C) are zero after initialization.
    A[i] for all i and C must be symmetric and normalized in the following way:
    |C|_F = 1, |A|_2 = 1, where A[x] = [<A[0], x>, ..., <A[`constraints_number` - 1], x>],
    i.e. A is a linear map that maps a matrix x to a vector of size `constraints_number`.
    |A[0]|_F = |A[1]|_F = .. = |A[`constraints_number` - 1]|_F, for more details see
    https://arxiv.org/pdf/1912.02949.
    Args:
        constraints_number: number of constraints;
        variables_number: number of variables."""

    def __init__(self, constraints_number: int, variables_number: int) -> None: ...
    """A value insertion to a particular position in the objective function matrix.
    Args:
        row_idx: a row index of the insertion;
        col_idx: a column index of the insertion;
        value: a value to be inserted."""
    def add_element_to_objective_matrix(
        self, row_idx: int, col_idx: int, value: float
    ) -> None: ...
    """A value insertion to the vector b.
    Args:
        constraint_number: a number of the constraint (starts from 0);
        value: a value to be inserted."""
    def add_element_to_b_vector(self, constraint_number: int, value: float) -> None: ...
    """A value insertion to a particular position in the particular constraint matrix.
    Args:
        constraint_number: a number of the constraint (starts from 0);
        row_idx: a row index of the insertion;
        col_idx: a column index of the insertion;
        value: a value to be inserted."""
    def add_element_to_constraint_matrix(
        self, constraint_number: int, row_idx: int, col_idx: int, element: float
    ) -> None: ...
    """MaxCut constraints initialization, i.e. sets constraint matrices and b vector in such a way that:
    X[i, i] = 1 / `constraints_number`."""
    def add_maxcut_constraints(self) -> None: ...
    """Creation of the SPD task, after calling this method the builder class
    is reinitialized (all matrices are zero).
    Returns:
        sdp task that can be sent to a solver."""
    def build(self) -> SDPF64: ...
    """Normalizes the objective function matrix in such a way that |C|_F = 1"""
    def normalize_objective_matrix(self) -> None: ...
    def constraints_number(self) -> int: ...
    def variables_number(self) -> int: ...

class SDPF64:
    @property
    def constraints_number(self) -> int: ...
    @property
    def variables_number(self) -> int: ...
    def _get_b(self) -> NDArray: ...
    def _apply_objective_matrix(
        self, src: NDArray, dst: NDArray, alpha: float, beta: float
    ) -> None: ...
    def _apply_weighted_constraints(
        self, src: NDArray, weights: NDArray, dst: NDArray, alpha: float, beta: float
    ) -> None: ...
    def _compute_brackets(
        self, src: NDArray, dst: NDArray, alpha: float, beta: float
    ) -> None: ...

class SDPBuilderF64:
    """A double-precision builder class for the following semidefinite
    programming (SDP) problem:
        min. <C, X>
            s.t. <A[i], X> = b[i];
                         tr(X) = 1;
                         X >= 0,
    constraints and objective function matrix (C) are zero after initialization.
    A[i] for all i and C must be symmetric and normalized in the following way:
    |C|_F = 1, |A|_2 = 1, where A[x] = [<A[0], x>, ..., <A[`constraints_number` - 1], x>],
    i.e. A is a linear map that maps a matrix x to a vector of size `constraints_number`.
    |A[0]|_F = |A[1]|_F = .. = |A[`constraints_number` - 1]|_F, for more details see
    https://arxiv.org/pdf/1912.02949.
    Args:
        constraints_number: number of constraints;
        variables_number: number of variables."""

    def __init__(self, constraints_number: int, variables_number: int) -> None: ...
    """A value insertion to a particular position in the objective function matrix.
    Args:
        row_idx: a row index of the insertion;
        col_idx: a column index of the insertion;
        value: a value to be inserted."""
    def add_element_to_objective_matrix(
        self, row_idx: int, col_idx: int, value: float
    ) -> None: ...
    """A value insertion to the vector b.
    Args:
        constraint_number: a number of the constraint (starts from 0);
        value: a value to be inserted."""
    def add_element_to_b_vector(self, constraint_number: int, value: float) -> None: ...
    """MaxCut constraints initialization, i.e. sets constraint matrices and b vector in such a way that:
    X[i, i] = 1 / `constraints_number`."""
    def add_maxcut_constraints(self) -> None: ...
    """A value insertion to a particular position in the particular constraint matrix.
    Args:
        constraint_number: a number of the constraint (starts from 0);
        row_idx: a row index of the insertion;
        col_idx: a column index of the insertion;
        value: a value to be inserted."""
    def add_element_to_constraint_matrix(
        self, constraint_number: int, row_idx: int, col_idx: int, element: float
    ) -> None: ...
    """Creation of the SPD task, after calling this method the builder class
    is reinitialized (all matrices are zero).
    Returns:
        sdp task that can be sent to a solver."""
    def build(self) -> SDPF64: ...
    """Normalizes the objective function matrix in such a way that |C|_F = 1"""
    def normalize_objective_matrix(self) -> None: ...
    def constraints_number(self) -> int: ...
    def variables_number(self) -> int: ...
