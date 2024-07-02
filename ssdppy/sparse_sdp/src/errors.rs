use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::error::Error;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SDPError {
    RepeatedPosition {
        constraint_number: Option<usize>,
        row_idx: usize,
        col_idx: usize,
    },
    RowOutOfBound {
        variables_number: usize,
        row_idx: usize,
    },
    ColOutOfBound {
        variables_number: usize,
        col_idx: usize,
    },
    ConstraintNumberOutOfBound {
        total_constraints_number: usize,
        constraint_number: usize,
    },
    ConstraintsNumberBSizeMismatch {
        total_constraints_number: usize,
        b_size: usize,
    },
    ZeroConstraint(usize),
}

impl Display for SDPError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SDPError::ZeroConstraint(number) => {
                write!(f, "Constraint number {number} is zero")?;
            }
            SDPError::RepeatedPosition {
                constraint_number,
                row_idx,
                col_idx,
            } => {
                if let &Some(constraint_number) = constraint_number {
                    write!(f, "An element in the constraint number: {constraint_number} with row index: {row_idx} and column index: {col_idx} already exists")?;
                } else {
                    write!(f, "An element in the objective matrix with row index: {row_idx} and column index: {col_idx} already exists")?;
                }
            }
            SDPError::ConstraintNumberOutOfBound {
                total_constraints_number,
                constraint_number,
            } => {
                write!(f, "Constraint number {constraint_number} is out {total_constraints_number} total constraints number")?;
            }
            SDPError::RowOutOfBound {
                variables_number,
                row_idx,
            } => {
                write!(
                    f,
                    "Row index {row_idx} is out of total {variables_number} variables"
                )?;
            }
            SDPError::ColOutOfBound {
                variables_number,
                col_idx,
            } => {
                write!(
                    f,
                    "Column index {col_idx} is out of total {variables_number} variables"
                )?;
            }
            SDPError::ConstraintsNumberBSizeMismatch {
                total_constraints_number,
                b_size,
            } => {
                write!(f, "Size of b and number of constraints must be equal, got b size: {b_size} and constraints number: {total_constraints_number}")?
            }
        }
        Ok(())
    }
}

impl Error for SDPError {}

pub(super) type SDPResult<T> = Result<T, SDPError>;

impl From<SDPError> for PyErr {
    fn from(value: SDPError) -> Self {
        PyValueError::new_err(format!("{}", value))
    }
}
