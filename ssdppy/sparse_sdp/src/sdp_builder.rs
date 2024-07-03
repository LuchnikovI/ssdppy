use super::errors::{SDPError, SDPResult};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(super) struct SDPBuilder<T> {
    pub(super) coordinates: BTreeMap<(usize, usize, usize), T>,
    pub(super) batch_size: usize,
    pub(super) matrix_size: usize,
}

impl<T> SDPBuilder<T> {
    #[inline(always)]
    pub(super) fn new(constraints_number: usize, variables_number: usize) -> Self {
        Self {
            coordinates: BTreeMap::new(),
            batch_size: constraints_number + 1,
            matrix_size: variables_number,
        }
    }
    #[inline(always)]
    pub(super) fn add_element(
        &mut self,
        batch_idx: usize,
        row_idx: usize,
        col_idx: usize,
        element: T,
    ) -> SDPResult<()> {
        if batch_idx >= self.batch_size {
            return Err(SDPError::ConstraintNumberOutOfBound {
                total_constraints_number: self.batch_size - 1,
                constraint_number: batch_idx - 1,
            });
        }
        if row_idx >= self.matrix_size {
            return Err(SDPError::RowOutOfBound {
                variables_number: self.matrix_size,
                row_idx,
            });
        }
        if col_idx >= self.matrix_size {
            return Err(SDPError::ColOutOfBound {
                variables_number: self.matrix_size,
                col_idx,
            });
        }
        let coordinate = (batch_idx, row_idx, col_idx);
        if let std::collections::btree_map::Entry::Vacant(e) = self.coordinates.entry(coordinate) {
            e.insert(element);
            Ok(())
        } else if batch_idx == 1 {
            Err(SDPError::RepeatedPosition {
                constraint_number: None,
                row_idx,
                col_idx,
            })
        } else {
            Err(SDPError::RepeatedPosition {
                constraint_number: Some(batch_idx - 1),
                row_idx,
                col_idx,
            })
        }
    }
    #[inline(always)]
    pub(super) fn get_constraints_number(&self) -> usize {
        self.batch_size - 1
    }
    #[inline(always)]
    pub(super) fn get_variables_number(&self) -> usize {
        self.matrix_size
    }
    #[inline(always)]
    pub(super) fn get_non_zero_elements_number(&self) -> usize {
        self.coordinates.len()
    }
    #[inline(always)]
    pub(super) fn dtype_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}
