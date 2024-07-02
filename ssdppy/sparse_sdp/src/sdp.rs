use super::errors::SDPError;
use super::sdp_builder::SDPBuilder;
use num_complex::ComplexFloat;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::AddAssign;

#[derive(Debug, Clone)]
pub(super) struct Sdp<T> {
    data: Vec<T>,
    src_row_idx: Vec<usize>,
    dst_row_idx: Vec<usize>,
    batch_starts: Vec<usize>,
    batch_size: usize,
    matrix_size: usize,
}

impl<T: Copy + Clone + Debug> TryFrom<SDPBuilder<T>> for Sdp<T> {
    type Error = SDPError;
    #[inline(always)]
    fn try_from(value: SDPBuilder<T>) -> Result<Self, Self::Error> {
        let mut data = Vec::<T>::with_capacity(value.coordinates.len());
        let mut src_row_idx = Vec::<usize>::with_capacity(value.coordinates.len());
        let mut dst_row_idx = Vec::<usize>::with_capacity(value.coordinates.len());
        let mut batch_starts = vec![0];
        let mut curr_batch_idx = 0;
        for (pos, (&(batch_idx, row_idx, col_idx), &elem)) in value.coordinates.iter().enumerate() {
            match batch_idx.cmp(&(curr_batch_idx + 1)) {
                Ordering::Equal => {
                    batch_starts.push(pos);
                    curr_batch_idx += 1;
                }
                Ordering::Greater => return Err(SDPError::ZeroConstraint(curr_batch_idx)),
                Ordering::Less => {}
            }
            src_row_idx.push(col_idx);
            dst_row_idx.push(row_idx);
            data.push(elem);
        }
        batch_starts.push(data.len());
        assert_eq!(batch_starts.len(), value.batch_size + 1);
        Ok(Self {
            data,
            src_row_idx,
            dst_row_idx,
            batch_starts,
            batch_size: value.batch_size,
            matrix_size: value.matrix_size,
        })
    }
}

impl<T> Sdp<T>
where
    T: ComplexFloat + AddAssign + Sync + Send,
{
    #[inline(always)]
    pub(super) unsafe fn apply_single_matrix(
        &self,
        src: &[T],
        dst: &mut [T],
        alpha: T,
        batch_index: usize,
    ) {
        let start = self.batch_starts[batch_index];
        let end = self.batch_starts[batch_index + 1];
        let dst_ptr = dst.as_mut_ptr();
        for ((&elem, i_row_dst), i_row_src) in self.data[start..end]
            .iter()
            .zip(&self.dst_row_idx[start..end])
            .zip(&self.src_row_idx[start..end])
        {
            *dst_ptr.add(*i_row_dst) += *src.get_unchecked(*i_row_src) * elem * alpha;
        }
    }
    #[inline(always)]
    pub(super) unsafe fn compute_single_bracket(
        &self,
        src: &[T],
        alpha: T,
        batch_index: usize,
    ) -> T {
        let start = self.batch_starts[batch_index];
        let end = self.batch_starts[batch_index + 1];
        let mut result = T::zero();
        for ((&elem, i_row_dst), i_row_src) in self.data[start..end]
            .iter()
            .zip(&self.dst_row_idx[start..end])
            .zip(&self.src_row_idx[start..end])
        {
            result +=
                *src.get_unchecked(*i_row_dst) * *src.get_unchecked(*i_row_src) * elem * alpha;
        }
        result
    }
    #[inline(always)]
    pub(super) unsafe fn apply_weighted_matrix(
        &self,
        src: &[T],
        weights: &[T],
        dst: &mut [T],
        alpha: T,
        batch_index_start: usize,
    ) {
        for (batch_index, &weight) in (batch_index_start..).zip(weights) {
            self.apply_single_matrix(src, dst, alpha * weight, batch_index);
        }
    }
    #[inline(always)]
    pub(super) unsafe fn compute_brackets(
        &self,
        src: &[T],
        dst: &mut [T],
        alpha: T,
        batch_index_start: usize,
    ) {
        for (batch_index, dst) in (batch_index_start..).zip(dst) {
            *dst += self.compute_single_bracket(src, alpha, batch_index);
        }
    }
    #[inline(always)]
    pub(super) fn get_variables_number(&self) -> usize {
        self.matrix_size
    }
    #[inline(always)]
    pub(super) fn get_constraints_number(&self) -> usize {
        self.batch_size - 1
    }
    #[inline(always)]
    pub(super) fn dtype_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
    #[inline(always)]
    pub(super) fn get_non_zero_elements_number(&self) -> usize {
        self.data.len()
    }
}
