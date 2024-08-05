use crate::{sdp::Sdp, sdp_builder::SDPBuilder};
use num_traits::{One, Zero};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, Bound, Py, PyErr, PyResult, Python};
use std::borrow::BorrowMut;

macro_rules! impl_sdppy {
    ($dtype:ident, $builder_name:ident, $name:ident) => {
        #[pyclass]
        pub struct $builder_name {
            builder: SDPBuilder<$dtype>,
            b: Py<PyArray1<$dtype>>,
            objective_norm: $dtype,
        }

        #[pymethods]
        impl $builder_name {
            #[new]
            pub fn new(
                py: Python<'_>,
                constraints_number: usize,
                variables_number: usize,
            ) -> PyResult<Self> {
                let b = PyArray1::zeros_bound(py, [constraints_number], false).into();
                Ok(Self {
                    builder: SDPBuilder::new(constraints_number, variables_number),
                    b,
                    objective_norm: $dtype::one(),
                })
            }
            pub fn add_element_to_objective_matrix(
                &mut self,
                row_idx: usize,
                col_idx: usize,
                value: $dtype,
            ) -> PyResult<()> {
                self.builder
                    .add_element(0, row_idx, col_idx, value)
                    .map_err(|err| err.into())
            }
            pub fn add_element_to_constraint_matrix(
                &mut self,
                constraint_number: usize,
                row_idx: usize,
                col_idx: usize,
                element: $dtype,
            ) -> PyResult<()> {
                self.builder
                    .add_element(constraint_number + 1, row_idx, col_idx, element)
                    .map_err(|err| err.into())
            }
            pub fn normalize_objective_matrix(&mut self) {
                let mut norm = $dtype::zero();
                for (key, val) in self.builder.coordinates.iter() {
                    if key.0 != 0 {
                        break;
                    }
                    norm += val * val;
                }
                norm = norm.sqrt();
                self.objective_norm = norm;
                for (key, val) in self.builder.coordinates.iter_mut() {
                    if key.0 != 0 {
                        break;
                    }
                    *val /= norm;
                }
            }
            pub fn add_element_to_b_vector(
                &mut self,
                py: Python<'_>,
                constraint_number: usize,
                value: $dtype,
            ) -> PyResult<()> {
                unsafe {
                    let b_slice = self.b.bind(py).borrow_mut().as_slice_mut()?;
                    if let Some(v) = b_slice.get_mut(constraint_number) {
                        *v = value;
                    } else {
                        let total_constraints_number = self.builder.get_constraints_number();
                        return Err(PyValueError::new_err(format!("Constraint number {constraint_number} is out of total constraints number {total_constraints_number}")))
                    }
                }
                Ok(())
            }
            pub fn add_maxcut_constraints(
                &mut self,
                py: Python<'_>,
            ) -> PyResult<()> {
                let constraints_number = self.builder.get_constraints_number();
                let variables_number = self.builder.get_variables_number();
                if constraints_number != variables_number {
                    return Err(PyValueError::new_err(format!("Cannot initialize maxcut constraints since the number of constraints {constraints_number} does not match the variables number {variables_number}")))
                }
                let b_val = 1. / constraints_number as $dtype;
                let b_slice = unsafe { self.b.bind(py).borrow_mut().as_slice_mut()? };
                assert_eq!(b_slice.len(), constraints_number);
                for b_dst in b_slice.iter_mut() {
                    *b_dst = b_val;
                }
                for constr_num in 0..constraints_number {
                    self.add_element_to_constraint_matrix(constr_num, constr_num, constr_num, $dtype::one())?;
                }
                Ok(())
            }
            pub fn build(&mut self) -> PyResult<$name> {
                let empty_builder = SDPBuilder::new(
                    self.builder.get_constraints_number(),
                    self.builder.get_variables_number(),
                );
                let builder = std::mem::replace(&mut self.builder, empty_builder);
                Ok($name {
                    sdp: Sdp::try_from(builder).map_err(|err| Into::<PyErr>::into(err))?,
                    b: self.b.clone(),
                    norm: self.objective_norm,
                })
            }
            #[getter]
            fn constraints_number(&self) -> usize {
                self.builder.get_constraints_number()
            }
            #[getter]
            fn variables_number(&self) -> usize {
                self.builder.get_variables_number()
            }
            fn __repr__(&self) -> String {
                format!(
                    "sparse_sdp_builder:\n\tdtype_name: {}\n\tconstraints_number: {}\n\tvariables_number: {}\n\ttotal_non_zero_elements_number: {}",
                    self.builder.dtype_name(),
                    self.builder.get_constraints_number(),
                    self.builder.get_variables_number(),
                    self.builder.get_non_zero_elements_number(),
                )
            }
        }

        #[pyclass]
        pub struct $name {
            sdp: Sdp<$dtype>,
            b: Py<PyArray1<$dtype>>,
            norm: $dtype,
        }

        #[pymethods]
        impl $name {
            fn _apply_objective_matrix<'py>(
                &self,
                src: Bound<'py, PyArray1<$dtype>>,
                dst: Bound<'py, PyArray1<$dtype>>,
                alpha: $dtype,
                beta: $dtype,
            ) {
                unsafe {
                    let src = src.as_slice().unwrap();
                    let dst = dst.as_slice_mut().unwrap();
                    assert_eq!(src.len(), self.sdp.get_variables_number());
                    assert_eq!(dst.len(), self.sdp.get_variables_number());
                    if beta != $dtype::one() {
                        dst.iter_mut().for_each(|x| *x *= beta);
                    }
                    self.sdp.apply_single_matrix(src, dst, alpha, 0);
                }
            }
            #[getter]
            fn _objective_norm(&self) -> $dtype {
                self.norm
            }
            fn _apply_weighted_constraints<'py>(
                &self,
                src: Bound<'py, PyArray1<$dtype>>,
                weights: Bound<'py, PyArray1<$dtype>>,
                dst: Bound<'py, PyArray1<$dtype>>,
                alpha: $dtype,
                beta: $dtype,
            ) {
                unsafe {
                    let src = src.as_slice().unwrap();
                    let weights = weights.as_slice().unwrap();
                    let dst = dst.as_slice_mut().unwrap();
                    assert_eq!(src.len(), self.sdp.get_variables_number());
                    assert_eq!(dst.len(), self.sdp.get_variables_number());
                    assert_eq!(weights.len(), self.sdp.get_constraints_number());
                    if beta != $dtype::one() {
                        dst.iter_mut().for_each(|x| *x *= beta);
                    }
                    self.sdp.apply_weighted_matrix(src, weights, dst, alpha, 1);
                }
            }
            fn _compute_brackets<'py>(
                &self,
                src: Bound<'py, PyArray1<$dtype>>,
                dst: Bound<'py, PyArray1<$dtype>>,
                alpha: $dtype,
                beta: $dtype,
            ) {
                unsafe {
                    let src = src.as_slice().unwrap();
                    let dst = dst.as_slice_mut().unwrap();
                    assert_eq!(src.len(), self.sdp.get_variables_number());
                    assert_eq!(dst.len(), self.sdp.get_constraints_number());
                    if beta != $dtype::one() {
                        dst.iter_mut().for_each(|x| *x *= beta);
                    }
                    self.sdp.compute_brackets(src, dst, alpha, 1);
                }
            }
            fn _compute_infeasibility(
                &self,
                py: Python<'_>,
                u: Bound<'_, PyArray2<$dtype>>,
                s: Bound<'_, PyArray1<$dtype>>,
            ) -> PyResult<$dtype> {
                let u_shape = u.shape();
                let lda = u_shape[0];
                let cols_num = u_shape[1];
                let u = unsafe { u.as_slice().unwrap() };
                let s = unsafe { s.as_slice().unwrap() };
                let b = self.b.bind(py);
                let b_slice = unsafe { b.as_slice().unwrap() };
                assert_eq!(lda, self.variables_number());
                assert_eq!(cols_num, s.len());
                let mut frob_dist_sq = $dtype::zero();
                let mut frob_b_sq = $dtype::zero();
                for constr_num in 0..self.constraints_number() {
                    let mut trace_val = $dtype::zero();
                    for i in 0..cols_num {
                        let col = &u[(i * lda)..((i + 1) * lda)];
                        trace_val += unsafe { self.sdp.compute_single_bracket(col, s[i], constr_num + 1) };
                    }
                    frob_dist_sq += (b_slice[constr_num] - trace_val).powi(2);
                    frob_b_sq += b_slice[constr_num].powi(2);
                }
                Ok(frob_dist_sq.sqrt() / frob_b_sq.sqrt())
            }
            fn compute_objective_value(
                &self,
                u: Bound<'_, PyArray2<$dtype>>,
                s: Bound<'_, PyArray1<$dtype>>,
            ) -> PyResult<$dtype> {
                let u_shape = u.shape();
                let lda = u_shape[0];
                let cols_num = u_shape[1];
                let u = unsafe { u.as_slice().unwrap() };
                let s = unsafe { s.as_slice().unwrap() };
                assert_eq!(lda, self.variables_number());
                assert_eq!(cols_num, s.len());
                let mut objective_value = $dtype::zero();
                for i in 0..cols_num {
                    let col = &u[(i * lda)..((i + 1) * lda)];
                    objective_value += unsafe { self.sdp.compute_single_bracket(col, s[i], 0) };
                }
                Ok(objective_value * self.norm)
            }
            #[getter]
            fn constraints_number(&self) -> usize {
                self.sdp.get_constraints_number()
            }
            #[getter]
            fn variables_number(&self) -> usize {
                self.sdp.get_variables_number()
            }
            fn _get_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$dtype>> {
                self.b.clone().into_bound(py)
            }
            fn __repr__(&self) -> String {
                format!(
                    "sparse_sdp:\n\tdtype_name: {}\n\tconstraints_number: {}\n\tvariables_number: {}\n\ttotal_non_zero_elements_number: {}",
                    self.sdp.dtype_name(),
                    self.sdp.get_constraints_number(),
                    self.sdp.get_variables_number(),
                    self.sdp.get_non_zero_elements_number(),
                )
            }
        }
    };
}

impl_sdppy!(f32, SDPBuilderF32, SDPF32);
impl_sdppy!(f64, SDPBuilderF64, SDPF64);
