use crate::{sdp::Sdp, sdp_builder::SDPBuilder};
use num_traits::One;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, Bound, Py, PyErr, PyResult, Python};
use std::borrow::BorrowMut;

macro_rules! impl_sdppy {
    ($dtype:ident, $builder_name:ident, $name:ident) => {
        #[pyclass]
        pub struct $builder_name {
            builder: SDPBuilder<$dtype>,
            b: Py<PyArray1<$dtype>>,
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
            pub fn build(&mut self) -> PyResult<$name> {
                let empty_builder = SDPBuilder::new(
                    self.builder.get_constraints_number(),
                    self.builder.get_variables_number(),
                );
                let builder = std::mem::replace(&mut self.builder, empty_builder);
                Ok($name {
                    sdp: Sdp::try_from(builder).map_err(|err| Into::<PyErr>::into(err))?,
                    b: self.b.clone(),
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
                    "sparse_sdp_builder:\n\tdtype_name: {}\n\tconstraints_number: {}\n\tvariables_number: {}\n\ttotal_non_zero_elements_number: {}",
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
