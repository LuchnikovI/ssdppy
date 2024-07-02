mod errors;
mod sdp;
mod sdp_builder;
mod sdppy;

use pyo3::prelude::*;
use pyo3::PyResult;

#[pymodule]
fn sparse_sdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sdppy::SDPBuilderF32>()?;
    m.add_class::<sdppy::SDPBuilderF64>()?;
    m.add_class::<sdppy::SDPF32>()?;
    m.add_class::<sdppy::SDPF64>()?;
    Ok(())
}
