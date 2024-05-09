#![allow(clippy::unused_unit)] // needed for pyo3_polars

#[cfg(feature = "equity")]
mod equity;
mod funcs;
pub(crate) mod output_func;
#[cfg(feature = "strategy")]
mod strategy;

use pyo3::types::PyModule;
use pyo3::{pymodule, Bound, PyResult, Python};

#[pymodule]
fn polars_qt(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
