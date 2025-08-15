#![allow(clippy::unused_unit)] // needed for pyo3_polars

#[cfg(feature = "equity")]
mod equity;
mod funcs;
pub(crate) mod output_func;
#[cfg(feature = "strategy")]
mod strategy;

use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{pymodule, Bound, PyResult, Python};

macro_rules! auto_cast {
    // for one expression
    ($arm: ident ($se: expr)) => {
        if let polars::prelude::DataType::$arm = $se.dtype() {
            &$se
        } else {
            &$se.cast(&polars::prelude::DataType::$arm)?
        }
    };
    // for multiple expressions
    ($arm: ident ($($se: expr),*)) => {
        ($(
            if let polars::prelude::DataType::$arm = $se.dtype() {
                $se
            } else {
                &$se.cast(&polars::prelude::DataType::$arm)?
            }
        ),*)
    };
}
pub(crate) use auto_cast;

#[pymodule]
fn polars_qt(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
