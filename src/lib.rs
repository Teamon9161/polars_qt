
mod if_then;
mod rolling_rank;
#[cfg(feature = "equity")]
mod equity;
#[cfg(feature = "strategy")]
mod strategy;

use pyo3::types::PyModule;
use pyo3::{pymodule, PyResult, Python};

#[pymodule]
fn polars_qt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
