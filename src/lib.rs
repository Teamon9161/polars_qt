#[cfg(feature = "equity")]
mod equity;
mod if_then;
mod rolling_rank;

use pyo3::types::PyModule;
use pyo3::{pymodule, PyResult, Python};

#[pymodule]
fn polars_qt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
