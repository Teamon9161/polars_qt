mod compose;
#[cfg(feature = "equity")]
mod equity;
mod if_then;
mod rolling_rank;
#[cfg(feature = "strategy")]
mod strategy;

use pyo3::types::PyModule;
use pyo3::{pymodule, Bound, PyResult, Python};

#[pymodule]
fn polars_qt(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
