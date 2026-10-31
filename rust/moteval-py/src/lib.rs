//! PyO3 module behind the `moteval` Python package.
//!
//! The results-cli-bindings package fills this module: every public name in
//! `moteval/__init__.py`, arrays through the `numpy` crate, and the GIL released
//! during `evaluate`.

use pyo3::prelude::*;

#[pymodule]
fn _moteval(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
