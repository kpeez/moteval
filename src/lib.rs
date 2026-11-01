#![forbid(unsafe_code)]

pub mod benchmarks;
pub mod cli;
pub mod convert;
pub mod eval;
pub mod formats;
pub mod lsap;
pub mod metrics;
pub mod model;
pub mod numeric;
pub mod protocol;
pub mod pyfmt;
pub mod results;
pub mod rle;
pub mod similarity;

#[cfg(feature = "python")]
mod python;
