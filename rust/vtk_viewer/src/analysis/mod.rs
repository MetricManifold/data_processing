//! Shared analysis library for cell simulation trajectory data.
//!
//! Provides trajectory I/O, observable computation, batch processing,
//! and structured JSON output. Used by the `cell_analyze` binary.

pub mod batch;
pub mod checkpoint;
pub mod io;
pub mod observables;
pub mod output;
pub mod study;
