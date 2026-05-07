//! Shared analysis library for cell simulation trajectory data.
//!
//! Provides trajectory I/O, observable computation, batch processing,
//! and structured JSON output. Used by the `cell_analyze` binary.

pub mod batch;
pub mod checkpoint;
pub mod io;
pub mod metadata;
pub mod observables;
pub mod output;
pub mod panels;
pub mod study;

/// v2 architecture (in-progress migration).
/// See [crate-root ARCHITECTURE.md](../../ARCHITECTURE.md).
pub mod v2;
