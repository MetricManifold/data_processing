//! `cell_analyze` library: trajectory analysis + study pipeline.
//!
//! This crate also produces a binary `cell_analyze` (TOML-driven CLI;
//! see `src/main.rs`). The library form exists so the runnable
//! `examples/` can use the analysis pipeline directly without going
//! through the TOML driver.
//!
//! Public surface:
//!   - [`analysis`]      — observables, aggregators, panels, studies.
//!   - [`vtk`]           — VTK frame parser used by the `snapshot`
//!                         subcommand.
//!   - [`demo`]          — small helpers for the runnable examples
//!                         (load a trajectory from a directory layout
//!                         and build a fully-populated `Context`).

pub mod analysis;
pub mod demo;
pub mod vtk;
