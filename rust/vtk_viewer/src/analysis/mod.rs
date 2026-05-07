//! Shared analysis library for cell simulation trajectory data.
//!
//! Trajectory I/O, checkpoint parsing, sim-marker metadata, and the v2
//! pipeline (observables / aggregators / panels / studies). The v2
//! pipeline is the only public surface; the foundation modules
//! ([`io`], [`checkpoint`], [`metadata`]) are reused directly by the
//! `snapshot` and `check` subcommands of `cell_analyze`.

pub mod checkpoint;
pub mod io;
pub mod metadata;

/// Declarative analysis pipeline. See `ARCHITECTURE.md`.
pub mod v2;
