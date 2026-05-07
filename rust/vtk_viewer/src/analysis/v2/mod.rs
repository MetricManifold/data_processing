//! v2 analysis pipeline (in-progress migration).
//!
//! See [ARCHITECTURE.md](../../../ARCHITECTURE.md) for the full plan.
//!
//! ```text
//! Layer 6: Studies     (declarative TOML, no per-study Rust)
//! Layer 5: Panels      (typed Data input, plotters backend)
//! Layer 4: Aggregators (groupby, mean_stderr, sweep, ...)
//! Layer 3: analyze_run (RunSpec -> RunAnalysis with ObservableBag)
//! Layer 2: Observables (typed registry, pure functions)
//! Layer 1: Discovery   (filesystem -> RunSpecs)
//! Layer 0: I/O         (UNCHANGED, reused from analysis::{io, checkpoint})
//! ```
//!
//! During migration this lives under `analysis::v2` so the legacy
//! `analysis::observables` / `analysis::study` paths stay untouched. At
//! cutover (phase 9) v2 is promoted to a top-level `pipeline::` module
//! and the legacy code is deleted.

pub mod aggregate;
pub mod analyze_run;
pub mod discovery;
pub mod observable;
pub mod observables;
pub mod panels;
pub mod studies;
