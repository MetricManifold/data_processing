//! Cell-simulation trajectory analysis library.
//!
//! Layout (each module owns one stage of the pipeline):
//!
//! ```text
//! Layer 6: studies      declarative TOML driver
//! Layer 5: panels       typed plotters renderers
//! Layer 4: aggregate    groupby / mean_stderr / sweep / pair_*
//! Layer 3: analyze_run  RunSpec -> RunAnalysis (observable bag + metadata)
//! Layer 2: observables  typed observable registry (one file per metric)
//! Layer 1: discovery    filesystem -> RunSpecs
//! Layer 0: io / checkpoint / metadata    raw format readers
//! ```

pub mod aggregate;
pub mod analyze_run;
pub mod checkpoint;
pub mod discovery;
pub mod io;
pub mod merge_checkpoint;
pub mod metadata;
pub mod observable;
pub mod observables;
pub mod panels;
pub mod studies;
