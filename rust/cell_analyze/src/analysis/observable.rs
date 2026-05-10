//! The `Observable` trait, the typed `ObservableBag` storage, and the
//! `Context` carried into compute functions.
//!
//! Observables are pure functions over a loaded trajectory plus per-run
//! config. Each one declares what data it needs (positions only,
//! trajectory, checkpoint, ...) and returns a typed result that
//! serializes to JSON. The result lands in an `ObservableBag` which is
//! a typed map keyed by output type — no string keys at consume time.

use anyhow::Result;
use serde::Serialize;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use crate::analysis::checkpoint::Checkpoint;
use crate::analysis::io::{Trajectory, UnwrappedPositions};

// ---------------------------------------------------------------------------
// Requirements bitset
// ---------------------------------------------------------------------------
// What an observable needs from the loaded data. analyze_run uses this to
// decide whether to load the checkpoint, etc. Bitwise flags so multiple
// requirements compose cleanly.

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Requirements: u32 {
        /// Needs `UnwrappedPositions` (the most common case).
        const POSITIONS  = 0b0000_0001;
        /// Needs the raw `Trajectory` (e.g. for v_a_inherent or L_n columns).
        const TRAJECTORY = 0b0000_0010;
        /// Needs the v7 `Checkpoint` (e.g. for per-cell γ, target radius).
        const CHECKPOINT = 0b0000_0100;
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
/// All loaded data + per-run config that observables consume.
///
/// `analyze_run` constructs one of these per simulation run, then
/// invokes each requested observable's `compute(&ctx)`. Compute
/// functions never re-read files — that's the point of the layered
/// design.
pub struct Context {
    /// Always available: unwrapped positions.
    pub positions: Arc<UnwrappedPositions>,
    /// Available if any observable requested `TRAJECTORY`.
    pub trajectory: Option<Arc<Trajectory>>,
    /// Available if any observable requested `CHECKPOINT`.
    pub checkpoint: Option<Arc<Checkpoint>>,
    /// Per-run scientific parameters.
    pub params: RunParams,
}

/// Per-run scientific parameters (tau, R, tagged cells, etc.).
///
/// Distinct from `analysis::checkpoint::SimParams` because some fields
/// (`tagged_cells`, `soft_cells`) come from the study TOML, not the
/// checkpoint. The parts that *do* come from the checkpoint are mirrored
/// here so observables don't have to reach through `ctx.checkpoint`.
#[derive(Clone, Debug, Serialize)]
pub struct RunParams {
    /// Persistence time τ (sim units).
    pub tau: f64,
    /// Target cell radius R (sim units).
    pub cell_radius: f64,
    /// Active speed v_A.
    pub v_a: f64,
    /// Cells tagged as "the cell of interest" by the study config.
    /// For Phase-1 single-cell studies this is `[0]`; for Phase-3A
    /// pairwise it's `[0, 1]`.
    pub tagged_cells: Vec<u32>,
    /// Cells tagged as "soft" (γ < γ_ref). May overlap with `tagged_cells`.
    pub soft_cells: Vec<u32>,
}

impl Default for RunParams {
    fn default() -> Self {
        Self {
            tau: 10000.0,
            cell_radius: 49.0,
            v_a: 0.01,
            tagged_cells: vec![0],
            soft_cells: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Observable trait
// ---------------------------------------------------------------------------
/// A typed observable. Implementors declare their I/O contract via
/// `Output` (the result type, must be Serialize for JSON round-trip) and
/// `requires()` (which loaded data they need).
///
/// `id()` returns a stable string identifier used in:
///   - the study TOML's `compute = [...]` list,
///   - the JSON serialization of the bag,
///   - error messages ("observable `msd` failed: ...").
///
/// Compute is invoked by `analyze_run` after data is loaded.
pub trait Observable: Send + Sync {
    /// Result type. Must be JSON-serializable so `RunAnalysis` can be
    /// dumped to disk and re-loaded by downstream tooling.
    type Output: Serialize + Send + Sync + 'static;

    /// Stable identifier. Convention: snake_case, matches the TOML name.
    fn id(&self) -> &'static str;

    /// What loaded data this observable needs.
    fn requires(&self) -> Requirements;

    /// Compute on the given context. Should be deterministic and
    /// side-effect-free.
    fn compute(&self, ctx: &Context) -> Result<Self::Output>;
}

// ---------------------------------------------------------------------------
// ObservableBag
// ---------------------------------------------------------------------------
/// Type-erased typed map: keyed by `TypeId` of the observable's `Output`.
///
/// Insert with `bag.insert::<MyObs>(result)`, look up with
/// `bag.get::<MyObs>()`. The compiler enforces that the call site knows
/// which observable's output it's asking for — no string typos.
///
/// Internally a `HashMap<TypeId, Box<dyn Any + Send + Sync>>`; the
/// downcast back to `&O::Output` is safe because we tagged on insert.
#[derive(Default)]
pub struct ObservableBag {
    inner: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl ObservableBag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a result computed by observable `O`.
    pub fn insert<O: Observable>(&mut self, value: O::Output)
    where
        O::Output: 'static + Send + Sync,
    {
        self.inner.insert(TypeId::of::<O::Output>(), Box::new(value));
    }

    /// Look up the result of observable `O`, if it was computed.
    pub fn get<O: Observable>(&self) -> Option<&O::Output>
    where
        O::Output: 'static,
    {
        self.inner
            .get(&TypeId::of::<O::Output>())
            .and_then(|b| b.downcast_ref::<O::Output>())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    // A fake observable for the bag test. Requires nothing, returns a
    // single number.
    struct DummyObs;

    #[derive(Serialize)]
    struct DummyOut {
        x: f64,
    }

    impl Observable for DummyObs {
        type Output = DummyOut;
        fn id(&self) -> &'static str {
            "dummy"
        }
        fn requires(&self) -> Requirements {
            Requirements::empty()
        }
        fn compute(&self, _ctx: &Context) -> Result<DummyOut> {
            Ok(DummyOut { x: 42.0 })
        }
    }

    #[test]
    fn bag_round_trip() {
        let mut bag = ObservableBag::new();
        bag.insert::<DummyObs>(DummyOut { x: 3.14 });
        assert_eq!(bag.len(), 1);
        let out = bag.get::<DummyObs>().expect("dummy result missing");
        assert!((out.x - 3.14).abs() < 1e-12);
    }

    #[test]
    fn bag_misses_missing() {
        let bag = ObservableBag::new();
        assert!(bag.get::<DummyObs>().is_none());
    }
}
