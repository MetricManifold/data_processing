//! Concrete observables, one module per topic.
//!
//! Each observable is a unit struct implementing
//! [`super::observable::Observable`]. The struct itself carries any
//! parameters the observable needs (e.g. `Diffusion { lag_tau: f64 }`);
//! the trait `Output` is the typed result that lands in the
//! [`ObservableBag`].
//!
//! New observables are added by:
//!   1. dropping a new `.rs` file in this directory,
//!   2. adding a `pub mod foo;` line below,
//!   3. adding the observable to `register_builtin()` in this file.

pub mod alpha2;
pub mod bursts;
pub mod cage_length;
pub mod diffusion;
pub mod displacement;
pub mod displacement_velocities;
pub mod fits;
pub mod hexatic_order;
pub mod kinetic_energy;
pub mod ln_perimeter;
pub mod msd;
pub mod msd_log_slope;
pub mod msd_palmieri;
pub mod overlap;
pub mod per_cell_diffusion;
pub mod polarity_tau;
pub mod self_scattering;
pub mod shape_index;
pub mod spatial_correlation;
pub mod structure_factor;
pub mod va_mobility;
pub mod van_hove;
pub mod velocity_autocorrelation;
pub mod velocity_distribution;
pub mod voronoi_shape;

use super::observable::{Observable, ObservableBag};

/// All built-in observables we may want to compute. Used by future
/// `analyze_run` to dispatch from the TOML's `compute = [...]` list.
pub fn register_builtin() -> Vec<Box<dyn ErasedObservable>> {
    vec![
        // Core (Phase 1)
        Box::new(EraseAdaptor(msd::Msd)),
        Box::new(EraseAdaptor(msd_palmieri::MsdPalmieri)),
        Box::new(EraseAdaptor(ln_perimeter::LnPerimeter)),
        Box::new(EraseAdaptor(displacement_velocities::DisplacementVelocities)),
        Box::new(EraseAdaptor(velocity_distribution::VelocityDistribution::default())),
        Box::new(EraseAdaptor(bursts::Bursts::default())),
        // MSD-derived
        Box::new(EraseAdaptor(diffusion::Diffusion::default())),
        Box::new(EraseAdaptor(msd_log_slope::MsdLogSlope)),
        Box::new(EraseAdaptor(cage_length::CageLength)),
        Box::new(EraseAdaptor(per_cell_diffusion::PerCellDiffusion)),
        Box::new(EraseAdaptor(displacement::Displacement)),
        // Glass / jamming
        Box::new(EraseAdaptor(alpha2::NonGaussian)),
        Box::new(EraseAdaptor(overlap::Overlap::default())),
        Box::new(EraseAdaptor(structure_factor::StructureFactor::default())),
        Box::new(EraseAdaptor(self_scattering::SelfScattering::default())),
        Box::new(EraseAdaptor(van_hove::VanHove::default())),
        // Spatial / correlation
        Box::new(EraseAdaptor(spatial_correlation::SpatialCorrelation::default())),
        Box::new(EraseAdaptor(velocity_autocorrelation::VelocityAutocorrelation)),
        Box::new(EraseAdaptor(va_mobility::VaMobilityCorrelation)),
        // Geometry / order
        Box::new(EraseAdaptor(shape_index::ShapeIndex)),
        Box::new(EraseAdaptor(hexatic_order::HexaticOrder)),
        Box::new(EraseAdaptor(voronoi_shape::VoronoiShape)),
        // Polarity / energy
        Box::new(EraseAdaptor(polarity_tau::PolarityTau)),
        Box::new(EraseAdaptor(kinetic_energy::KineticEnergy)),
    ]
}

// ---------------------------------------------------------------------------
// Erased observable wrapper
// ---------------------------------------------------------------------------
// `Observable` has an associated `Output` type, which makes it
// non-object-safe. To put many observables in a single Vec we wrap each
// concrete `O: Observable` in `EraseAdaptor<O>` whose only public method
// is `compute_into_bag(ctx, bag)` — the associated type is consumed
// internally on the bag insert call and disappears from the public API.

use anyhow::Result;
use super::observable::{Context, Requirements};
use serde::Serialize;

pub trait ErasedObservable: Send + Sync {
    fn id(&self) -> &'static str;
    fn requires(&self) -> Requirements;
    fn compute_into_bag(&self, ctx: &Context, bag: &mut ObservableBag) -> Result<()>;
    fn serialize_output(&self, bag: &ObservableBag) -> Option<serde_json::Value>;
}

struct EraseAdaptor<O: Observable>(O);

impl<O> ErasedObservable for EraseAdaptor<O>
where
    O: Observable + 'static,
    O::Output: 'static + Send + Sync + Serialize,
{
    fn id(&self) -> &'static str {
        self.0.id()
    }
    fn requires(&self) -> Requirements {
        self.0.requires()
    }
    fn compute_into_bag(&self, ctx: &Context, bag: &mut ObservableBag) -> Result<()> {
        let out = self.0.compute(ctx)?;
        bag.insert::<O>(out);
        Ok(())
    }

    fn serialize_output(&self, bag: &ObservableBag) -> Option<serde_json::Value> {
        bag.get::<O>().and_then(|out| serde_json::to_value(out).ok())
    }
}
