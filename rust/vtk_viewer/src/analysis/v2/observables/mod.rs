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

pub mod bursts;
pub mod displacement_velocities;
pub mod ln_perimeter;
pub mod msd;
pub mod msd_palmieri;
pub mod velocity_distribution;

use super::observable::{Observable, ObservableBag};

/// All built-in observables we may want to compute. Used by future
/// `analyze_run` to dispatch from the TOML's `compute = [...]` list.
pub fn register_builtin() -> Vec<Box<dyn ErasedObservable>> {
    vec![
        Box::new(EraseAdaptor(msd::Msd)),
        Box::new(EraseAdaptor(msd_palmieri::MsdPalmieri)),
        Box::new(EraseAdaptor(ln_perimeter::LnPerimeter)),
        Box::new(EraseAdaptor(displacement_velocities::DisplacementVelocities)),
        Box::new(EraseAdaptor(velocity_distribution::VelocityDistribution::default())),
        Box::new(EraseAdaptor(bursts::Bursts::default())),
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

pub trait ErasedObservable: Send + Sync {
    fn id(&self) -> &'static str;
    fn requires(&self) -> Requirements;
    fn compute_into_bag(&self, ctx: &Context, bag: &mut ObservableBag) -> Result<()>;
}

struct EraseAdaptor<O: Observable>(O);

impl<O> ErasedObservable for EraseAdaptor<O>
where
    O: Observable + 'static,
    O::Output: 'static + Send + Sync,
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
}
