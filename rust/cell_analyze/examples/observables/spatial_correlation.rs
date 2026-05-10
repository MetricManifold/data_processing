//! Example: spatial_correlation
//!
//! C(r) of mobility + ξ at 1/e.
//!
//! Run:
//!   cargo run --release --example spatial_correlation
//!   cargo run --release --example spatial_correlation -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::spatial_correlation::SpatialCorrelation;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== spatial_correlation on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = SpatialCorrelation::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
