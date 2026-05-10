//! Example: velocity_autocorrelation
//!
//! C_v(τ) + correlation time τ_c.
//!
//! Run:
//!   cargo run --release --example velocity_autocorrelation
//!   cargo run --release --example velocity_autocorrelation -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::velocity_autocorrelation::VelocityAutocorrelation;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== velocity_autocorrelation on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = VelocityAutocorrelation.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
