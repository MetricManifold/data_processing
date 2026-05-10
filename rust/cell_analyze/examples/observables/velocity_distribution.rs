//! Example: velocity_distribution
//!
//! G(v_i) histogram + σ + kurtosis.
//!
//! Run:
//!   cargo run --release --example velocity_distribution
//!   cargo run --release --example velocity_distribution -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::velocity_distribution::VelocityDistribution;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== velocity_distribution on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = VelocityDistribution::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
