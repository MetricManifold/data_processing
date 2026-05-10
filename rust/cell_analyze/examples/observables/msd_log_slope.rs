//! Example: msd_log_slope
//!
//! Δ(t) instantaneous diffusion exponent.
//!
//! Run:
//!   cargo run --release --example msd_log_slope
//!   cargo run --release --example msd_log_slope -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::msd_log_slope::MsdLogSlope;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== msd_log_slope on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = MsdLogSlope.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
