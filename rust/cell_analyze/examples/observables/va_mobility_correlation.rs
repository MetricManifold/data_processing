//! Example: va_mobility_correlation
//!
//! Pearson(inherent v_A, time-avg speed).
//!
//! Run:
//!   cargo run --release --example va_mobility_correlation
//!   cargo run --release --example va_mobility_correlation -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::va_mobility::VaMobilityCorrelation;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== va_mobility_correlation on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = VaMobilityCorrelation.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
