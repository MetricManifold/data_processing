//! Example: alpha2
//!
//! Non-Gaussian parameter α₂(Δt).
//!
//! Run:
//!   cargo run --release --example alpha2
//!   cargo run --release --example alpha2 -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::alpha2::NonGaussian;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== alpha2 on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = NonGaussian.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
