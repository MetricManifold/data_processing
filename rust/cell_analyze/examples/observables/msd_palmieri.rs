//! Example: msd_palmieri
//!
//! Dense 0..8τ MSD/Δt + D_eff(8τ).
//!
//! Run:
//!   cargo run --release --example msd_palmieri
//!   cargo run --release --example msd_palmieri -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::msd_palmieri::MsdPalmieri;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== msd_palmieri on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = MsdPalmieri.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
