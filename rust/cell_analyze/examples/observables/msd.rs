//! Example: msd
//!
//! Ensemble + cell-0 MSD(Δt).
//!
//! Run:
//!   cargo run --release --example msd
//!   cargo run --release --example msd -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::msd::Msd;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== msd on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = Msd.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
