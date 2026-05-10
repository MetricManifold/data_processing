//! Example: per_cell_diffusion
//!
//! Per-cell D_i + mean/std/CV.
//!
//! Run:
//!   cargo run --release --example per_cell_diffusion
//!   cargo run --release --example per_cell_diffusion -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::per_cell_diffusion::PerCellDiffusion;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== per_cell_diffusion on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = PerCellDiffusion.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
