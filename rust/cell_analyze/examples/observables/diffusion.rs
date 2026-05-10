//! Example: diffusion
//!
//! D_eff from MSD long-time slope.
//!
//! Run:
//!   cargo run --release --example diffusion
//!   cargo run --release --example diffusion -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::diffusion::Diffusion;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== diffusion on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = Diffusion::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
