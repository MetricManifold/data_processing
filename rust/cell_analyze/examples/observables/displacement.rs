//! Example: displacement
//!
//! Net Δr first→last frame summary.
//!
//! Run:
//!   cargo run --release --example displacement
//!   cargo run --release --example displacement -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::displacement::Displacement;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== displacement on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = Displacement.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
