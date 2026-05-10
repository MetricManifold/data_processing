//! Example: cage_length
//!
//! L_c from MSD plateau (min Δ near τ).
//!
//! Run:
//!   cargo run --release --example cage_length
//!   cargo run --release --example cage_length -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::cage_length::CageLength;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== cage_length on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = CageLength.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
