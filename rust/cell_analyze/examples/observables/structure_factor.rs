//! Example: structure_factor
//!
//! S(q) + first peak q*.
//!
//! Run:
//!   cargo run --release --example structure_factor
//!   cargo run --release --example structure_factor -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::structure_factor::StructureFactor;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== structure_factor on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = StructureFactor::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
