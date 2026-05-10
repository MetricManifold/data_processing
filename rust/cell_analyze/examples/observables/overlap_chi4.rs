//! Example: overlap_chi4
//!
//! Self-overlap Q(t) + dynamic susceptibility χ₄.
//!
//! Run:
//!   cargo run --release --example overlap_chi4
//!   cargo run --release --example overlap_chi4 -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::overlap::Overlap;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== overlap_chi4 on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = Overlap::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
