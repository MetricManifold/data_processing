//! Example: ln_perimeter
//!
//! Tagged-cell L_n(t) shape index.
//!
//! Run:
//!   cargo run --release --example ln_perimeter
//!   cargo run --release --example ln_perimeter -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::ln_perimeter::LnPerimeter;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== ln_perimeter on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = LnPerimeter.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
