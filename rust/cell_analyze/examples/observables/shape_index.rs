//! Example: shape_index
//!
//! p_eff = L_n × 2√π (vertex-model).
//!
//! Run:
//!   cargo run --release --example shape_index
//!   cargo run --release --example shape_index -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::shape_index::ShapeIndex;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== shape_index on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = ShapeIndex.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
