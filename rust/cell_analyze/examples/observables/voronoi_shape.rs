//! Example: voronoi_shape
//!
//! q = P/√A from Voronoi polygons.
//!
//! Run:
//!   cargo run --release --example voronoi_shape
//!   cargo run --release --example voronoi_shape -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::voronoi_shape::VoronoiShape;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== voronoi_shape on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = VoronoiShape.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
