//! Example: bursts
//!
//! Speed bursts |v| > μ + k·σ.
//!
//! Run:
//!   cargo run --release --example bursts
//!   cargo run --release --example bursts -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::bursts::Bursts;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== bursts on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = Bursts::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
