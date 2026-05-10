//! Example: polarity_tau
//!
//! Persistence τ from polarity autocorrelation.
//!
//! Run:
//!   cargo run --release --example polarity_tau
//!   cargo run --release --example polarity_tau -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::polarity_tau::PolarityTau;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== polarity_tau on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = PolarityTau.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
