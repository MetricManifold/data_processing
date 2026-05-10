//! Example: van_hove
//!
//! G_s(Δx, t) at three lags.
//!
//! Run:
//!   cargo run --release --example van_hove
//!   cargo run --release --example van_hove -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::van_hove::VanHove;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== van_hove on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = VanHove::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
