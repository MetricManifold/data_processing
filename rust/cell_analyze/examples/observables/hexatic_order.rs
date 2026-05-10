//! Example: hexatic_order
//!
//! ψ₆ + g₆(r) hexatic order.
//!
//! Run:
//!   cargo run --release --example hexatic_order
//!   cargo run --release --example hexatic_order -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::hexatic_order::HexaticOrder;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== hexatic_order on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = HexaticOrder.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
