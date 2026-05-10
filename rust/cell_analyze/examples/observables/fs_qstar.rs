//! Example: fs_qstar
//!
//! F_s(q*, t) + τ_α stretched-exp fit.
//!
//! Run:
//!   cargo run --release --example fs_qstar
//!   cargo run --release --example fs_qstar -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::self_scattering::SelfScattering;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== fs_qstar on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = SelfScattering::default().compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
