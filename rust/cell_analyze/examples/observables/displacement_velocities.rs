//! Example: displacement_velocities
//!
//! Tagged-cell (vx, vy) + speeds.
//!
//! Run:
//!   cargo run --release --example displacement_velocities
//!   cargo run --release --example displacement_velocities -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::displacement_velocities::DisplacementVelocities;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== displacement_velocities on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = DisplacementVelocities.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
