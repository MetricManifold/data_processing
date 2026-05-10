//! Example: kinetic_energy
//!
//! ½⟨v²⟩(t) per cell + total.
//!
//! Run:
//!   cargo run --release --example kinetic_energy
//!   cargo run --release --example kinetic_energy -- <run_dir>
//!
//! With no args, defaults to results/local_test/100c_rho90_ctrl/run_01.

use anyhow::Result;
use cell_analyze::analysis::observable::Observable;
use cell_analyze::analysis::observables::kinetic_energy::KineticEnergy;
use cell_analyze::demo;

fn main() -> Result<()> {
    let dir = demo::run_dir_from_args();
    println!("=== kinetic_energy on {} ===", dir.display());
    let ctx = demo::load_run(&dir)?;
    let out = KineticEnergy.compute(&ctx)?;
    demo::print_output(&out)?;
    Ok(())
}
