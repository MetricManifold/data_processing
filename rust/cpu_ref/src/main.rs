//! CPU reference simulator (binary entry point).
//!
//! Workflow:
//!   1. Read v7 checkpoint (`--ic`).
//!   2. Read polarities from a raw little-endian f64 binary file
//!      (`--polarities`, layout: `[px_0, py_0, px_1, py_1, …]`, 2*n f64).
//!   3. Run forward-Euler integration with the same physics as
//!      `cpu_reference.py`, snapshotting every `--save-every` steps.
//!   4. Write `cpu_traj.npz` (or whatever `--out` points to) with arrays
//!      `t (K,)`, `phi (K, n, Ny, Nx) f32`, `vx/vy/vol/px/py (K, n) f64`,
//!      and scalars `Nx, Ny, dt, dx, dy, v_A, polarity_seed`.

mod checkpoint;
mod npz;
mod sim;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;

use crate::sim::{Cell, Params, TumbleEvent, Workspace, Xoshiro256Plus};

/// Periodic Σφ²-weighted centroid (circular-mean trick on each axis).
/// Returns (cx, cy) in pixel coordinates ∈ [0, Nx) × [0, Ny).
fn periodic_centroid(phi: &[f64], nx: usize, ny: usize) -> (f64, f64) {
    let two_pi = std::f64::consts::TAU;
    let mut tot = 0.0f64;
    let mut col_psq = vec![0.0f64; nx]; // Σ_y φ² over each x column
    let mut row_psq = vec![0.0f64; ny]; // Σ_x φ² over each y row
    for iy in 0..ny {
        let row = iy * nx;
        let mut s_row = 0.0f64;
        for ix in 0..nx {
            let v = phi[row + ix];
            let psq = v * v;
            col_psq[ix] += psq;
            s_row += psq;
            tot += psq;
        }
        row_psq[iy] = s_row;
    }
    if tot <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let mut ux = 0.0; let mut vx = 0.0;
    for ix in 0..nx {
        let theta = two_pi * ix as f64 / nx as f64;
        ux += col_psq[ix] * theta.cos();
        vx += col_psq[ix] * theta.sin();
    }
    let mut uy = 0.0; let mut vy = 0.0;
    for iy in 0..ny {
        let theta = two_pi * iy as f64 / ny as f64;
        uy += row_psq[iy] * theta.cos();
        vy += row_psq[iy] * theta.sin();
    }
    let ang_x = (vx.atan2(ux) + two_pi).rem_euclid(two_pi);
    let ang_y = (vy.atan2(uy) + two_pi).rem_euclid(two_pi);
    (
        ang_x * nx as f64 / two_pi,
        ang_y * ny as f64 / two_pi,
    )
}

#[derive(Parser, Debug)]
#[command(about = "CPU reference for cell-simulation phase-field PDE", long_about = None)]
struct Cli {
    /// Path to a v7 checkpoint produced by cell_sim.
    #[arg(long)]
    ic: PathBuf,

    /// Raw little-endian f64 file: 2*n_cells values, [px0,py0,px1,py1,...]
    /// Optional. If omitted, initial polarities are drawn from the per-cell
    /// PRNG seeded by `--polarity-seed`.
    #[arg(long)]
    polarities: Option<PathBuf>,

    #[arg(long, default_value_t = 0.014)]
    v_a: f64,

    /// Run-and-tumble persistence time. tau <= 0 disables tumbling.
    #[arg(long, default_value_t = -1.0)]
    tau: f64,

    #[arg(long, default_value_t = 1000.0)]
    t_end: f64,

    /// dt override; if `< 0` (default), use the dt from the checkpoint.
    #[arg(long, default_value_t = -1.0)]
    dt: f64,

    /// Snapshot every N steps. Snapshot 0 is t=0.
    #[arg(long, default_value_t = 2000)]
    save_every: u64,

    /// Output `.npz` path.
    #[arg(long)]
    out: PathBuf,

    /// Optional file: append every tumble event "t cid old_theta new_theta".
    #[arg(long)]
    events: Option<PathBuf>,

    /// Pre-determined tumble events for deterministic replay (mirrors
    /// the GPU's `--scripted-events`). When set, the per-step PRNG tumble
    /// path is bypassed entirely. File format matches `events.txt`:
    ///   `# t cid old_theta new_theta`  (3- or 4-col; `#` lines ignored).
    #[arg(long)]
    scripted_events: Option<PathBuf>,

    /// Log every N steps.
    #[arg(long, default_value_t = 2000)]
    log_every: u64,

    /// Polarity / RNG seed. Used to initialise theta and the per-cell
    /// run-and-tumble PRNG.
    #[arg(long, default_value_t = 12345)]
    polarity_seed: i32,

    /// Number of rayon threads (0 = use rayon default = num CPUs).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Optional trajectory.txt output: one line per (t, cid, cx, cy)
    /// using a Σφ²-weighted periodic-aware centroid. Mirrors the GPU's
    /// trajectory.txt so existing comparison scripts can consume it.
    #[arg(long)]
    trajectory: Option<PathBuf>,
}

fn paint_tile_periodic(
    full: &mut [f64],
    nx: usize,
    ny: usize,
    tile: &[f32],
    tile_t: usize,
    ox: i32,
    oy: i32,
) {
    let nxi = nx as i32;
    let nyi = ny as i32;
    for ty in 0..tile_t {
        let gy = (((oy + ty as i32) % nyi) + nyi) % nyi;
        for tx in 0..tile_t {
            let gx = (((ox + tx as i32) % nxi) + nxi) % nxi;
            full[gy as usize * nx + gx as usize] = tile[ty * tile_t + tx] as f64;
        }
    }
}

fn read_polarities_bin(path: &PathBuf, n: usize) -> Result<Vec<(f64, f64)>> {
    let mut f = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw)?;
    let expected = 2 * n * 8;
    if raw.len() != expected {
        bail!(
            "polarity file size mismatch: got {} bytes, expected {} (2*n*8 for n={})",
            raw.len(), expected, n
        );
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let px = f64::from_le_bytes(raw[16 * i..16 * i + 8].try_into().unwrap());
        let py = f64::from_le_bytes(raw[16 * i + 8..16 * i + 16].try_into().unwrap());
        out.push((px, py));
    }
    Ok(out)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    let ckpt = checkpoint::read(&cli.ic)?;
    let n = ckpt.cells.len();
    let nx = ckpt.params.nx;
    let ny = ckpt.params.ny;
    let dt = if cli.dt > 0.0 { cli.dt } else { ckpt.params.dt };

    let pols: Option<Vec<(f64, f64)>> = match &cli.polarities {
        Some(path) => Some(read_polarities_bin(path, n)?),
        None => None,
    };

    let p = Params {
        nx, ny,
        dx: ckpt.params.dx,
        dy: ckpt.params.dy,
        dt,
        lambd: ckpt.params.lambd,
        gamma: ckpt.params.gamma,
        kappa: ckpt.params.kappa,
        mu: ckpt.params.mu,
        xi: ckpt.params.xi,
        target_radius: ckpt.params.target_radius,
    };
    println!(
        "[init] domain {}x{} n_cells={} dt={} γ={} λ={} κ={} μ={} ξ={} R={}",
        p.nx, p.ny, n, p.dt, p.gamma, p.lambd, p.kappa, p.mu, p.xi, p.target_radius
    );

    // Build cells: paint each checkpoint tile into a full-domain f64 field.
    let pseed = cli.polarity_seed as u64; // sign-cast preserves bits
    let polr_sidecar: Option<Vec<f32>> = ckpt.sidecars.polr.clone();
    let gama_sidecar: Option<Vec<f32>> = ckpt.sidecars.gama.clone();
    if let Some(g) = &gama_sidecar {
        let n_uniq: std::collections::BTreeSet<i64> =
            g.iter().map(|x| (*x as f64 * 1000.0).round() as i64).collect();
        println!("[init] using GAMA sidecar: {} unique values, range=[{:.4}, {:.4}]",
                 n_uniq.len(),
                 g.iter().cloned().fold(f32::INFINITY, f32::min),
                 g.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }
    if polr_sidecar.is_some() && pols.is_none() {
        println!("[init] using POLR sidecar from checkpoint for initial polarities");
    } else if pols.is_some() {
        println!("[init] using --polarities file for initial polarities");
    } else {
        println!("[init] no POLR sidecar, no --polarities; seeding theta from PRNG (seed={})",
                 cli.polarity_seed);
    }
    let mut cells: Vec<Cell> = Vec::with_capacity(n);
    for (i, c) in ckpt.cells.iter().enumerate() {
        let mut phi = vec![0.0f64; nx * ny];
        paint_tile_periodic(&mut phi, nx, ny, &c.phi_tile, ckpt.tile_t, c.ox, c.oy);
        let mut rng = Xoshiro256Plus::seed_for_cell(pseed, i as u64);
        // Initial theta priority:
        //   1. --polarities file (explicit)
        //   2. POLR sidecar in checkpoint
        //   3. PRNG (--polarity-seed)
        let (theta_init, px_init, py_init) = if let Some(pp) = &pols {
            let (px, py) = pp[i];
            (py.atan2(px), px, py)
        } else if let Some(polr) = &polr_sidecar {
            let theta = polr[i] as f64;
            (theta, theta.cos(), theta.sin())
        } else {
            let u = rng.next_f64();
            let theta = u * std::f64::consts::TAU;
            (theta, theta.cos(), theta.sin())
        };
        let cell_gamma = gama_sidecar
            .as_ref()
            .and_then(|g| g.get(i).copied())
            .map(|v| v as f64)
            .unwrap_or(ckpt.params.gamma);
        cells.push(Cell {
            phi,
            vx: c.vx as f64,
            vy: c.vy as f64,
            vol: c.volume as f64,
            v_a: cli.v_a,
            gamma: cell_gamma,
            theta: theta_init,
            px: px_init,
            py: py_init,
            rng,
        });
    }

    // Snapshot allocation.
    let n_steps = ((cli.t_end - ckpt.t) / dt).round() as u64;
    if n_steps == 0 { bail!("n_steps == 0; t_end={} t0={} dt={}", cli.t_end, ckpt.t, dt); }
    let save_every = cli.save_every.max(1);
    let k_max = (n_steps / save_every) as usize + 1;
    println!(
        "[init] n_steps={} save_every={} k_max={} (~{:.2} GB phi snapshots)",
        n_steps, save_every, k_max,
        (k_max as f64 * n as f64 * nx as f64 * ny as f64 * 4.0) / 1e9
    );

    let mut t_arr = vec![0.0f64; k_max];
    let mut phi_arr = vec![0.0f32; k_max * n * ny * nx];
    let mut vx_arr = vec![0.0f64; k_max * n];
    let mut vy_arr = vec![0.0f64; k_max * n];
    let mut vol_arr = vec![0.0f64; k_max * n];
    let mut px_arr = vec![0.0f64; k_max * n];
    let mut py_arr = vec![0.0f64; k_max * n];

    let mut snapshot = |k: usize, t: f64, cs: &[Cell], phi_arr: &mut [f32],
                        vx_a: &mut [f64], vy_a: &mut [f64], vol_a: &mut [f64],
                        px_a: &mut [f64], py_a: &mut [f64], t_a: &mut [f64]| {
        t_a[k] = t;
        let stride_pix = ny * nx;
        for (i, c) in cs.iter().enumerate() {
            let base = (k * n + i) * stride_pix;
            for j in 0..stride_pix {
                phi_arr[base + j] = c.phi[j] as f32;
            }
            vx_a[k * n + i] = c.vx;
            vy_a[k * n + i] = c.vy;
            vol_a[k * n + i] = c.vol;
            px_a[k * n + i] = c.px;
            py_a[k * n + i] = c.py;
        }
    };

    snapshot(0, ckpt.t, &cells,
             &mut phi_arr, &mut vx_arr, &mut vy_arr, &mut vol_arr,
             &mut px_arr, &mut py_arr, &mut t_arr);

    // Optional trajectory.txt file (mirrors GPU layout: "# t cid cx cy" header,
    // one whitespace-separated line per (t, cid, cx, cy)).
    let mut traj_w: Option<BufWriter<File>> = if let Some(path) = &cli.trajectory {
        let f = File::create(path)
            .with_context(|| format!("creating trajectory file {:?}", path))?;
        let mut bw = BufWriter::new(f);
        writeln!(bw, "# t cid cx cy")?;
        // initial frame
        for (i, c) in cells.iter().enumerate() {
            let (cx, cy) = periodic_centroid(&c.phi, nx, ny);
            writeln!(bw, "{:.10e} {} {:.10e} {:.10e}", ckpt.t, i, cx, cy)?;
        }
        Some(bw)
    } else { None };

    let mut ws = Workspace::new(n, nx, ny);
    let mut events: Vec<TumbleEvent> = Vec::new();
    // Open events file (truncate). Header: column names.
    let mut events_w: Option<BufWriter<File>> = if let Some(path) = &cli.events {
        let f = File::create(path)
            .with_context(|| format!("creating events file {:?}", path))?;
        let mut bw = BufWriter::new(f);
        writeln!(bw, "# t cid old_theta new_theta")?;
        Some(bw)
    } else { None };

    // ---- Optional: load scripted (pre-determined) tumble events. ----
    // File format: "# t cid [old_theta] new_theta" (3- or 4-col; '#' ignored).
    // Each event's t is converted to step_i = round((t - ckpt.t) / dt).
    // When scripted_active, the PRNG tumble path is bypassed (tau passed as
    // 0.0 to step()), and tumbles are applied directly here.
    let scripted: Vec<(u64, u32, f64)> = if let Some(path) = &cli.scripted_events {
        let f = File::open(path)
            .with_context(|| format!("opening scripted-events file {:?}", path))?;
        let reader = BufReader::new(f);
        let mut evs: Vec<(u64, u32, f64)> = Vec::new();
        for (lineno, line) in reader.lines().enumerate() {
            let line = line?;
            let s = line.trim();
            if s.is_empty() || s.starts_with('#') { continue; }
            let toks: Vec<f64> = s.split_whitespace()
                .filter_map(|t| t.parse::<f64>().ok())
                .collect();
            let (t, cid, new_theta) = match toks.len() {
                3 => (toks[0], toks[1] as i32, toks[2]),
                4 => (toks[0], toks[1] as i32, toks[3]),
                k => bail!("scripted-events {:?} line {}: expected 3 or 4 cols, got {}",
                           path, lineno+1, k),
            };
            if cid < 0 || cid >= n as i32 {
                bail!("scripted-events {:?} line {}: cid {} out of range (n={})",
                      path, lineno+1, cid, n);
            }
            if t <= ckpt.t {
                bail!("scripted-events {:?} line {}: t={} <= start_t={}",
                      path, lineno+1, t, ckpt.t);
            }
            let step_i = ((t - ckpt.t) / dt).round() as u64;
            evs.push((step_i, cid as u32, new_theta));
        }
        evs.sort_by_key(|&(s, c, _)| (s, c));
        println!("[scripted] {} events loaded from {:?} (PRNG tumble path disabled)",
                 evs.len(), path);
        evs
    } else { Vec::new() };
    let scripted_active = !scripted.is_empty();
    let effective_tau = if scripted_active { 0.0 } else { cli.tau };
    let mut sc_cursor: usize = 0;

    let t_start = Instant::now();
    let mut k_idx = 1usize;
    for step_i in 1..=n_steps {
        let t_after = ckpt.t + (step_i as f64) * dt;

        // Apply any scripted events whose step matches this step.
        if scripted_active {
            while sc_cursor < scripted.len() && scripted[sc_cursor].0 == step_i {
                let (_, cid, new_theta) = scripted[sc_cursor];
                let c = &mut cells[cid as usize];
                let old_theta = c.theta;
                c.theta = new_theta;
                c.px = new_theta.cos();
                c.py = new_theta.sin();
                events.push(TumbleEvent { t: t_after, cid, old_theta, new_theta });
                sc_cursor += 1;
            }
        }

        sim::step(&mut cells, &p, &mut ws, effective_tau, t_after, &mut events);

        // Flush new tumble events to file (always: cheap, low rate).
        if let Some(bw) = events_w.as_mut() {
            for ev in events.drain(..) {
                writeln!(bw, "{:.10e} {} {:.10e} {:.10e}",
                         ev.t, ev.cid, ev.old_theta, ev.new_theta)?;
            }
        } else {
            events.clear();
        }

        if step_i % save_every == 0 && k_idx < k_max {
            let t_now = t_after;
            snapshot(k_idx, t_now, &cells,
                     &mut phi_arr, &mut vx_arr, &mut vy_arr, &mut vol_arr,
                     &mut px_arr, &mut py_arr, &mut t_arr);
            if let Some(bw) = traj_w.as_mut() {
                for (i, c) in cells.iter().enumerate() {
                    let (cx, cy) = periodic_centroid(&c.phi, nx, ny);
                    writeln!(bw, "{:.10e} {} {:.10e} {:.10e}", t_now, i, cx, cy)?;
                }
            }
            k_idx += 1;
        }
        if step_i % cli.log_every == 0 {
            let elapsed = t_start.elapsed().as_secs_f64();
            let rate = step_i as f64 / elapsed;
            let eta = (n_steps - step_i) as f64 / rate / 3600.0;
            println!(
                "[step {}/{}]  t={:.2}  {:.1} steps/s  ETA {:.2} h",
                step_i, n_steps, ckpt.t + (step_i as f64) * dt, rate, eta
            );
        }
    }
    let total_s = t_start.elapsed().as_secs_f64();
    println!("[done] {} steps in {:.1} s ({:.1} steps/s)",
             n_steps, total_s, n_steps as f64 / total_s);

    // Write npz.
    println!("[write] {:?}", cli.out);
    let mut nz = npz::NpzBuilder::create(&cli.out)?;
    nz.add_f64("t", &[k_idx], &t_arr[..k_idx])?;
    nz.add_f32("phi", &[k_idx, n, ny, nx],
               &phi_arr[..k_idx * n * ny * nx])?;
    nz.add_f64("vx", &[k_idx, n], &vx_arr[..k_idx * n])?;
    nz.add_f64("vy", &[k_idx, n], &vy_arr[..k_idx * n])?;
    nz.add_f64("vol", &[k_idx, n], &vol_arr[..k_idx * n])?;
    nz.add_f64("px", &[k_idx, n], &px_arr[..k_idx * n])?;
    nz.add_f64("py", &[k_idx, n], &py_arr[..k_idx * n])?;
    nz.add_scalar_i32("Nx", nx as i32)?;
    nz.add_scalar_i32("Ny", ny as i32)?;
    nz.add_scalar_f64("dt", dt)?;
    nz.add_scalar_f64("dx", p.dx)?;
    nz.add_scalar_f64("dy", p.dy)?;
    nz.add_scalar_f64("v_A", cli.v_a)?;
    nz.add_scalar_i32("polarity_seed", cli.polarity_seed)?;
    nz.finish()?;

    if let Some(mut bw) = traj_w {
        bw.flush()?;
    }
    if let Some(mut bw) = events_w {
        bw.flush()?;
    }

    let _ = (BufWriter::new(std::io::stdout()).flush(),);
    Ok(())
}
