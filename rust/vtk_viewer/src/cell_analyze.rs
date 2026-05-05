//! Unified analysis CLI for cell simulation trajectory data.
//!
//! Usage:
//!   cell_analyze run    <dir> [-o output.json] [--observables msd,overlap,...]
//!   cell_analyze study  <config.toml> -d <data_dir> [-o output.json]
//!   cell_analyze snapshot <file_or_dir> [-o output.png] [--movie] [--skip N] [--fps N]
//!   cell_analyze list
#![allow(dead_code)]

mod analysis;
mod colormap;
mod vtk;

use analysis::io::{load_trajectory, load_trajectory_subsample, unwrap_trajectory};
use analysis::observables::*;
use analysis::output::*;
use analysis::study;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "cell_analyze")]
#[command(about = "High-performance analysis for cell simulation trajectories")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a single simulation run
    Run {
        /// Simulation output directory (must contain trajectory.txt)
        dir: PathBuf,
        /// Output JSON file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Directory for SVG plots (default: no plots)
        #[arg(long)]
        plot_dir: Option<PathBuf>,
        /// Comma-separated list of observables (default: all)
        #[arg(long, value_delimiter = ',')]
        observables: Option<Vec<String>>,
        /// Persistence time τ (default: 10000)
        #[arg(long, default_value_t = 10000.0)]
        tau: f64,
        /// Cell radius R for displacement normalization (default: 49)
        #[arg(long, default_value_t = 49.0)]
        cell_radius: f64,
        /// Fraction of MSD used for D_eff fit (default: 0.3)
        #[arg(long, default_value_t = 0.3)]
        fit_frac: f64,
        /// Number of S(q) bins (default: 200)
        #[arg(long, default_value_t = 200)]
        sq_bins: usize,
        /// Number of S(q) frames to average (default: 20)
        #[arg(long, default_value_t = 20)]
        sq_frames: usize,
        /// Keep every Nth frame (default: 1 = all frames)
        #[arg(long, default_value_t = 1)]
        subsample: usize,
    },
    /// Run a TOML-defined study: discover, analyze, pair, aggregate, plot
    Study {
        /// Path to the study TOML config file
        config: PathBuf,
        /// Base directory containing simulation data
        #[arg(long, short = 'd')]
        data_dir: PathBuf,
        /// Output JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Directory for plots (default: same as output JSON)
        #[arg(long)]
        plot_dir: Option<PathBuf>,
        /// Dry run: show discovery results without analyzing
        #[arg(long)]
        dry_run: bool,
        /// Number of parallel threads (default: all available)
        #[arg(long)]
        threads: Option<usize>,
        /// Keep every Nth frame (default: 1 = all frames)
        #[arg(long, default_value_t = 1)]
        subsample: usize,
    },
    /// Render phase field snapshot(s) from checkpoint, VTK file, or directory of VTK frames
    Snapshot {
        /// Path to checkpoint.bin, frame_NNNNNN.vtk, or directory containing VTK frames
        input: PathBuf,
        /// Output PNG file (single) or directory for frames (directory mode)
        #[arg(short, long, default_value = "snapshot.png")]
        output: PathBuf,
        /// Image width in pixels (default: 800)
        #[arg(long, default_value_t = 800)]
        width: u32,
        /// Label each cell with its ID at the centroid position
        #[arg(long)]
        label_cells: bool,
        /// Movie mode: render all VTK frames in directory, then assemble with ffmpeg
        #[arg(long)]
        movie: bool,
        /// Skip every Nth frame in movie mode (default: 1 = all frames)
        #[arg(long, default_value_t = 1)]
        skip: usize,
        /// Frames per second for movie output (default: 15)
        #[arg(long, default_value_t = 15)]
        fps: u32,
        /// Color cell contours by property: auto, v_a, gamma, cell_id, none (default: auto)
        /// auto = detects from checkpoint: uses v_a if it varies, gamma if it varies, else cell_id
        #[arg(long, default_value = "auto")]
        color_by: String,
        /// Shade cell interiors by displacement speed (grayscale). Requires trajectory.txt
        #[arg(long)]
        shade_speed: bool,
        /// Speed averaging window in trajectory frames (for --shade-speed, default: 5)
        #[arg(long, default_value_t = 5)]
        speed_window: usize,
        /// Draw polarity arrows at cell centroids. Requires trajectory.txt
        #[arg(long)]
        show_polarity: bool,
        /// Overlay per-cell energy (½v²) as a heat colormap on cell interiors
        #[arg(long)]
        show_energy: bool,
        /// Emit a JSON sidecar with all banner metadata next to the PNG.
        #[arg(long)]
        emit_metadata: bool,
    },
    /// List available observables
    List,
    /// Validate trajectory/checkpoint integrity. Exits 0 on pass, 1 on any failure.
    Check {
        /// Simulation output directory (must contain trajectory.txt; checkpoint.bin optional)
        dir: PathBuf,
        /// Expected number of cells (skip check if not provided)
        #[arg(long)]
        n_cells: Option<usize>,
        /// Expected number of frames in trajectory (skip check if not provided)
        #[arg(long)]
        expected_frames: Option<usize>,
        /// Expected first timestamp (within 1% tolerance; skip if not provided)
        #[arg(long)]
        t_start: Option<f64>,
        /// Expected last timestamp (within 1% tolerance; skip if not provided)
        #[arg(long)]
        t_end: Option<f64>,
        /// Emit JSON report in addition to text
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            dir,
            output,
            plot_dir,
            observables,
            tau,
            cell_radius,
            fit_frac,
            sq_bins,
            sq_frames,
            subsample,
        } => {
            let obs = parse_observables(observables)?;
            let result = analyze_single_run(
                &dir,
                &obs,
                tau,
                cell_radius,
                fit_frac,
                sq_bins,
                sq_frames,
                subsample,
            )?;
            if let Some(ref pd) = plot_dir {
                std::fs::create_dir_all(pd)?;
                plot_run_result(&result, pd, cell_radius)?;
            }
            write_json(&result, &output)?;
        }
        Commands::Study {
            config,
            data_dir,
            output,
            plot_dir,
            dry_run,
            threads,
            subsample,
        } => {
            if let Some(n) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global()
                    .ok();
            }

            let toml_str = std::fs::read_to_string(&config)
                .with_context(|| format!("Reading study config: {}", config.display()))?;
            let study_config: study::StudyConfig = toml::from_str(&toml_str)
                .with_context(|| format!("Parsing study config: {}", config.display()))?;

            eprintln!("Study: {}", study_config.study.name);
            if !study_config.study.description.is_empty() {
                eprintln!("  {}", study_config.study.description);
            }

            if dry_run {
                let discovered = study::discover_study_runs(&data_dir, &study_config.discovery)?;
                eprintln!("\nDry run: found {} runs", discovered.len());
                for run in &discovered {
                    let vars: Vec<String> = run.variables.iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect();
                    eprintln!("  {} [{}]", run.trajectory.display(), vars.join(", "));
                }
                return Ok(());
            }

            let plot_out = plot_dir
                .or_else(|| output.as_ref().and_then(|p| p.parent().map(|pp| pp.to_path_buf())))
                .unwrap_or_else(|| PathBuf::from("."));

            let result = study::run_study(&data_dir, &study_config, &plot_out, subsample)?;

            // Print summary to stderr
            eprintln!("\n{}", "=".repeat(70));
            eprintln!("STUDY RESULTS: {}", result.study_name);
            eprintln!("{}", "=".repeat(70));
            eprintln!("Runs analyzed: {}", result.n_runs_total);
            eprintln!("Groups: {}", result.n_groups);

            if !result.paired.is_empty() {
                eprintln!("\nPaired comparisons:");
                for pg in &result.paired {
                    eprintln!("  {} ({}n/{}d):", pg.group_key, pg.numerator.n_seeds, pg.denominator.n_seeds);
                    for (name, val) in &pg.paired_metrics {
                        eprintln!("    {}: {:.4} ± {:.4}", name, val.mean, val.stderr);
                    }
                }
            }

            if !result.groups.is_empty() {
                eprintln!("\nGroups:");
                for g in &result.groups {
                    eprintln!("  {} ({} seeds):", g.group_key, g.n_seeds);
                    for (name, val) in &g.metrics {
                        eprintln!("    {}: {:.4} ± {:.4}", name, val.mean, val.stderr);
                    }
                }
            }

            write_json(&result, &output)?;
        }
        Commands::Snapshot { input, output, width: _, label_cells, movie, skip, fps, color_by, shade_speed, speed_window, show_polarity, show_energy, emit_metadata } => {
            let is_dir = input.is_dir();
            let _is_vtk = input.extension().map_or(false, |e| e == "vtk");
            // Activate per-cell rendering when user wants colored contours or speed shading
            let use_cell_render = color_by != "none" || shade_speed;

            if is_dir || movie {
                // Directory mode: render all VTK frames
                let vtk_dir = if is_dir { input.clone() } else { input.parent().unwrap().to_path_buf() };
                let mut vtk_files = vtk::find_vtk_frames(&vtk_dir)?;
                if vtk_files.is_empty() {
                    anyhow::bail!("No VTK frames found in {}", vtk_dir.display());
                }
                vtk_files.sort_by_key(|p| {
                    p.file_stem()
                        .and_then(|s| s.to_str())
                        .and_then(|s| s.strip_prefix("frame_"))
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0)
                });

                // Apply skip
                let selected: Vec<_> = vtk_files.iter().step_by(skip.max(1)).collect();
                eprintln!("Rendering {} of {} VTK frames (skip={})...", selected.len(), vtk_files.len(), skip);

                // Determine output paths
                let frames_dir = if is_dir {
                    output.clone()
                } else {
                    output.parent().unwrap_or(std::path::Path::new(".")).to_path_buf()
                };
                std::fs::create_dir_all(&frames_dir)?;

                // Get domain size from first frame for consistent rendering
                let first_vtk = vtk::parse_vtk(&selected[0])?;
                let nx = first_vtk.dims.nx;
                let ny = first_vtk.dims.ny;
                // Ensure even dimensions for h264
                let w = (nx / 2) * 2;
                let h = (ny / 2) * 2;

                // ── Per-cell rendering: preload checkpoint + trajectory ──
                let movie_ctx = if use_cell_render {
                    Some(MovieContext::load(&vtk_dir, nx, ny, &selected, speed_window, &color_by, shade_speed)?)
                } else {
                    None
                };

                if movie {
                    // Direct ffmpeg piping: stream raw RGB frames, no intermediate PNGs
                    let movie_path = frames_dir.join("movie.mp4");
                    eprintln!("Piping {} frames directly to ffmpeg at {} fps...", selected.len(), fps);
                    let ffmpeg_result = std::process::Command::new("ffmpeg")
                        .args([
                            "-y",
                            "-f", "rawvideo",
                            "-pixel_format", "rgb24",
                            "-video_size", &format!("{}x{}", w, h),
                            "-framerate", &fps.to_string(),
                            "-i", "pipe:0",
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-crf", "18",
                            &movie_path.to_string_lossy(),
                        ])
                        .stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::piped())
                        .spawn();

                    match ffmpeg_result {
                        Ok(mut proc) => {
                            {
                                let stdin = proc.stdin.as_mut().unwrap();
                                let start = std::time::Instant::now();
                                for (frame_idx, vtk_path) in selected.iter().enumerate() {
                                    let rgb = if let Some(ref ctx) = movie_ctx {
                                        ctx.render_frame(vtk_path, frame_idx, label_cells)?
                                    } else {
                                        match render_single_vtk(vtk_path, label_cells, None) {
                                            Ok((data, _nx, _ny, _)) => data,
                                            Err(e) => { eprintln!("Error rendering {}: {}", vtk_path.display(), e); continue; }
                                        }
                                    };
                                    // Crop to even dimensions if needed
                                    let frame_data = if w == nx && h == ny {
                                        rgb
                                    } else {
                                        let mut cropped = vec![0u8; w * h * 3];
                                        for y in 0..h {
                                            let src_off = y * nx * 3;
                                            let dst_off = y * w * 3;
                                            cropped[dst_off..dst_off + w * 3].copy_from_slice(&rgb[src_off..src_off + w * 3]);
                                        }
                                        cropped
                                    };
                                    if let Err(e) = stdin.write_all(&frame_data) {
                                        eprintln!("ffmpeg pipe broken at frame {}: {}", frame_idx, e);
                                        break;
                                    }
                                    if (frame_idx + 1) % 50 == 0 || frame_idx + 1 == selected.len() {
                                        let elapsed = start.elapsed().as_secs_f64();
                                        let fps_actual = (frame_idx + 1) as f64 / elapsed;
                                        eprintln!("  {}/{} frames ({:.1} fps)", frame_idx + 1, selected.len(), fps_actual);
                                    }
                                }
                            } // stdin drops here, closing the pipe
                            let status = proc.wait()?;
                            if status.success() && movie_path.exists() {
                                let size = std::fs::metadata(&movie_path).map(|m| m.len()).unwrap_or(0);
                                eprintln!("Movie saved: {} ({:.1} MB)", movie_path.display(), size as f64 / 1_048_576.0);
                            } else {
                                eprintln!("ERROR: ffmpeg failed (exit={})", status);
                            }
                        }
                        Err(e) => {
                            eprintln!("ffmpeg not found ({}), falling back to PNG output...", e);
                            // Fallback: write PNGs then assemble
                            render_frames_to_png(&selected, &frames_dir, label_cells, movie_ctx.as_ref())?;
                            assemble_movie_from_png(&frames_dir, fps)?;
                        }
                    }
                } else {
                    // Non-movie directory mode: just render PNGs
                    render_frames_to_png(&selected, &frames_dir, label_cells, movie_ctx.as_ref())?;
                }
                eprintln!("Done.");
            } else {
                // Single file mode (existing behavior)
                let is_vtk = input.extension().map_or(false, |e| e == "vtk");

                // Centroids for cell labeling: (cell_id, x, y, is_soft)
                let mut centroids: Vec<(u32, f64, f64, bool)> = Vec::new();
                // Per-cell text rendered ABOVE the ID label (e.g. "γ=0.35"
                // for cells whose stiffness differs from the population mode).
                let mut gamma_labels: std::collections::HashMap<u32, String> =
                    std::collections::HashMap::new();

                let (phi, nx, ny, _title) = if is_vtk {
                    // VTK file: parse structured points, extract "phi" field
                    let vtk_data = vtk::parse_vtk(&input)?;
                    let phi_field = vtk_data.scalars.get("phi")
                        .ok_or_else(|| anyhow::anyhow!("No 'phi' field in VTK file. Fields: {:?}", vtk_data.field_names()))?;
                    let nx = vtk_data.dims.nx;
                    let ny = vtk_data.dims.ny;
                    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    let step: i64 = stem.strip_prefix("frame_").and_then(|s| s.parse().ok()).unwrap_or(0);
                    let dt = 0.01;
                    let time = step as f64 * dt;
                    let title = format!("VTK frame: step={}, t={:.0} ({:.1}tau), {}x{}",
                                        step, time, time / 10000.0, nx, ny);
                    eprintln!("{}", title);

                    if label_cells {
                        if let Some(parent) = input.parent() {
                            let traj_path = parent.join("trajectory.txt");
                            if traj_path.exists() {
                                centroids = read_centroids_at_time(&traj_path, time);
                                eprintln!("  Loaded {} cell centroids at t={:.0}", centroids.len(), time);
                            }
                        }
                    }

                    (phi_field.clone(), nx, ny, title)
                } else {
                    // Checkpoint file
                    use analysis::checkpoint::load_checkpoint;
                    let ckpt = load_checkpoint(&input)?;
                    let phi = ckpt.composite_phi();
                    let nx = ckpt.params.nx as usize;
                    let ny = ckpt.params.ny as usize;
                    let title = format!("phi field: N={}, t={:.0} ({:.1}tau), {}x{}",
                                        ckpt.header.num_cells, ckpt.header.time,
                                        ckpt.header.time / 10000.0, nx, ny);

                    if label_cells {
                        // Determine which cells are "soft" by comparing per-cell gamma
                        // to the majority (mode) gamma. Cells with lower gamma are soft.
                        let gammas = &ckpt.per_cell_gamma;
                        let mode_gamma = if gammas.len() > 1 {
                            // Find mode: most common gamma value (1% relative tolerance)
                            let mut sorted = gammas.clone();
                            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let tol = (sorted.last().unwrap_or(&1.0) * 0.01).max(1e-6);
                            let mut best_val = sorted[0];
                            let mut best_count = 1usize;
                            let mut cur_val = sorted[0];
                            let mut cur_count = 1usize;
                            for &g in &sorted[1..] {
                                if (g - cur_val).abs() < tol {
                                    cur_count += 1;
                                } else {
                                    if cur_count > best_count { best_count = cur_count; best_val = cur_val; }
                                    cur_val = g; cur_count = 1;
                                }
                            }
                            if cur_count > best_count { best_val = cur_val; }
                            best_val
                        } else { 1.0 };

                        for (i, cell) in ckpt.cells.iter().enumerate() {
                            let tol_check = (mode_gamma * 0.01).max(1e-6);
                            let is_soft = if i < gammas.len() {
                                (gammas[i] - mode_gamma).abs() > tol_check && gammas[i] < mode_gamma
                            } else { false };
                            centroids.push((cell.id as u32, cell.centroid.0 as f64, cell.centroid.1 as f64, is_soft));
                            // Annotate cells whose gamma deviates from the mode
                            // (in either direction) with their numeric value.
                            if i < gammas.len() && (gammas[i] - mode_gamma).abs() > tol_check {
                                gamma_labels.insert(
                                    cell.id as u32,
                                    format!("{:.2}", gammas[i]),
                                );
                            }
                        }
                        let n_soft = centroids.iter().filter(|c| c.3).count();
                        eprintln!("  Loaded {} cell centroids from checkpoint ({} soft)", centroids.len(), n_soft);
                    }

                    (phi, nx, ny, title)
                };

                // Load trajectory data for polarity/energy overlays
                let overlays = if show_polarity || show_energy {
                    let traj_path = input.parent().unwrap_or(Path::new(".")).join("trajectory.txt");
                    if traj_path.exists() {
                        let traj = load_trajectory(&traj_path)?;
                        // Find the last frame
                        if let Some((_, cells)) = traj.frames.last() {
                            let mut pol: Vec<(f64, f64, f64, f64)> = Vec::new(); // (cx, cy, px, py)
                            let mut energy: Vec<(f64, f64, f64)> = Vec::new(); // (cx, cy, ke)
                            for c in cells.values() {
                                if show_polarity {
                                    pol.push((c.x, c.y, c.px, c.py));
                                }
                                if show_energy {
                                    energy.push((c.x, c.y, 0.5 * (c.vx * c.vx + c.vy * c.vy)));
                                }
                            }
                            Some((pol, energy))
                        } else { None }
                    } else {
                        eprintln!("  Warning: --show-polarity/--show-energy requires trajectory.txt");
                        None
                    }
                } else { None };

                let polarity_data = overlays.as_ref().map(|(p, _)| p.as_slice()).unwrap_or(&[]);
                let energy_data = overlays.as_ref().map(|(_, e)| e.as_slice()).unwrap_or(&[]);

                let (img_data, _, _) = render_phi_to_rgb(&phi, nx, ny, label_cells, &centroids,
                    polarity_data, energy_data, &gamma_labels);

                // Build metadata banner for checkpoint snapshots
                let (final_img, final_w, final_h) = if !is_vtk {
                    use analysis::checkpoint::load_checkpoint;
                    let ckpt = load_checkpoint(&input)?;
                    let marker = analysis::metadata::load_marker_for(&input);
                    let lines = build_metadata_lines(&ckpt, marker.as_ref(), &input);
                    for l in &lines { eprintln!("  {}", l); }

                    // Optional sidecar JSON with all metadata fields
                    if emit_metadata {
                        let p = &ckpt.params;
                        let h = &ckpt.header;
                        let lx = (p.nx as f32) * p.dx;
                        let ly = (p.ny as f32) * p.dy;
                        let phi_computed = analysis::metadata::compute_confluence(
                            h.num_cells, p.target_radius, lx, ly);
                        let phi_original = marker.as_ref()
                            .and_then(|m| analysis::metadata::marker_param_f64(m, "confluence"));
                        let stats = |xs: &[f32]| -> Option<serde_json::Value> {
                            if xs.is_empty() { return None; }
                            let mn = xs.iter().cloned().fold(f32::INFINITY, f32::min);
                            let mx = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64;
                            Some(serde_json::json!({"min": mn, "mean": mean, "max": mx}))
                        };
                        let meta = serde_json::json!({
                            "source": input.to_string_lossy(),
                            "checkpoint": {
                                "version": h.version,
                                "step": h.step,
                                "time": h.time,
                                "time_tau": h.time / 10000.0,
                                "num_cells": h.num_cells,
                            },
                            "domain": {
                                "nx": p.nx, "ny": p.ny,
                                "lx": lx, "ly": ly,
                                "dx": p.dx, "dy": p.dy,
                            },
                            "physics": {
                                "target_radius": p.target_radius,
                                "v_a_param": p.v_a,
                                "tau": p.tau,
                                "dt": p.dt,
                                "lambda": p.lambda,
                                "halo_width": p.halo_width,
                            },
                            "rng": {
                                "seed": p.seed.map(|s| serde_json::Value::Number(s.into())).unwrap_or(serde_json::Value::String("n/a".to_string())),
                                "polarity_seed": p.polarity_seed.map(|s| serde_json::Value::Number(s.into())).unwrap_or(serde_json::Value::String("n/a".to_string())),
                            },
                            "per_cell": {
                                "v_a": stats(&ckpt.per_cell_v_a),
                                "gamma": stats(&ckpt.per_cell_gamma),
                                "radius": stats(&ckpt.per_cell_radius),
                            },
                            "confluence": {
                                "computed": phi_computed,
                                "original": phi_original,
                                "delta": phi_original.map(|o| phi_computed - o),
                            },
                            "marker": marker,
                        });
                        let json_path = output.with_extension("meta.json");
                        std::fs::write(&json_path, serde_json::to_string_pretty(&meta)?)?;
                        eprintln!("Metadata saved: {}", json_path.display());
                    }

                    let (banner_img, bw, bh) = compose_with_banner(&img_data, nx, ny, &lines);
                    (banner_img, bw, bh)
                } else {
                    (img_data, nx, ny)
                };

                let out_path = if output.extension().map_or(true, |e| e != "png") {
                    output.with_extension("png")
                } else {
                    output.clone()
                };
                write_png(&out_path, &final_img, final_w, final_h)?;
                eprintln!("Snapshot saved: {} ({}×{} native)", out_path.display(), final_w, final_h);
            }
        }
        Commands::Check { dir, n_cells, expected_frames, t_start, t_end, json } => {
            let exit_code = run_check(&dir, n_cells, expected_frames, t_start, t_end, json.as_deref())?;
            std::process::exit(exit_code);
        }
        Commands::List => {
            println!("Available observables:");
            println!();
            for name in ALL_OBSERVABLES {
                let desc = match *name {
                    "msd" => "Mean squared displacement MSD(Δt)",
                    "diffusion" => "Effective diffusion coefficient D_eff from MSD slope",
                    "log_slope" => "MSD log-slope Δ(t) — diffusion exponent",
                    "cage" => "Cage length l_c from MSD plateau",
                    "alpha2" => "Non-Gaussian parameter α₂(t)",
                    "overlap" => "Self-overlap Q(t), four-point susceptibility χ₄(t), τ_α, β",
                    "structure" => "Static structure factor S(q) + peak q*",
                    "scattering" => "Self-intermediate scattering function F_s(q*, t)",
                    "van_hove" => "van Hove self-correlation G_s(Δx, t)",
                    "per_cell_diffusion" => "Per-cell diffusion coefficient D_i",
                    "displacement" => "Displacement statistics (Phase 0 quench analysis)",
                    "va_mobility_correlation" => "Pearson r between inherent v_A and time-averaged speed (σ>0 runs)",
                    "spatial_correlation" => "Spatial autocorrelation C(r) of mobility + correlation length ξ",
                    "shape_index" => "Shape index p_eff = L_n × 2√π from trajectory perimeter (vertex model)",
                    "velocity_autocorrelation" => "Velocity autocorrelation C_v(τ) and correlation time τ_c",
                    "burst_detection" => "Speed burst events (|v| > μ+3σ), frequency, duration, amplitude",
                    "velocity_distribution" => "Velocity distribution P(v_x), kurtosis for cell 0 and population",
                    "polarity_tau" => "Persistence time τ from polarity autocorrelation ⟨p̂(t+Δt)·p̂(t)⟩ = exp(-Δt/τ)",
                    "hexatic_order" => "Hexatic order ψ₆ per cell + g₆(r) orientational correlation",
                    "voronoi_shape" => "Voronoi shape index q = P/√A from Delaunay dual",
                    "kinetic_energy" => "Kinetic energy time series KE(t) = ½Σ(v²)",
                    _ => "",
                };
                println!("  {:<22} {}", name, desc);
            }
        }
    }

    Ok(())
}

fn parse_observables(input: Option<Vec<String>>) -> Result<Vec<String>> {
    match input {
        None => Ok(ALL_OBSERVABLES.iter().map(|s| s.to_string()).collect()),
        Some(names) => {
            for name in &names {
                if !is_valid_observable(name) {
                    anyhow::bail!(
                        "Unknown observable '{}'. Run 'cell_analyze list' to see options.",
                        name
                    );
                }
            }
            Ok(names)
        }
    }
}

/// Trajectory/checkpoint integrity checker. Returns process exit code.
fn run_check(
    dir: &std::path::Path,
    expected_n_cells: Option<usize>,
    expected_frames: Option<usize>,
    expected_t_start: Option<f64>,
    expected_t_end: Option<f64>,
    json_out: Option<&std::path::Path>,
) -> Result<i32> {
    use std::io::{BufRead, BufReader};

    #[derive(serde::Serialize)]
    struct CheckResult {
        name: String,
        passed: bool,
        detail: String,
    }

    let mut results: Vec<CheckResult> = Vec::new();
    let mut push = |name: &str, passed: bool, detail: String| {
        results.push(CheckResult { name: name.to_string(), passed, detail });
    };

    let traj_path = dir.join("trajectory.txt");
    if !traj_path.exists() {
        println!("FAIL: trajectory.txt not found in {}", dir.display());
        return Ok(1);
    }

    // Parse trajectory
    let f = std::fs::File::open(&traj_path)?;
    let reader = BufReader::new(f);

    let mut header_fields: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut timestamps: Vec<f64> = Vec::new();
    let mut rows_per_t: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut any_nan = false;
    let mut any_non_numeric = false;
    let mut row_count: usize = 0;

    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        if trimmed.starts_with('#') {
            for tok in trimmed.split_whitespace() {
                if let Some((k, v)) = tok.split_once('=') {
                    header_fields.insert(k.to_string(), v.to_string());
                }
            }
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 4 { continue; }
        let t = match parts[0].parse::<f64>() {
            Ok(v) => v,
            Err(_) => { any_non_numeric = true; continue; }
        };
        if !t.is_finite() { any_nan = true; continue; }
        // Check x,y for NaN
        for idx in [2usize, 3] {
            if idx < parts.len() {
                match parts[idx].parse::<f64>() {
                    Ok(v) if !v.is_finite() => any_nan = true,
                    Err(_) => any_non_numeric = true,
                    _ => {}
                }
            }
        }
        let t_bits = t.to_bits();
        *rows_per_t.entry(t_bits).or_insert(0) += 1;
        if rows_per_t[&t_bits] == 1 {
            timestamps.push(t);
        }
        row_count += 1;
    }

    // Check 1: header present with required keys
    let required_keys = ["N", "Lx", "Ly", "dim", "tau", "v_A"];
    let missing: Vec<&str> = required_keys.iter().filter(|k| !header_fields.contains_key(**k)).copied().collect();
    push("trajectory_header",
         missing.is_empty(),
         if missing.is_empty() {
             format!("all keys present: {}", required_keys.join(", "))
         } else {
             format!("MISSING keys: {}", missing.join(", "))
         });

    // Check 2: no NaN/non-numeric
    push("trajectory_no_nan", !any_nan && !any_non_numeric,
         if any_nan { "NaN/Inf found in data".to_string() }
         else if any_non_numeric { "non-numeric tokens found".to_string() }
         else { "all values finite".to_string() });

    // Check 3: timestamps strictly increasing
    let mut monotonic = true;
    let mut first_bad: Option<(usize, f64, f64)> = None;
    for i in 1..timestamps.len() {
        if timestamps[i] <= timestamps[i-1] {
            monotonic = false;
            if first_bad.is_none() {
                first_bad = Some((i, timestamps[i-1], timestamps[i]));
            }
        }
    }
    push("timestamps_monotonic", monotonic,
         if monotonic { format!("{} unique timestamps strictly increasing", timestamps.len()) }
         else { format!("NON-MONOTONIC at frame {}: {:.6} → {:.6}",
                        first_bad.unwrap().0, first_bad.unwrap().1, first_bad.unwrap().2) });

    // Check 4: rows per timestamp consistent (== N_cells from header)
    let header_n: Option<usize> = header_fields.get("N").and_then(|v| v.parse().ok());
    let expected_rows_per_frame = expected_n_cells.or(header_n);
    let mut bad_frame: Option<(f64, usize)> = None;
    if let Some(n) = expected_rows_per_frame {
        for &t in &timestamps {
            let c = rows_per_t[&t.to_bits()];
            if c != n {
                bad_frame = Some((t, c));
                break;
            }
        }
    }
    push("rows_per_frame_consistent", bad_frame.is_none(),
         match (expected_rows_per_frame, bad_frame) {
             (None, _) => "skipped (no expected N)".to_string(),
             (Some(n), None) => format!("every frame has {} rows", n),
             (Some(n), Some((t, c))) => format!("frame t={:.3} has {} rows (expected {})", t, c, n),
         });

    // Check 5: number of frames matches expected_frames (if provided)
    if let Some(ef) = expected_frames {
        let tol = (ef as f64 * 0.02).max(2.0); // 2% tolerance, min 2
        let diff = (timestamps.len() as f64 - ef as f64).abs();
        push("frame_count",
             diff <= tol,
             format!("got {}, expected {} (tol ±{:.0})", timestamps.len(), ef, tol));
    } else {
        push("frame_count", true, format!("{} frames (no expectation)", timestamps.len()));
    }

    // Check 6: frame interval approximately constant
    if timestamps.len() >= 3 {
        let intervals: Vec<f64> = (1..timestamps.len()).map(|i| timestamps[i] - timestamps[i-1]).collect();
        let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let max_dev = intervals.iter().map(|&x| (x - mean).abs()).fold(0.0f64, f64::max);
        let rel_dev = if mean > 0.0 { max_dev / mean } else { 1.0 };
        push("frame_interval_uniform",
             rel_dev < 0.10,
             format!("mean Δt = {:.3}, max deviation {:.1}%", mean, 100.0 * rel_dev));
    }

    // Check 7: t_start / t_end match expectations
    if let Some(ts_exp) = expected_t_start {
        if let Some(&ts_got) = timestamps.first() {
            let tol = (ts_exp.abs() * 0.01).max(1.0);
            push("t_start",
                 (ts_got - ts_exp).abs() <= tol,
                 format!("got {:.3}, expected {:.3} (tol ±{:.1})", ts_got, ts_exp, tol));
        }
    }
    if let Some(te_exp) = expected_t_end {
        if let Some(&te_got) = timestamps.last() {
            let tol = (te_exp.abs() * 0.01).max(1.0);
            push("t_end",
                 (te_got - te_exp).abs() <= tol,
                 format!("got {:.3}, expected {:.3} (tol ±{:.1})", te_got, te_exp, tol));
        }
    }

    // Check 8: checkpoint.bin consistency (optional)
    let ckpt_path = dir.join("checkpoint.bin");
    if ckpt_path.exists() {
        match analysis::checkpoint::load_checkpoint_header_only(&ckpt_path) {
            Ok((ckpt_t, ckpt_n, ckpt_ver)) => {
                let mut ok = true;
                let mut msgs: Vec<String> = Vec::new();
                msgs.push(format!("v{} step_t={:.3} N={}", ckpt_ver, ckpt_t, ckpt_n));
                if let Some(&last_t) = timestamps.last() {
                    let tol = (ckpt_t.abs() * 0.01).max(1.0);
                    if (ckpt_t - last_t).abs() > tol {
                        ok = false;
                        msgs.push(format!("checkpoint t={:.3} disagrees with last trajectory t={:.3}", ckpt_t, last_t));
                    }
                }
                if let Some(n) = expected_rows_per_frame {
                    if ckpt_n as usize != n {
                        ok = false;
                        msgs.push(format!("checkpoint N={} disagrees with expected {}", ckpt_n, n));
                    }
                }
                push("checkpoint_consistency", ok, msgs.join("; "));
            }
            Err(e) => {
                push("checkpoint_consistency", false, format!("failed to parse checkpoint: {}", e));
            }
        }
    } else {
        push("checkpoint_consistency", true, "no checkpoint.bin (skipped)".to_string());
    }

    // Report
    let all_pass = results.iter().all(|r| r.passed);
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    println!("=== cell_analyze check: {} ===", dir.display());
    for r in &results {
        let mark = if r.passed { "PASS" } else { "FAIL" };
        println!("  [{}] {:<28} {}", mark, r.name, r.detail);
    }
    println!("--- rows parsed: {} ---", row_count);
    println!("=== {} / {} checks passed ===", passed, total);

    if let Some(p) = json_out {
        let json = serde_json::to_string_pretty(&serde_json::json!({
            "dir": dir.display().to_string(),
            "all_pass": all_pass,
            "passed": passed,
            "total": total,
            "checks": results.iter().map(|r| serde_json::json!({
                "name": r.name, "passed": r.passed, "detail": r.detail
            })).collect::<Vec<_>>(),
        }))?;
        std::fs::write(p, json)?;
    }

    Ok(if all_pass { 0 } else { 1 })
}

fn analyze_single_run(
    dir: &PathBuf,
    observables: &[String],
    tau: f64,
    cell_radius: f64,
    fit_frac: f64,
    sq_bins: usize,
    sq_frames: usize,
    subsample: usize,
) -> Result<RunResult> {
    let t0 = Instant::now();
    let traj_path = dir.join("trajectory.txt");
    if !traj_path.exists() {
        anyhow::bail!("No trajectory.txt found in {}", dir.display());
    }

    let traj = load_trajectory_subsample(&traj_path, subsample)?;
    let pos = unwrap_trajectory(&traj);

    let has = |name: &str| observables.iter().any(|s| s == name);

    let cell_spacing = (pos.lx * pos.ly / pos.n_cells as f64).sqrt();
    let cage_radius = cell_spacing * 0.3;

    // Compute requested observables
    let msd = if has("msd") || has("diffusion") || has("log_slope") || has("cage") {
        eprintln!("  Computing MSD...");
        Some(compute_msd(&pos))
    } else {
        None
    };

    let diffusion = if has("diffusion") {
        msd.as_ref().map(|m| compute_diffusion(m, fit_frac))
    } else {
        None
    };

    let log_slope = if has("log_slope") {
        msd.as_ref().map(|m| msd_log_slope(m))
    } else {
        None
    };

    let cage = if has("cage") {
        msd.as_ref().map(|m| cage_length(m, tau))
    } else {
        None
    };

    let alpha2 = if has("alpha2") {
        eprintln!("  Computing α₂...");
        Some(non_gaussian_parameter(&pos))
    } else {
        None
    };

    let overlap = if has("overlap") {
        eprintln!("  Computing Q(t), χ₄...");
        Some(overlap_and_chi4(&pos, cage_radius))
    } else {
        None
    };

    let structure = if has("structure") || has("scattering") {
        eprintln!("  Computing S(q)...");
        Some(structure_factor(&pos, sq_bins, sq_frames))
    } else {
        None
    };

    let scattering = if has("scattering") {
        let q_star = structure
            .as_ref()
            .map_or(0.1, |s| s.q_star);
        eprintln!("  Computing F_s(q*={:.4}, t)...", q_star);
        Some(self_intermediate_scattering(&pos, q_star))
    } else {
        None
    };

    let van_hove_result = if has("van_hove") {
        eprintln!("  Computing van Hove G_s...");
        Some(van_hove(&pos, tau, 200))
    } else {
        None
    };

    let pcd = if has("per_cell_diffusion") {
        eprintln!("  Computing per-cell D...");
        Some(per_cell_diffusion(&pos, fit_frac, tau))
    } else {
        None
    };

    let displacement = if has("displacement") {
        Some(compute_displacement(&pos, cell_radius))
    } else {
        None
    };

    let va_corr = if has("va_mobility_correlation") {
        eprintln!("  Computing v_A-mobility correlation...");
        Some(va_mobility_correlation(&pos))
    } else {
        None
    };

    let spatial_corr = if has("spatial_correlation") {
        eprintln!("  Computing spatial correlation C(r)...");
        Some(spatial_correlation(&pos, 40))
    } else {
        None
    };

    let shape_idx = if has("shape_index") {
        eprintln!("  Computing shape index p_eff from L_n...");
        Some(shape_index(&traj))
    } else {
        None
    };
    let vel_autocorr = if has("velocity_autocorrelation") {
        eprintln!("  Computing velocity autocorrelation C_v(τ)...");
        Some(velocity_autocorrelation(&pos))
    } else {
        None
    };

    let bursts = if has("burst_detection") {
        eprintln!("  Detecting speed bursts (3σ threshold)...");
        Some(detect_bursts(&pos, &traj, 3.0, 1))
    } else {
        None
    };

    let vel_dist = if has("velocity_distribution") {
        eprintln!("  Computing velocity distribution P(v_x)...");
        Some(velocity_distribution(&pos, 100))
    } else {
        None
    };

    let pol_tau = if has("polarity_tau") {
        eprintln!("  Estimating τ from polarity autocorrelation...");
        Some(polarity_tau(&traj))
    } else {
        None
    };

    let hex_order = if has("hexatic_order") {
        eprintln!("  Computing hexatic order ψ₆ and g₆(r)...");
        Some(compute_hexatic_order(&pos, cell_radius))
    } else {
        None
    };

    let vor_shape = if has("voronoi_shape") {
        eprintln!("  Computing Voronoi shape index q = P/√A...");
        Some(compute_voronoi_shape(&pos, cell_radius))
    } else {
        None
    };

    let kin_energy = if has("kinetic_energy") {
        eprintln!("  Computing kinetic energy time series...");
        Some(compute_kinetic_energy(&pos))
    } else {
        None
    };

    let se = match (&diffusion, &overlap) {
        (Some(d), Some(o)) => {
            let val = stokes_einstein(d.d_eff, o.tau_alpha);
            if val.is_finite() {
                Some(val)
            } else {
                None
            }
        }
        _ => None,
    };

    let extra: BTreeMap<String, String> = traj
        .params
        .extra
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let elapsed = t0.elapsed();
    eprintln!(
        "  Done: {} frames, {} cells, {:.1}s",
        pos.n_times,
        pos.n_cells,
        elapsed.as_secs_f64()
    );

    Ok(RunResult {
        path: dir.display().to_string(),
        params: RunParams {
            v_a: traj.params.v_a,
            n_cells: traj.params.n_cells,
            lx: traj.params.lx,
            ly: traj.params.ly,
            confluence: traj.params.n_cells as f64 * std::f64::consts::PI * cell_radius * cell_radius / (traj.params.lx * traj.params.ly),
            subdomain_padding: None,
            bbox_mean: None,
            extra,
        },
        msd: if has("msd") { msd } else { None },
        diffusion,
        log_slope,
        cage,
        alpha2,
        overlap,
        structure: if has("structure") { structure } else { None },
        scattering,
        van_hove: van_hove_result,
        per_cell_diffusion: pcd,
        displacement,
        stokes_einstein: se,
        va_mobility_correlation: va_corr,
        spatial_correlation: spatial_corr,
        shape_index: shape_idx,
        velocity_autocorrelation: vel_autocorr,
        burst_detection: bursts,
        velocity_distribution: vel_dist,
        polarity_tau: pol_tau,
        hexatic_order: hex_order,
        voronoi_shape: vor_shape,
        kinetic_energy: kin_energy,
    })
}

// ============================================================================
// SVG plot generation for single-run observables
// ============================================================================

fn plot_run_result(result: &RunResult, plot_dir: &Path, cell_radius: f64) -> Result<()> {
    use plotters::prelude::*;

    let n_cells = result.params.n_cells;
    let label = format!("N={}", n_cells);

    // --- Hexatic order plots ---
    if let Some(ref h) = result.hexatic_order {
        // 1. g₆(r) line plot
        if !h.g6_r.is_empty() {
            let r_norm: Vec<f64> = h.g6_r.iter().map(|r| r / (2.0 * cell_radius)).collect();
            let out_path = plot_dir.join("g6_r.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;

            let x_max = r_norm.last().copied().unwrap_or(5.0) + 0.1;
            let y_min = h.g6_values.iter().copied().fold(f64::INFINITY, f64::min).min(-0.05) - 0.02;
            let y_max = h.g6_values.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.05) + 0.02;

            let mut chart = ChartBuilder::on(&root)
                .caption(format!("g₆(r) — {} (⟨ψ₆⟩={:.3})", label, h.psi6_mean), ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(65)
                .build_cartesian_2d(0.0..x_max, y_min..y_max)?;
            chart.configure_mesh()
                .x_desc("r / (2R)").y_desc("g₆(r)")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
            chart.draw_series(LineSeries::new(vec![(0.0, 0.0), (x_max, 0.0)],
                ShapeStyle::from(&RGBAColor(150, 150, 150, 0.5)).stroke_width(1)))?;
            chart.draw_series(LineSeries::new(
                r_norm.iter().zip(h.g6_values.iter()).map(|(&x, &y)| (x, y)),
                ShapeStyle::from(&BLUE).stroke_width(2)))?.label(&label);
            chart.draw_series(r_norm.iter().zip(h.g6_values.iter()).map(|(&x, &y)| Circle::new((x, y), 3, BLUE.filled())))?;
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }

        // 2. ψ₆ histogram
        if !h.psi6_per_cell.is_empty() {
            let out_path = plot_dir.join("psi6_histogram.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;
            let n_bins = 20usize;
            let mut counts = vec![0u32; n_bins];
            for &v in &h.psi6_per_cell {
                let b = ((v * n_bins as f64) as usize).min(n_bins - 1);
                counts[b] += 1;
            }
            let max_count = *counts.iter().max().unwrap_or(&1);
            let mut chart = ChartBuilder::on(&root)
                .caption(format!("|ψ₆| distribution — {} (⟨ψ₆⟩={:.3})", label, h.psi6_mean), ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(55)
                .build_cartesian_2d(0.0..1.0f64, 0u32..(max_count + 1))?;
            chart.configure_mesh()
                .x_desc("|ψ₆|").y_desc("Count")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
            let bin_w = 1.0 / n_bins as f64;
            chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
                let x0 = i as f64 * bin_w;
                Rectangle::new([(x0, 0), (x0 + bin_w * 0.9, c)], BLUE.mix(0.7).filled())
            }))?;
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- Voronoi shape index plot ---
    if let Some(ref v) = result.voronoi_shape {
        if !v.q_per_cell.is_empty() {
            let out_path = plot_dir.join("voronoi_q_histogram.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;
            let q_min = v.q_per_cell.iter().copied().fold(f64::INFINITY, f64::min);
            let q_max = v.q_per_cell.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range_pad = (q_max - q_min).max(0.1) * 0.15;
            let hist_min = (q_min - range_pad).max(3.0);
            let hist_max = q_max + range_pad;
            let n_bins = 20usize;
            let bw = (hist_max - hist_min) / n_bins as f64;
            let mut counts = vec![0u32; n_bins];
            for &q in &v.q_per_cell {
                if q > 0.0 {
                    let b = ((q - hist_min) / bw) as usize;
                    if b < n_bins { counts[b] += 1; }
                }
            }
            let max_count = *counts.iter().max().unwrap_or(&1);
            let mut chart = ChartBuilder::on(&root)
                .caption(format!("Voronoi q = P/√A — {} (⟨q⟩={:.2})", label, v.q_mean), ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(55)
                .build_cartesian_2d(hist_min..hist_max, 0u32..(max_count + 1))?;
            chart.configure_mesh()
                .x_desc("q = P/√A").y_desc("Count")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
            chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
                let x0 = hist_min + i as f64 * bw;
                Rectangle::new([(x0, 0), (x0 + bw * 0.9, c)], BLUE.mix(0.7).filled())
            }))?;
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- MSD/Δt plot (Palmieri Fig 5 convention: linear axes, Δt up to 8τ) ---
    if let Some(ref msd) = result.msd {
        if !msd.lag_times.is_empty() {
            let out_path = plot_dir.join("msd.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;

            // Palmieri Fig 5 caps the x-axis at Δt = 8τ.
            let tau = result.params.extra.get("tau")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(10000.0);
            let x_cap = 8.0 * tau;

            let pts: Vec<(f64, f64)> = msd.lag_times.iter().zip(msd.values.iter())
                .filter(|(&t, _)| t > 0.0 && t <= x_cap)
                .map(|(&t, &v)| (t / tau, v / t))
                .collect();
            if !pts.is_empty() {
                let x_min = 0.0;
                let x_max = 8.0;
                let y_max = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) * 1.1 + 1e-12;

                let mut chart = ChartBuilder::on(&root)
                    .caption(format!("MSD/Δt → 4D_eff — {}", label), ("sans-serif", 22).into_font())
                    .margin(15)
                    .x_label_area_size(45)
                    .y_label_area_size(65)
                    .build_cartesian_2d(x_min..x_max, 0.0..y_max)?;

                chart.configure_mesh()
                    .x_desc("Δt / τ")
                    .y_desc("MSD/Δt")
                    .x_label_style(("sans-serif", 16))
                    .y_label_style(("sans-serif", 16))
                    .axis_desc_style(("sans-serif", 18))
                    .light_line_style(TRANSPARENT)
                    .draw()?;

                chart.draw_series(LineSeries::new(
                    pts.iter().copied(),
                    ShapeStyle::from(&BLUE).stroke_width(2),
                ))?;

                root.present()?;
                eprintln!("  Plot: {}", out_path.display());
            }
        }
    }

    // --- Cell-0 perimeter L_n(t) plot ---
    // Tagged-cell elasticity: raw normalized perimeter L_n vs time, overlaid
    // with population mean. shape_index stores p_eff = L_n × 2√π, so divide
    // back by the same factor to recover L_n.
    if let Some(ref si) = result.shape_index {
        if !si.times.is_empty() && !si.cell0_p_vs_time.is_empty() {
            let factor = 2.0 * std::f64::consts::PI.sqrt();
            let l_n_cell0: Vec<f64> = si.cell0_p_vs_time.iter().map(|&p| p / factor).collect();
            let l_n_pop: Vec<f64>   = si.p_vs_time.iter().map(|&p| p / factor).collect();

            let out_path = plot_dir.join("cell0_perimeter.svg");
            let root = SVGBackend::new(&out_path, (900, 500)).into_drawing_area();
            root.fill(&WHITE)?;

            let x_min = *si.times.first().unwrap();
            let x_max = *si.times.last().unwrap();
            let mut y_min = l_n_cell0.iter().chain(l_n_pop.iter())
                .copied().fold(f64::INFINITY, f64::min);
            let mut y_max = l_n_cell0.iter().chain(l_n_pop.iter())
                .copied().fold(f64::NEG_INFINITY, f64::max);
            let pad = ((y_max - y_min) * 0.05).max(1e-3);
            y_min -= pad; y_max += pad;

            let cell0_mean = l_n_cell0.iter().sum::<f64>() / l_n_cell0.len() as f64;
            let cell0_var  = l_n_cell0.iter().map(|v| (v - cell0_mean).powi(2)).sum::<f64>()
                / l_n_cell0.len() as f64;
            let cell0_std  = cell0_var.sqrt();

            let mut chart = ChartBuilder::on(&root)
                .caption(
                    format!("Cell-0 perimeter L_n(t) — {} (⟨L_n⟩={:.3}, σ={:.3})",
                            label, cell0_mean, cell0_std),
                    ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(65)
                .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
            chart.configure_mesh()
                .x_desc("t").y_desc("L_n  (normalized perimeter)")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;

            // Population mean (background, grey)
            chart.draw_series(LineSeries::new(
                si.times.iter().zip(l_n_pop.iter()).map(|(&x, &y)| (x, y)),
                ShapeStyle::from(&RGBAColor(120, 120, 120, 0.7)).stroke_width(1)))?
                .label("⟨L_n⟩ population")
                .legend(|(x, y)| PathElement::new(
                    vec![(x, y), (x + 18, y)],
                    ShapeStyle::from(&RGBAColor(120, 120, 120, 0.9)).stroke_width(2)));

            // Cell-0 trace (foreground, red)
            chart.draw_series(LineSeries::new(
                si.times.iter().zip(l_n_cell0.iter()).map(|(&x, &y)| (x, y)),
                ShapeStyle::from(&RED).stroke_width(2)))?
                .label("L_n cell 0 (tagged)")
                .legend(|(x, y)| PathElement::new(
                    vec![(x, y), (x + 18, y)],
                    ShapeStyle::from(&RED).stroke_width(2)));

            chart.configure_series_labels()
                .background_style(WHITE.mix(0.85))
                .border_style(BLACK)
                .label_font(("sans-serif", 14))
                .position(plotters::chart::SeriesLabelPosition::UpperRight)
                .draw()?;

            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- Kinetic energy plot ---
    if let Some(ref ke) = result.kinetic_energy {
        if !ke.times.is_empty() {
            let out_path = plot_dir.join("kinetic_energy.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;
            let x_min = *ke.times.first().unwrap();
            let x_max = *ke.times.last().unwrap();
            let y_min = ke.ke_per_cell.iter().copied().fold(f64::INFINITY, f64::min) * 0.9;
            let y_max = ke.ke_per_cell.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.1;
            let mut chart = ChartBuilder::on(&root)
                .caption(format!("KE per cell — {} (⟨KE⟩={:.2e})", label, ke.ke_mean), ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(65)
                .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
            chart.configure_mesh()
                .x_desc("t").y_desc("½⟨v²⟩")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
            chart.draw_series(LineSeries::new(
                ke.times.iter().zip(ke.ke_per_cell.iter()).map(|(&x, &y)| (x, y)),
                ShapeStyle::from(&BLUE).stroke_width(2)))?;
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- Velocity autocorrelation C_v(τ) ---
    if let Some(ref va) = result.velocity_autocorrelation {
        if !va.lag_times.is_empty() {
            let out_path = plot_dir.join("velocity_autocorrelation.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;
            let x_min = *va.lag_times.first().unwrap();
            let x_max = *va.lag_times.last().unwrap();
            let y_min = va.cv.iter().copied().fold(f64::INFINITY, f64::min).min(-0.05);
            let y_max = va.cv.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.05);
            let mut chart = ChartBuilder::on(&root)
                .caption(format!("Velocity autocorrelation — {} (τ_c={:.3e}, β={:.2})", label, va.tau_c, va.beta), ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(65)
                .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
            chart.configure_mesh()
                .x_desc("τ").y_desc("C_v(τ)")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
            chart.draw_series(LineSeries::new(vec![(x_min, 0.0), (x_max, 0.0)],
                ShapeStyle::from(&RGBAColor(150, 150, 150, 0.5)).stroke_width(1)))?;
            chart.draw_series(LineSeries::new(
                va.lag_times.iter().zip(va.cv.iter()).map(|(&x, &y)| (x, y)),
                ShapeStyle::from(&BLUE).stroke_width(2)))?;
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- Velocity distribution P(v_x): cell 0 vs population ---
    if let Some(ref vd) = result.velocity_distribution {
        if vd.bin_edges.len() >= 2 && !vd.pop_hist.is_empty() {
            let out_path = plot_dir.join("velocity_distribution.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;
            let centers: Vec<f64> = vd.bin_edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
            let x_min = *vd.bin_edges.first().unwrap();
            let x_max = *vd.bin_edges.last().unwrap();
            let y_max = vd.pop_hist.iter().chain(vd.cell0_hist.iter())
                .copied().fold(0.0_f64, f64::max) * 1.1 + 1e-12;
            let mut chart = ChartBuilder::on(&root)
                .caption(format!("P(v_x) — {} (κ_pop={:.2}, κ_cell0={:.2})", label, vd.pop_kurtosis, vd.cell0_kurtosis), ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(65)
                .build_cartesian_2d(x_min..x_max, 0.0..y_max)?;
            chart.configure_mesh()
                .x_desc("v_x").y_desc("P(v_x)")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
            chart.draw_series(LineSeries::new(
                centers.iter().zip(vd.pop_hist.iter()).map(|(&x, &y)| (x, y)),
                ShapeStyle::from(&RGBAColor(120, 120, 120, 0.9)).stroke_width(2)))?
                .label("population")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)],
                    ShapeStyle::from(&RGBAColor(120, 120, 120, 0.9)).stroke_width(2)));
            if !vd.cell0_hist.is_empty() {
                chart.draw_series(LineSeries::new(
                    centers.iter().zip(vd.cell0_hist.iter()).map(|(&x, &y)| (x, y)),
                    ShapeStyle::from(&RED).stroke_width(2)))?
                    .label("cell 0")
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)],
                        ShapeStyle::from(&RED).stroke_width(2)));
            }
            chart.configure_series_labels()
                .background_style(WHITE.mix(0.85)).border_style(BLACK)
                .label_font(("sans-serif", 14))
                .position(plotters::chart::SeriesLabelPosition::UpperRight).draw()?;
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- G(v_i) = -sqrt(|ln CCDF(|v_i|)|) (Palmieri Fig 4 convention) ---
    // Reuses analysis::panels::draw_gvi_panel so single-run and study/compare
    // share exactly one renderer. Pass any number of (label, vx, vy) series.
    if let Some(ref vd) = result.velocity_distribution {
        if !vd.cell0_vx.is_empty() {
            use analysis::panels::{draw_gvi_panel, GviSeries, GviPanelOpts, GviMarker};
            let out_path = plot_dir.join("velocity_gvi.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;

            // Single-run G(v_i): cell 0 only (the soft cell). Population
            // is omitted -- in homogeneous-v_A configs it tracks cell 0
            // anyway; the relevant comparison is data vs Gaussian + Eq.5.
            let series = vec![GviSeries {
                label: "cell 0".to_string(),
                vx: &vd.cell0_vx, vy: &vd.cell0_vy,
                color: RGBAColor(220, 50, 50, 0.95),
                marker: GviMarker::Triangle,
            }];

            let opts = GviPanelOpts {
                title: format!("G(v_i) — {}", label),
                // Naive moment-based σ from cell 0 (matches second
                // moment of the data). This is the *correct* Gaussian
                // reference: any visible deviation is the non-Gaussian
                // (active / burst) signal — fitting σ to the curve shape
                // is misleading because the data is genuinely non-Gaussian.
                gaussian_ref_sigma: None,
                // Fit Palmieri Eq. 5 (Gaussian noise + arcsine bursts)
                // to the cell 0 series.
                palmieri_fit_index: Some(0),
                v_a: 0.01,
                ..Default::default()
            };
            draw_gvi_panel(&root, &series, &opts)?;
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- Per-cell diffusion histogram ---
    if let Some(ref pcd) = result.per_cell_diffusion {
        if !pcd.d_values.is_empty() {
            let out_path = plot_dir.join("per_cell_diffusion.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;
            let d_min = pcd.d_values.iter().copied().fold(f64::INFINITY, f64::min);
            let d_max = pcd.d_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let pad = (d_max - d_min).max(1e-12) * 0.1;
            let x_lo = (d_min - pad).max(0.0);
            let x_hi = d_max + pad;
            let n_bins = 20usize;
            let bw = (x_hi - x_lo) / n_bins as f64;
            let mut counts = vec![0u32; n_bins];
            for &d in &pcd.d_values {
                if bw > 0.0 {
                    let b = (((d - x_lo) / bw) as usize).min(n_bins - 1);
                    counts[b] += 1;
                }
            }
            let max_count = *counts.iter().max().unwrap_or(&1);
            let mut chart = ChartBuilder::on(&root)
                .caption(format!("Per-cell D — {} (⟨D⟩={:.3e}, CV={:.2})", label, pcd.d_mean, pcd.cv), ("sans-serif", 20).into_font())
                .margin(15).x_label_area_size(45).y_label_area_size(55)
                .build_cartesian_2d(x_lo..x_hi, 0u32..(max_count + 1))?;
            chart.configure_mesh()
                .x_desc("D (per cell)").y_desc("Count")
                .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
            chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
                let x0 = x_lo + i as f64 * bw;
                Rectangle::new([(x0, 0), (x0 + bw * 0.9, c)], BLUE.mix(0.7).filled())
            }))?;
            // Mark cell 0's D value with a red vertical line
            if let Some(idx) = pcd.cell_ids.iter().position(|&id| id == 0) {
                let d0 = pcd.d_values[idx];
                chart.draw_series(LineSeries::new(
                    vec![(d0, 0u32), (d0, max_count + 1)],
                    ShapeStyle::from(&RED).stroke_width(2)))?
                    .label(format!("cell 0: D={:.3e}", d0))
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)],
                        ShapeStyle::from(&RED).stroke_width(2)));
                chart.configure_series_labels()
                    .background_style(WHITE.mix(0.85)).border_style(BLACK)
                    .label_font(("sans-serif", 14))
                    .position(plotters::chart::SeriesLabelPosition::UpperRight).draw()?;
            }
            root.present()?;
            eprintln!("  Plot: {}", out_path.display());
        }
    }

    // --- Non-Gaussian parameter α₂(t) ---
    if let Some(ref a2) = result.alpha2 {
        if !a2.lag_times.is_empty() {
            let out_path = plot_dir.join("alpha2.svg");
            let root = SVGBackend::new(&out_path, (720, 480)).into_drawing_area();
            root.fill(&WHITE)?;
            let t_pos: Vec<f64> = a2.lag_times.iter().filter(|&&t| t > 0.0).copied().collect();
            let v_pos: Vec<f64> = a2.lag_times.iter().zip(a2.values.iter())
                .filter(|(&t, _)| t > 0.0).map(|(_, &v)| v).collect();
            if !t_pos.is_empty() {
                let t_log: Vec<f64> = t_pos.iter().map(|v| v.log10()).collect();
                let x_min = t_log.first().copied().unwrap();
                let x_max = t_log.last().copied().unwrap();
                let y_min = v_pos.iter().copied().fold(f64::INFINITY, f64::min).min(-0.05);
                let y_max = v_pos.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.05);
                let mut chart = ChartBuilder::on(&root)
                    .caption(format!("Non-Gaussian α₂(Δt) — {}", label), ("sans-serif", 20).into_font())
                    .margin(15).x_label_area_size(45).y_label_area_size(65)
                    .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
                chart.configure_mesh()
                    .x_desc("log₁₀(Δt)").y_desc("α₂")
                    .x_label_style(("sans-serif", 16)).y_label_style(("sans-serif", 16))
                    .axis_desc_style(("sans-serif", 18)).light_line_style(TRANSPARENT).draw()?;
                chart.draw_series(LineSeries::new(vec![(x_min, 0.0), (x_max, 0.0)],
                    ShapeStyle::from(&RGBAColor(150, 150, 150, 0.5)).stroke_width(1)))?;
                chart.draw_series(LineSeries::new(
                    t_log.iter().zip(v_pos.iter()).map(|(&x, &y)| (x, y)),
                    ShapeStyle::from(&BLUE).stroke_width(2)))?;
                root.present()?;
                eprintln!("  Plot: {}", out_path.display());
            }
        }
    }

    Ok(())
}

/// RdYlBu_r colormap approximation: 0=blue, 0.5=yellow, 1=dark red.
// ============================================================================
// Cell label rendering for snapshot --label-cells
// ============================================================================

/// Simple 4x6 bitmap patterns for digits 0-9 (each is 6 rows of 4-bit wide)
fn digit_bitmap(d: u8) -> [u8; 6] {
    match d {
        0 => [0b0110, 0b1001, 0b1001, 0b1001, 0b1001, 0b0110],
        1 => [0b0010, 0b0110, 0b0010, 0b0010, 0b0010, 0b0111],
        2 => [0b0110, 0b1001, 0b0010, 0b0100, 0b1000, 0b1111],
        3 => [0b0110, 0b1001, 0b0010, 0b0001, 0b1001, 0b0110],
        4 => [0b1001, 0b1001, 0b1111, 0b0001, 0b0001, 0b0001],
        5 => [0b1111, 0b1000, 0b1110, 0b0001, 0b0001, 0b1110],
        6 => [0b0110, 0b1000, 0b1110, 0b1001, 0b1001, 0b0110],
        7 => [0b1111, 0b0001, 0b0010, 0b0100, 0b0100, 0b0100],
        8 => [0b0110, 0b1001, 0b0110, 0b1001, 0b1001, 0b0110],
        9 => [0b0110, 0b1001, 0b0111, 0b0001, 0b0001, 0b0110],
        _ => [0b0000, 0b0000, 0b0000, 0b0000, 0b0000, 0b0000],
    }
}

/// Draw a cell ID label at pixel position (cx, cy) on an RGB image buffer.
/// `highlight` = true for cell 0 (green label), false = white label.
fn draw_label(img: &mut [u8], w: usize, h: usize, cx: i32, cy: i32, text: &str, highlight: bool) {
    draw_label_with_spatial(img, w, h, cx, cy, text, "", "", highlight);
}

/// Draw cell ID + spatial index label. Spatial index drawn smaller below the ID.
/// `above_text` is drawn in the 5x7 font above the ID (used for "γ=0.35"
/// annotations on cells whose stiffness deviates from the population mode).
fn draw_label_with_spatial(img: &mut [u8], w: usize, h: usize, cx: i32, cy: i32,
                            id_text: &str, spatial_text: &str, above_text: &str,
                            highlight: bool) {
    let scale = 2i32;
    let small_scale = 1i32;
    let char_w = 4 * scale + scale;
    let char_h = 6 * scale;
    let n_chars = id_text.len() as i32;
    let total_w = n_chars * char_w - scale;

    let has_spatial = !spatial_text.is_empty();
    let small_char_w = 4 * small_scale + small_scale;
    let small_char_h = 6 * small_scale;
    let small_n = spatial_text.len() as i32;
    let small_total_w = if has_spatial { small_n * small_char_w - small_scale } else { 0 };
    let gap = if has_spatial { 2 } else { 0 };

    // 5x7-font sizing for the above-text band (e.g. "γ=0.35"). Glyph width is
    // 5*scale, kerning is 1*scale; chars().count() avoids byte-counting bugs
    // for multibyte γ.
    let has_above = !above_text.is_empty();
    let above_scale = 1i32;
    let above_char_total = 6 * above_scale; // 5 px glyph + 1 px kern
    let above_n = above_text.chars().count() as i32;
    let above_total_w = if has_above { (above_n * above_char_total).saturating_sub(above_scale) } else { 0 };
    let above_h = if has_above { 7 * above_scale } else { 0 };
    let above_gap = if has_above { 2 } else { 0 };

    let pad = scale;
    let label_w = total_w.max(small_total_w).max(above_total_w);
    let label_h = above_h + above_gap + char_h + if has_spatial { gap + small_char_h } else { 0 };

    // Clamp label position so it stays fully inside the image
    let half_w = label_w / 2 + pad;
    let half_h = label_h / 2 + pad;
    let cx = cx.max(half_w).min(w as i32 - 1 - half_w);
    let cy = cy.max(half_h).min(h as i32 - 1 - half_h);

    // Background rectangle
    let bg_x0 = cx - label_w / 2 - pad;
    let bg_y0 = cy - label_h / 2 - pad;
    let bg_x1 = cx + label_w / 2 + pad;
    let bg_y1 = cy + label_h / 2 + pad;

    let (bg_r, bg_g, bg_b) = if highlight { (0u8, 80, 0) } else { (0, 0, 0) };
    let bg_a = 180u8; // slightly transparent effect via blending

    for py in bg_y0..=bg_y1 {
        for px in bg_x0..=bg_x1 {
            if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                let idx = (py as usize * w + px as usize) * 3;
                // Alpha blend
                let a = bg_a as u16;
                img[idx]     = ((img[idx] as u16 * (255 - a) + bg_r as u16 * a) / 255) as u8;
                img[idx + 1] = ((img[idx + 1] as u16 * (255 - a) + bg_g as u16 * a) / 255) as u8;
                img[idx + 2] = ((img[idx + 2] as u16 * (255 - a) + bg_b as u16 * a) / 255) as u8;
            }
        }
    }

    let label_top = cy - label_h / 2;

    // Draw above-text (γ value) using 5x7 font. Use a warm yellow tint so it
    // visually distinguishes from the ID (white/green).
    if has_above {
        let above_x = cx - above_total_w / 2;
        let above_rgb = if highlight { [255, 230, 120] } else { [255, 200, 80] };
        draw_text_5x7(img, w, h, above_x, label_top, above_text, above_scale, above_rgb);
    }

    // Draw cell ID (main text, scale 2)
    let (fg_r, fg_g, fg_b) = if highlight { (100, 255, 100) } else { (255, 255, 255) };
    let start_x = cx - total_w / 2;
    let start_y = label_top + above_h + above_gap;

    draw_text_bitmap(img, w, h, start_x, start_y, id_text, scale, fg_r, fg_g, fg_b);

    // Draw spatial index below (scale 1, dimmer)
    if has_spatial {
        let (sr, sg, sb) = if highlight { (80, 200, 80) } else { (180, 180, 180) };
        let small_start_x = cx - small_total_w / 2;
        let small_start_y = start_y + char_h + gap;
        draw_text_bitmap(img, w, h, small_start_x, small_start_y, spatial_text, small_scale, sr, sg, sb);
    }
}

/// Render bitmap text at (x, y) with given scale and color.
fn draw_text_bitmap(img: &mut [u8], w: usize, h: usize, x: i32, y: i32,
                    text: &str, scale: i32, r: u8, g: u8, b: u8) {
    let char_w = 4 * scale + scale;
    for (ci, ch) in text.chars().enumerate() {
        if let Some(d) = ch.to_digit(10) {
            let bitmap = digit_bitmap(d as u8);
            let ox = x + ci as i32 * char_w;
            for row in 0..6 {
                for col in 0..4 {
                    if bitmap[row] & (0b1000 >> col) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                let px = ox + col as i32 * scale + sx;
                                let py = y + row as i32 * scale + sy;
                                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                                    let idx = (py as usize * w + px as usize) * 3;
                                    img[idx] = r;
                                    img[idx + 1] = g;
                                    img[idx + 2] = b;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Read cell centroids from trajectory.txt at the closest time to `target_time`.
fn read_centroids_at_time(traj_path: &std::path::Path, target_time: f64) -> Vec<(u32, f64, f64, bool)> {
    use std::io::BufRead;
    let file = match std::fs::File::open(traj_path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = std::io::BufReader::new(file);

    let mut best_time = -1.0f64;
    let mut best_cells: Vec<(u32, f64, f64, bool)> = Vec::new();
    let mut current_cells: Vec<(u32, f64, f64, bool)> = Vec::new();
    let mut current_time = -1.0f64;

    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        if line.starts_with('#') || line.is_empty() { continue; }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 { continue; }
        let t: f64 = match parts[0].parse() { Ok(v) => v, Err(_) => continue };
        let cid: u32 = match parts[1].parse() { Ok(v) => v, Err(_) => continue };
        let x: f64 = match parts[2].parse() { Ok(v) => v, Err(_) => continue };
        let y: f64 = match parts[3].parse() { Ok(v) => v, Err(_) => continue };

        if (t - current_time).abs() > 0.01 {
            // New time point — check if previous was better
            if !current_cells.is_empty() {
                if best_time < 0.0 || (current_time - target_time).abs() < (best_time - target_time).abs() {
                    best_time = current_time;
                    best_cells = current_cells.clone();
                }
                // If we've passed the target, stop
                if current_time > target_time + 1000.0 { break; }
            }
            current_cells.clear();
            current_time = t;
        }
        if x.is_finite() && y.is_finite() {
            current_cells.push((cid, x, y, false));  // no gamma info from trajectory
        }
    }
    // Check last batch
    if !current_cells.is_empty() {
        if best_time < 0.0 || (current_time - target_time).abs() < (best_time - target_time).abs() {
            best_cells = current_cells;
        }
    }
    best_cells
}

/// Load centroids for a VTK frame, enriching with soft-cell info from checkpoint if available.
fn load_centroids_for_vtk(vtk_path: &std::path::Path, time: f64) -> Vec<(u32, f64, f64, bool)> {
    let parent = match vtk_path.parent() {
        Some(p) => p,
        None => return Vec::new(),
    };
    let traj_path = parent.join("trajectory.txt");
    if !traj_path.exists() { return Vec::new(); }

    let mut centroids = read_centroids_at_time(&traj_path, time);
    if centroids.is_empty() { return centroids; }

    // Try to load checkpoint.bin in the same directory for gamma-based soft detection
    let ckpt_path = parent.join("checkpoint.bin");
    if ckpt_path.exists() {
        if let Ok(ckpt) = analysis::checkpoint::load_checkpoint(&ckpt_path) {
            let gammas = &ckpt.per_cell_gamma;
            if gammas.len() > 1 {
                // Find mode gamma (most common value, 1% relative tolerance)
                let mut sorted = gammas.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut best_val = sorted[0];
                let mut best_count = 1usize;
                let mut cur_val = sorted[0];
                let mut cur_count = 1usize;
                let tol = (sorted.last().unwrap_or(&1.0) * 0.01).max(1e-6);
                for &g in &sorted[1..] {
                    if (g - cur_val).abs() < tol {
                        cur_count += 1;
                    } else {
                        if cur_count > best_count { best_count = cur_count; best_val = cur_val; }
                        cur_val = g; cur_count = 1;
                    }
                }
                if cur_count > best_count { best_val = cur_val; }
                // Mark soft cells
                for c in &mut centroids {
                    let cid = c.0 as usize;
                    if cid < gammas.len() {
                        c.3 = (gammas[cid] - best_val).abs() > tol && gammas[cid] < best_val;
                    }
                }
            }
        }
    }
    centroids
}

// ── Rendering helpers ───────────────────────────────────────────────────

/// Render a single VTK file to RGB pixel data.
fn render_single_vtk(
    vtk_path: &std::path::Path,
    label_cells: bool,
    centroids: Option<&[(u32, f64, f64, bool)]>,
) -> Result<(Vec<u8>, usize, usize, String)> {
    let vtk_data = vtk::parse_vtk(vtk_path)?;
    let phi_field = vtk_data.scalars.get("phi")
        .ok_or_else(|| anyhow::anyhow!("No 'phi' field in VTK file"))?;
    let nx = vtk_data.dims.nx;
    let ny = vtk_data.dims.ny;
    let stem = vtk_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let step: i64 = stem.strip_prefix("frame_").and_then(|s| s.parse().ok()).unwrap_or(0);
    let time = step as f64 * 0.01;
    let title = format!("step={}, t={:.0} ({:.1}τ)", step, time, time / 10000.0);

    if label_cells {
        // Use provided centroids, or try to load from trajectory/checkpoint
        let owned_centroids;
        let cents = if let Some(c) = centroids {
            c
        } else {
            owned_centroids = load_centroids_for_vtk(vtk_path, time);
            &owned_centroids
        };
        let (img_data, _, _) = render_phi_to_rgb(phi_field, nx, ny, true, cents, &[], &[], &std::collections::HashMap::new());
        Ok((img_data, nx, ny, title))
    } else {
        let empty_centroids = Vec::new();
        let (img_data, _, _) = render_phi_to_rgb(phi_field, nx, ny, false, &empty_centroids, &[], &[], &std::collections::HashMap::new());
        Ok((img_data, nx, ny, title))
    }
}

/// Convert phi field to RGB image data.
fn render_phi_to_rgb(
    phi: &[f32], nx: usize, ny: usize,
    label_cells: bool, centroids: &[(u32, f64, f64, bool)],
    polarity: &[(f64, f64, f64, f64)],   // (cx, cy, px, py)
    energy: &[(f64, f64, f64)],           // (cx, cy, ke)
    cell_above_text: &std::collections::HashMap<u32, String>,
) -> (Vec<u8>, usize, usize) {
    let mut img_data = vec![0u8; nx * ny * 3];

    // If energy overlay, build a per-pixel KE map for coloring
    let ke_max = energy.iter().map(|e| e.2).fold(0.0f64, f64::max);

    for y in 0..ny {
        for x in 0..nx {
            let val = phi[y * nx + x] as f64;
            let (r, g, b) = phi_colormap(val.clamp(0.0, 1.0));
            let iy = ny - 1 - y;
            let idx = (iy * nx + x) * 3;
            img_data[idx] = r;
            img_data[idx + 1] = g;
            img_data[idx + 2] = b;
        }
    }

    // Energy overlay: tint cell interiors by KE (blue=low, red=high)
    if !energy.is_empty() && ke_max > 1e-20 {
        let cell_radius = 49.0; // approximate
        let r_sq = (cell_radius * 1.5) * (cell_radius * 1.5);
        for y in 0..ny {
            for x in 0..nx {
                let val = phi[y * nx + x] as f64;
                if val < 0.3 { continue; } // only color inside cells
                // Find nearest cell
                let mut best_ke = 0.0;
                let mut best_d2 = f64::MAX;
                for &(cx, cy, ke) in energy {
                    let mut dx = x as f64 - ((cx % nx as f64) + nx as f64) % nx as f64;
                    let mut dy = y as f64 - ((cy % ny as f64) + ny as f64) % ny as f64;
                    if dx > nx as f64 / 2.0 { dx -= nx as f64; }
                    if dx < -(nx as f64 / 2.0) { dx += nx as f64; }
                    if dy > ny as f64 / 2.0 { dy -= ny as f64; }
                    if dy < -(ny as f64 / 2.0) { dy += ny as f64; }
                    let d2 = dx * dx + dy * dy;
                    if d2 < best_d2 && d2 < r_sq {
                        best_d2 = d2;
                        best_ke = ke;
                    }
                }
                if best_d2 < r_sq {
                    let frac = (best_ke / ke_max).clamp(0.0, 1.0);
                    let (er, eg, eb) = rdylbu_colormap(frac);
                    let alpha = val.clamp(0.3, 1.0) * 0.6; // blend strength
                    let iy = ny - 1 - y;
                    let idx = (iy * nx + x) * 3;
                    img_data[idx] = (img_data[idx] as f64 * (1.0 - alpha) + er as f64 * alpha) as u8;
                    img_data[idx+1] = (img_data[idx+1] as f64 * (1.0 - alpha) + eg as f64 * alpha) as u8;
                    img_data[idx+2] = (img_data[idx+2] as f64 * (1.0 - alpha) + eb as f64 * alpha) as u8;
                }
            }
        }
    }

    // Polarity arrows
    if !polarity.is_empty() {
        let arrow_len = 25.0;
        let arrow_color: [u8; 3] = [255, 255, 0]; // yellow
        for &(cx, cy, px, py) in polarity {
            let cx_w = ((cx % nx as f64) + nx as f64) % nx as f64;
            let cy_w = ((cy % ny as f64) + ny as f64) % ny as f64;
            // Draw line from centroid in polarity direction
            let ex = cx_w + px * arrow_len;
            let ey = cy_w + py * arrow_len;
            // Bresenham-style line
            let steps = 40;
            for s in 0..=steps {
                let t = s as f64 / steps as f64;
                let lx = (cx_w + t * (ex - cx_w)) as i32;
                let ly = (cy_w + t * (ey - cy_w)) as i32;
                let lx = ((lx % nx as i32) + nx as i32) as usize % nx;
                let ly = ((ly % ny as i32) + ny as i32) as usize % ny;
                let iy = ny - 1 - ly;
                let _idx = (iy * nx + lx) * 3;
                // Thicker line: draw 3×3
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let px = (lx as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                        let py = (iy as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                        let pidx = (py * nx + px) * 3;
                        img_data[pidx] = arrow_color[0];
                        img_data[pidx+1] = arrow_color[1];
                        img_data[pidx+2] = arrow_color[2];
                    }
                }
            }
            // Arrowhead: small triangle at tip
            let tip_x = ((ex as i32 % nx as i32) + nx as i32) as usize % nx;
            let tip_y = ((ey as i32 % ny as i32) + ny as i32) as usize % ny;
            let tip_iy = ny - 1 - tip_y;
            for dy in -3i32..=3 {
                for dx in -3i32..=3 {
                    if dx * dx + dy * dy <= 9 {
                        let px = (tip_x as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                        let py = (tip_iy as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                        let pidx = (py * nx + px) * 3;
                        img_data[pidx] = 255;
                        img_data[pidx+1] = 200;
                        img_data[pidx+2] = 0;
                    }
                }
            }
        }
    }

    if label_cells && !centroids.is_empty() {
        // Build is_soft lookup from centroids
        let soft_map: std::collections::HashMap<u32, bool> = centroids.iter()
            .map(|&(cid, _, _, is_soft)| (cid, is_soft))
            .collect();
        let mut wrapped: Vec<(u32, f64, f64)> = centroids.iter()
            .map(|&(cid, cx, cy, _)| {
                let wx = ((cx % nx as f64) + nx as f64) % nx as f64;
                let wy = ((cy % ny as f64) + ny as f64) % ny as f64;
                (cid, wx, wy)
            }).collect();
        let r_band = 80.0;
        wrapped.sort_by(|a, b| {
            let row_a = (a.2 / r_band) as i32;
            let row_b = (b.2 / r_band) as i32;
            row_b.cmp(&row_a).then(a.1.partial_cmp(&b.1).unwrap())
        });
        let spatial_map: std::collections::HashMap<u32, usize> = wrapped.iter()
            .enumerate()
            .map(|(i, &(cid, _, _))| (cid, i + 1))
            .collect();
        for &(cid, wx, wy) in &wrapped {
            let ix = wx as i32;
            let iy = (ny as i32 - 1) - wy as i32;
            let spatial_idx = spatial_map[&cid];
            let highlight = soft_map.get(&cid).copied().unwrap_or(false);
            let above = cell_above_text.get(&cid).map(|s| s.as_str()).unwrap_or("");
            draw_label_with_spatial(&mut img_data, nx, ny, ix, iy,
                &format!("{}", cid), &format!("{}", spatial_idx), above, highlight);
        }
    }

    (img_data, nx, ny)
}

/// 5×7 bitmap font for plot banner text. Each glyph is 7 rows; bits are 5 wide
/// (lowest 5 bits, MSB = leftmost pixel). Returns all-zero for unsupported chars.
fn glyph_5x7(c: char) -> [u8; 7] {
    match c {
        ' ' => [0,0,0,0,0,0,0],
        '0' => [0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110],
        '1' => [0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110],
        '2' => [0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111],
        '3' => [0b11110,0b00001,0b00001,0b01110,0b00001,0b00001,0b11110],
        '4' => [0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010],
        '5' => [0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110],
        '6' => [0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110],
        '7' => [0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000],
        '8' => [0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110],
        '9' => [0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100],
        'A' => [0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
        'B' => [0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110],
        'C' => [0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110],
        'D' => [0b11100,0b10010,0b10001,0b10001,0b10001,0b10010,0b11100],
        'E' => [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111],
        'F' => [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000],
        'G' => [0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01111],
        'H' => [0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
        'I' => [0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110],
        'J' => [0b00111,0b00010,0b00010,0b00010,0b00010,0b10010,0b01100],
        'K' => [0b10001,0b10010,0b10100,0b11000,0b10100,0b10010,0b10001],
        'L' => [0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111],
        'M' => [0b10001,0b11011,0b10101,0b10101,0b10001,0b10001,0b10001],
        'N' => [0b10001,0b10001,0b11001,0b10101,0b10011,0b10001,0b10001],
        'O' => [0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        'P' => [0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000],
        'Q' => [0b01110,0b10001,0b10001,0b10001,0b10101,0b10010,0b01101],
        'R' => [0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001],
        'S' => [0b01110,0b10001,0b10000,0b01110,0b00001,0b10001,0b01110],
        'T' => [0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100],
        'U' => [0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        'V' => [0b10001,0b10001,0b10001,0b10001,0b10001,0b01010,0b00100],
        'W' => [0b10001,0b10001,0b10001,0b10101,0b10101,0b10101,0b01010],
        'X' => [0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001],
        'Y' => [0b10001,0b10001,0b10001,0b01010,0b00100,0b00100,0b00100],
        'Z' => [0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111],
        'a' => [0,0,0b01110,0b00001,0b01111,0b10001,0b01111],
        'b' => [0b10000,0b10000,0b11110,0b10001,0b10001,0b10001,0b11110],
        'c' => [0,0,0b01110,0b10001,0b10000,0b10001,0b01110],
        'd' => [0b00001,0b00001,0b01111,0b10001,0b10001,0b10001,0b01111],
        'e' => [0,0,0b01110,0b10001,0b11111,0b10000,0b01110],
        'f' => [0b00110,0b01001,0b01000,0b11110,0b01000,0b01000,0b01000],
        'g' => [0,0,0b01111,0b10001,0b01111,0b00001,0b01110],
        'h' => [0b10000,0b10000,0b11110,0b10001,0b10001,0b10001,0b10001],
        'i' => [0b00100,0,0b01100,0b00100,0b00100,0b00100,0b01110],
        'j' => [0b00010,0,0b00110,0b00010,0b00010,0b10010,0b01100],
        'k' => [0b10000,0b10000,0b10010,0b10100,0b11000,0b10100,0b10010],
        'l' => [0b01100,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110],
        'm' => [0,0,0b11010,0b10101,0b10101,0b10101,0b10101],
        'n' => [0,0,0b11110,0b10001,0b10001,0b10001,0b10001],
        'o' => [0,0,0b01110,0b10001,0b10001,0b10001,0b01110],
        'p' => [0,0,0b11110,0b10001,0b11110,0b10000,0b10000],
        'q' => [0,0,0b01111,0b10001,0b01111,0b00001,0b00001],
        'r' => [0,0,0b10110,0b11001,0b10000,0b10000,0b10000],
        's' => [0,0,0b01110,0b10000,0b01110,0b00001,0b11110],
        't' => [0b01000,0b01000,0b11110,0b01000,0b01000,0b01001,0b00110],
        'u' => [0,0,0b10001,0b10001,0b10001,0b10001,0b01110],
        'v' => [0,0,0b10001,0b10001,0b10001,0b01010,0b00100],
        'w' => [0,0,0b10001,0b10001,0b10101,0b10101,0b01010],
        'x' => [0,0,0b10001,0b01010,0b00100,0b01010,0b10001],
        'y' => [0,0,0b10001,0b10001,0b01111,0b00001,0b01110],
        'z' => [0,0,0b11111,0b00010,0b00100,0b01000,0b11111],
        '.' => [0,0,0,0,0,0b01100,0b01100],
        ',' => [0,0,0,0,0b01100,0b00100,0b01000],
        ':' => [0,0b01100,0b01100,0,0b01100,0b01100,0],
        '-' => [0,0,0,0b11111,0,0,0],
        '_' => [0,0,0,0,0,0,0b11111],
        '=' => [0,0,0b11111,0,0b11111,0,0],
        '/' => [0b00001,0b00010,0b00010,0b00100,0b01000,0b01000,0b10000],
        '+' => [0,0b00100,0b00100,0b11111,0b00100,0b00100,0],
        '%' => [0b11001,0b11010,0b00100,0b00100,0b01000,0b01011,0b10011],
        '(' => [0b00010,0b00100,0b01000,0b01000,0b01000,0b00100,0b00010],
        ')' => [0b01000,0b00100,0b00010,0b00010,0b00010,0b00100,0b01000],
        '#' => [0b01010,0b01010,0b11111,0b01010,0b11111,0b01010,0b01010],
        'τ' => [0b11111,0b00100,0b00100,0b00100,0b00100,0b00101,0b00010],
        'φ' => [0b00100,0b01110,0b10101,0b10101,0b10101,0b01110,0b00100],
        'ρ' => [0,0,0b01110,0b10001,0b10001,0b11110,0b10000],
        'γ' => [0,0b10001,0b10010,0b01010,0b00100,0b00100,0b01000],
        '×' => [0,0b00100,0b01010,0b11111,0b01010,0b00100,0],
        _ => [0,0,0,0,0,0,0],
    }
}

/// Draw a string at (x, y) using 5×7 bitmap font, scaled by `scale`.
fn draw_text_5x7(img: &mut [u8], w: usize, h: usize,
                 x: i32, y: i32, text: &str, scale: i32, rgb: [u8; 3]) -> i32 {
    let glyph_w = 5 * scale;
    let kern = scale; // 1px gap between glyphs (scaled)
    let mut cur_x = x;
    for ch in text.chars() {
        let bm = glyph_5x7(ch);
        for row in 0..7 {
            for col in 0..5 {
                if bm[row] & (1u8 << (4 - col)) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cur_x + (col as i32) * scale + sx;
                            let py = y + (row as i32) * scale + sy;
                            if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                                let idx = (py as usize * w + px as usize) * 3;
                                img[idx] = rgb[0];
                                img[idx + 1] = rgb[1];
                                img[idx + 2] = rgb[2];
                            }
                        }
                    }
                }
            }
        }
        cur_x += glyph_w + kern;
    }
    cur_x
}

/// Compose a metadata banner at the top of an RGB image. Returns the new
/// (image_data, width, height) with the banner stacked above the original.
fn compose_with_banner(
    img: &[u8], nx: usize, ny: usize,
    lines: &[String],
) -> (Vec<u8>, usize, usize) {
    if lines.is_empty() {
        return (img.to_vec(), nx, ny);
    }
    // Auto-scale font: prefer larger, but shrink if widest line wouldn't fit.
    let max_chars = lines.iter().map(|s| s.chars().count()).max().unwrap_or(1);
    let pad_x_per_scale = 4i32;
    // Glyph cell width = 5 * scale + scale (kerning) = 6 * scale per char
    // Total width for a scale s = 6 * s * max_chars + 2 * pad_x_per_scale * s
    // Solve for largest s such that total <= nx
    let max_scale: i32 = (1..=4).rev()
        .find(|&s| (6 * s * max_chars as i32 + 2 * pad_x_per_scale * s) <= nx as i32)
        .unwrap_or(1);
    let scale: i32 = max_scale.min(if nx >= 1200 { 3 } else if nx >= 600 { 2 } else { 1 });
    let scale = scale.max(1);
    let glyph_h = 7 * scale;
    let line_gap = 2 * scale;
    let pad_y = 4 * scale;
    let banner_h = (pad_y * 2) + (glyph_h * lines.len() as i32) + (line_gap * (lines.len() as i32 - 1).max(0));
    let banner_h = banner_h.max(0) as usize;
    let new_h = ny + banner_h;
    let mut out = vec![0u8; new_h * nx * 3];

    // Banner background: dark grey-blue
    for y in 0..banner_h {
        for x in 0..nx {
            let idx = (y * nx + x) * 3;
            out[idx] = 18; out[idx + 1] = 22; out[idx + 2] = 32;
        }
    }
    // Draw text lines
    let pad_x: i32 = pad_x_per_scale * scale;
    let mut cur_y: i32 = pad_y;
    for line in lines {
        draw_text_5x7(&mut out, nx, banner_h, pad_x, cur_y, line, scale, [230, 230, 235]);
        cur_y += glyph_h + line_gap;
    }
    // Copy original image below banner
    let dst_off = banner_h * nx * 3;
    out[dst_off..dst_off + img.len()].copy_from_slice(img);
    (out, nx, new_h)
}

/// Build human-readable metadata lines for a checkpoint snapshot.
fn build_metadata_lines(
    ckpt: &analysis::checkpoint::Checkpoint,
    marker: Option<&analysis::metadata::SimMarker>,
    ckpt_path: &std::path::Path,
) -> Vec<String> {
    use analysis::metadata::{compute_confluence, marker_param_f64};
    let p = &ckpt.params;
    let h = &ckpt.header;
    let lx = (p.nx as f32) * p.dx;
    let ly = (p.ny as f32) * p.dy;
    let phi_computed = compute_confluence(h.num_cells, p.target_radius, lx, ly);

    let mut lines: Vec<String> = Vec::new();

    // Line 1: filename / source
    if let Some(stem) = ckpt_path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) {
        lines.push(format!("Source: {}", stem));
    }

    // Line 2: timestamp + study
    let ts = marker.and_then(|m| m.provenance.as_ref()).and_then(|p| p.timestamp.clone());
    let study = marker.and_then(|m| m.study.as_ref()).and_then(|s| s.name.clone());
    match (ts, study) {
        (Some(t), Some(s)) => lines.push(format!("Submitted: {}  study: {}", t, s)),
        (Some(t), None)    => lines.push(format!("Submitted: {}", t)),
        (None, Some(s))    => lines.push(format!("Study: {}", s)),
        _ => {}
    }

    // Line 3: physics: N, R, dt, t (sim time)
    lines.push(format!(
        "N={}  R={:.1}  dt={:.3}  t={:.0} ({:.2}tau)  step={}",
        h.num_cells, p.target_radius, p.dt, h.time, h.time / 10000.0, h.step
    ));

    // Line 4: domain + confluence (computed vs original)
    let phi_orig = marker.and_then(|m| marker_param_f64(m, "confluence"));
    let phi_str = match phi_orig {
        Some(o) => format!(
            "phi_computed={:.4}  phi_original={:.4}  delta={:+.4}",
            phi_computed, o, phi_computed - o
        ),
        None => format!("phi_computed={:.4}  phi_original=N/A", phi_computed),
    };
    lines.push(format!("Domain: {}x{} ({:.0}x{:.0})  {}",
        p.nx, p.ny, lx, ly, phi_str));

    // Line 5: motility scalars; Line 6: per-cell stats
    let g = &ckpt.per_cell_gamma;
    let v = &ckpt.per_cell_v_a;
    let r = &ckpt.per_cell_radius;
    let stats = |xs: &[f32]| -> Option<(f32, f32, f32)> {
        if xs.is_empty() { return None; }
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        let mut s = 0.0f64;
        for &x in xs { mn = mn.min(x); mx = mx.max(x); s += x as f64; }
        Some((mn, (s / xs.len() as f64) as f32, mx))
    };
    lines.push(format!("v_A_param={:.4}  tau={:.0}  lambda={:.2}  halo={}",
        p.v_a, p.tau, p.lambda, p.halo_width));
    let seed_str = match p.seed {
        Some(s) => format!("{}", s),
        None => "n/a".to_string(),
    };
    let pol_seed_str = match p.polarity_seed {
        Some(s) => format!("{}", s),
        None => "n/a".to_string(),
    };
    lines.push(format!("seed={}  polarity_seed={}", seed_str, pol_seed_str));
    let mut cell_parts: Vec<String> = Vec::new();
    if let Some((mn, mean, mx)) = stats(v) {
        cell_parts.push(format!("v_A_cell[min/avg/max]={:.4}/{:.4}/{:.4}", mn, mean, mx));
    }
    if let Some((mn, mean, mx)) = stats(g) {
        cell_parts.push(format!("gamma=[{:.3}/{:.3}/{:.3}]", mn, mean, mx));
    }
    if let Some((mn, mean, mx)) = stats(r) {
        cell_parts.push(format!("R=[{:.2}/{:.2}/{:.2}]", mn, mean, mx));
    }
    if !cell_parts.is_empty() {
        lines.push(cell_parts.join("  "));
    }

    // Line 6: provenance jobs / source checkpoint (if resume)
    if let Some(prov) = marker.and_then(|m| m.provenance.as_ref()) {
        let mut prov_parts: Vec<String> = Vec::new();
        if !prov.job_ids.is_empty() {
            prov_parts.push(format!("jobs={}", prov.job_ids.join(",")));
        }
        if let Some(src) = &prov.source_checkpoint {
            // Show only last two path components for brevity
            let short = std::path::Path::new(src);
            let trail: Vec<&str> = short.iter().rev().take(3)
                .filter_map(|s| s.to_str()).collect();
            let mut joined = trail.into_iter().rev().collect::<Vec<_>>().join("/");
            if joined.is_empty() { joined = src.clone(); }
            prov_parts.push(format!("resumed_from=.../{}", joined));
        }
        if !prov_parts.is_empty() {
            lines.push(prov_parts.join("  "));
        }
    }
    lines
}

/// Write RGB image data to a PNG file.
fn write_png(path: &std::path::Path, img_data: &[u8], nx: usize, ny: usize) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, nx as u32, ny as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(img_data)?;
    Ok(())
}

fn phi_colormap(val: f64) -> (u8, u8, u8) {
    if val < 0.01 {
        return (49, 54, 149); // deep blue for background
    }
    // Simplified 5-stop RdYlBu_r
    let stops: [(f64, f64, f64); 5] = [
        (0.192, 0.212, 0.584),  // 0.0: blue
        (0.557, 0.769, 0.867),  // 0.25: light blue
        (1.000, 1.000, 0.749),  // 0.5: yellow
        (0.957, 0.427, 0.263),  // 0.75: orange
        (0.647, 0.059, 0.082),  // 1.0: dark red
    ];
    let t = val.clamp(0.0, 1.0) * 4.0;
    let i = (t as usize).min(3);
    let frac = t - i as f64;
    let (r0, g0, b0) = stops[i];
    let (r1, g1, b1) = stops[i + 1];
    let r = r0 + (r1 - r0) * frac;
    let g = g0 + (g1 - g0) * frac;
    let b = b0 + (b1 - b0) * frac;
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Blue→White→Red colormap for energy overlay (0=blue, 0.5=white, 1=red).
fn rdylbu_colormap(frac: f64) -> (u8, u8, u8) {
    let t = frac.clamp(0.0, 1.0);
    if t < 0.5 {
        let f = t * 2.0;
        let r = (49.0 + (255.0 - 49.0) * f) as u8;
        let g = (54.0 + (255.0 - 54.0) * f) as u8;
        let b = (149.0 + (255.0 - 149.0) * f) as u8;
        (r, g, b)
    } else {
        let f = (t - 0.5) * 2.0;
        let r = 255;
        let g = (255.0 * (1.0 - f)) as u8;
        let b = (255.0 * (1.0 - f)) as u8;
        (r, g, b)
    }
}

fn write_json<T: serde::Serialize>(data: &T, output: &Option<PathBuf>) -> Result<()> {
    let json = serde_json::to_string_pretty(data)?;
    match output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, &json).context("Writing output file")?;
            eprintln!("Output written to {}", path.display());
        }
        None => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(json.as_bytes())?;
            handle.write_all(b"\n")?;
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-cell movie rendering: contour coloring + speed shading
// ═══════════════════════════════════════════════════════════════════════════

/// Which property colors cell contours.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorBy { VA, Gamma, CellId, None }

/// Pre-loaded context for per-cell movie rendering.
struct MovieContext {
    color_by: ColorBy,
    /// Per-cell property values for contour coloring (v_A or gamma, indexed by cell_id)
    per_cell_color: Vec<f32>,
    color_min: f32,
    color_range: f32,
    shade_speed: bool,
    traj: Vec<(f32, std::collections::HashMap<u32, [f32; 2]>)>, // (time, cell_id -> [x, y])
    intervals: Vec<DisplacementInterval>,
    speed_max: f32,
    speed_window: usize,
    nx: usize,
    ny: usize,
    boundary_dilations: usize,
}

struct DisplacementInterval {
    time: f32,
    displacements: std::collections::HashMap<u32, f32>,
}

impl MovieContext {
    fn load(vtk_dir: &std::path::Path, nx: usize, ny: usize, _selected: &[&PathBuf],
            speed_window: usize, color_by_str: &str, shade_speed: bool) -> Result<Self> {
        // Load checkpoint for per-cell properties
        let ckpt_path = vtk_dir.join("checkpoint.bin");
        let ckpt = if ckpt_path.exists() {
            Some(analysis::checkpoint::load_checkpoint(&ckpt_path)?)
        } else {
            None
        };

        // Determine what to color by
        let (color_by, per_cell_color) = match color_by_str {
            "v_a" | "va" => {
                let vals = ckpt.as_ref().map(|c| c.per_cell_v_a.clone()).unwrap_or_default();
                if vals.is_empty() { anyhow::bail!("--color-by v_a requires checkpoint with per-cell v_A"); }
                (ColorBy::VA, vals)
            }
            "gamma" => {
                let vals = ckpt.as_ref().map(|c| c.per_cell_gamma.clone()).unwrap_or_default();
                if vals.is_empty() { anyhow::bail!("--color-by gamma requires checkpoint with per-cell gamma"); }
                (ColorBy::Gamma, vals)
            }
            "cell_id" => {
                let n = ckpt.as_ref().map(|c| c.header.num_cells as usize).unwrap_or(0);
                let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
                (ColorBy::CellId, vals)
            }
            "none" => (ColorBy::None, Vec::new()),
            "auto" | _ => {
                // Auto-detect: check if per-cell v_A varies, then gamma, else cell_id
                if let Some(ref c) = ckpt {
                    let va = &c.per_cell_v_a;
                    let gamma = &c.per_cell_gamma;
                    if va.len() > 1 {
                        let va_min = va.iter().copied().fold(f32::MAX, f32::min);
                        let va_max = va.iter().copied().fold(f32::MIN, f32::max);
                        if (va_max - va_min) > va_min * 0.01 {
                            eprintln!("  Auto-detected: coloring contours by v_A (range [{:.4}, {:.4}])", va_min, va_max);
                            (ColorBy::VA, va.clone())
                        } else if gamma.len() > 1 {
                            let g_min = gamma.iter().copied().fold(f32::MAX, f32::min);
                            let g_max = gamma.iter().copied().fold(f32::MIN, f32::max);
                            if (g_max - g_min) > g_min * 0.01 {
                                eprintln!("  Auto-detected: coloring contours by gamma (range [{:.4}, {:.4}])", g_min, g_max);
                                (ColorBy::Gamma, gamma.clone())
                            } else {
                                eprintln!("  Uniform v_A and gamma: coloring contours by cell_id");
                                let n = c.header.num_cells as usize;
                                (ColorBy::CellId, (0..n).map(|i| i as f32).collect())
                            }
                        } else {
                            eprintln!("  No per-cell gamma: coloring contours by cell_id");
                            let n = c.header.num_cells as usize;
                            (ColorBy::CellId, (0..n).map(|i| i as f32).collect())
                        }
                    } else if gamma.len() > 1 {
                        let g_min = gamma.iter().copied().fold(f32::MAX, f32::min);
                        let g_max = gamma.iter().copied().fold(f32::MIN, f32::max);
                        if (g_max - g_min) > g_min * 0.01 {
                            eprintln!("  Auto-detected: coloring contours by gamma (range [{:.4}, {:.4}])", g_min, g_max);
                            (ColorBy::Gamma, gamma.clone())
                        } else {
                            let n = c.header.num_cells as usize;
                            (ColorBy::CellId, (0..n).map(|i| i as f32).collect())
                        }
                    } else {
                        let n = c.header.num_cells as usize;
                        (ColorBy::CellId, (0..n).map(|i| i as f32).collect())
                    }
                } else {
                    eprintln!("  No checkpoint: coloring disabled");
                    (ColorBy::None, Vec::new())
                }
            }
        };

        let color_min = per_cell_color.iter().copied().fold(f32::MAX, f32::min);
        let color_max = per_cell_color.iter().copied().fold(f32::MIN, f32::max);
        let color_range = color_max - color_min;
        if color_by != ColorBy::None {
            eprintln!("  Contour color: {:?}, range [{:.4}, {:.4}]", color_by, color_min, color_max);
        }

        // Load trajectory for centroids and speed computation
        let traj_path = vtk_dir.join("trajectory.txt");
        let traj = if traj_path.exists() {
            Self::load_traj_positions(&traj_path)?
        } else if shade_speed {
            anyhow::bail!("--shade-speed requires trajectory.txt for displacement computation");
        } else {
            Vec::new()
        };
        if !traj.is_empty() {
            eprintln!("  Trajectory: {} time samples", traj.len());
        }

        // Compute displacement intervals (only if shade_speed is on).
        // intervals is consumed downstream by path_avg_speed_static; it must
        // outlive the `if` so don't shadow with an inner let-binding.
        let mut intervals: Vec<DisplacementInterval> = Vec::new();
        let mut speed_max_val = 1e-6f32;
        if shade_speed && traj.len() > 1 {
            let hlx = nx as f32 / 2.0;
            let hly = ny as f32 / 2.0;
            let lx = nx as f32;
            let ly = ny as f32;
            intervals.reserve(traj.len());
            for i in 1..traj.len() {
                let (t_now, cells_now) = &traj[i];
                let (t_prev, cells_prev) = &traj[i - 1];
                let dt = t_now - t_prev;
                if dt <= 0.0 { continue; }
                let mut dr_map = std::collections::HashMap::new();
                for (&cid, pos) in cells_now {
                    if let Some(prev) = cells_prev.get(&cid) {
                        let mut dx = pos[0] - prev[0];
                        let mut dy = pos[1] - prev[1];
                        if dx > hlx { dx -= lx; } if dx < -hlx { dx += lx; }
                        if dy > hly { dy -= ly; } if dy < -hly { dy += ly; }
                        dr_map.insert(cid, (dx * dx + dy * dy).sqrt());
                    }
                }
                intervals.push(DisplacementInterval { time: *t_now, displacements: dr_map });
            }

            // Fixed global speed normalization: scan trajectory for P95
            speed_max_val = {
                let mut all_speeds: Vec<f32> = Vec::new();
                let n_samples = 50.min(intervals.len());
                let sample_step = intervals.len().max(1) / n_samples.max(1);
                for si in (0..intervals.len()).step_by(sample_step.max(1)) {
                    let t = intervals[si].time;
                    let speeds = Self::path_avg_speed_static(&intervals, t, speed_window);
                    all_speeds.extend(speeds.values());
                }
                if all_speeds.is_empty() {
                    1e-6
                } else {
                    all_speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p95 = all_speeds[(all_speeds.len() as f32 * 0.95) as usize];
                    p95.max(1e-6)
                }
            };
            eprintln!("  Speed scale (P95): {:.6}", speed_max_val);
        }

        let cell_radius = ckpt.as_ref().map(|c| c.params.target_radius as f32).unwrap_or(49.0);
        let boundary_dilations = ((cell_radius / 25.0).round() as usize).max(1);

        Ok(Self { color_by, per_cell_color, color_min, color_range, shade_speed,
                  traj, intervals, speed_max: speed_max_val, speed_window, nx, ny, boundary_dilations })
    }

    fn load_traj_positions(path: &std::path::Path) -> Result<Vec<(f32, std::collections::HashMap<u32, [f32; 2]>)>> {
        let file = std::fs::File::open(path).context("Opening trajectory.txt")?;
        let reader = std::io::BufReader::new(file);
        let mut by_time: std::collections::HashMap<i64, std::collections::HashMap<u32, [f32; 2]>> = std::collections::HashMap::new();
        let mut time_order: Vec<(i64, f32)> = Vec::new();
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if line.starts_with('#') || line.is_empty() { continue; }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 { continue; }
            let t: f32 = parts[0].parse().unwrap_or(0.0);
            let cid: u32 = parts[1].parse().unwrap_or(0);
            let x: f32 = parts[2].parse().unwrap_or(0.0);
            let y: f32 = parts[3].parse().unwrap_or(0.0);
            let key = (t * 100.0) as i64;
            let frame = by_time.entry(key).or_insert_with(|| { time_order.push((key, t)); std::collections::HashMap::new() });
            frame.insert(cid, [x, y]);
        }
        time_order.sort_by_key(|&(k, _)| k);
        Ok(time_order.into_iter().map(|(k, t)| (t, by_time.remove(&k).unwrap())).collect())
    }

    fn path_avg_speed_static(intervals: &[DisplacementInterval], frame_time: f32, window: usize) -> std::collections::HashMap<u32, f32> {
        if intervals.is_empty() { return std::collections::HashMap::new(); }
        let center = intervals.iter().enumerate()
            .min_by(|(_, a), (_, b)| (a.time - frame_time).abs().partial_cmp(&(b.time - frame_time).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let half = window / 2;
        let lo = center.saturating_sub(half);
        let hi = (lo + window).min(intervals.len());
        let lo = hi.saturating_sub(window);
        let mut path_len: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        let mut total_dt: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for idx in lo..hi {
            let iv = &intervals[idx];
            for (&cid, &dr) in &iv.displacements {
                *path_len.entry(cid).or_insert(0.0) += dr;
                *total_dt.entry(cid).or_insert(0.0) += 1.0; // count frames
            }
        }
        path_len.into_iter().map(|(cid, pl)| {
            let dt = total_dt.get(&cid).copied().unwrap_or(1.0);
            (cid, if dt > 0.0 { pl / dt } else { 0.0 })
        }).collect()
    }

    fn render_frame(&self, vtk_path: &std::path::Path, _frame_idx: usize, label_cells: bool) -> Result<Vec<u8>> {
        let vtk_data = vtk::parse_vtk(vtk_path)?;
        let phi = vtk_data.scalars.get("phi")
            .ok_or_else(|| anyhow::anyhow!("No phi field in {:?}", vtk_path))?;
        let nx = self.nx;
        let ny = self.ny;

        // Get frame time from filename
        let stem = vtk_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let step: u64 = stem.strip_prefix("frame_").and_then(|s| s.parse().ok()).unwrap_or(0);
        let frame_time = step as f32 * 0.01; // dt=0.01

        // Find closest trajectory frame
        let traj_times: Vec<f32> = self.traj.iter().map(|t| t.0).collect();
        let closest_idx = traj_times.iter().enumerate()
            .min_by(|(_, a), (_, b)| ((**a) - frame_time).abs().partial_cmp(&((**b) - frame_time).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let centroids = &self.traj[closest_idx].1;
        let cell_speeds = if self.shade_speed {
            Self::path_avg_speed_static(&self.intervals, frame_time, self.speed_window)
        } else {
            std::collections::HashMap::new()
        };

        // Compute gradient magnitude
        let grad_mag: Vec<f32> = (0..ny).flat_map(|y| {
            (0..nx).map(move |x| {
                let idx = y * nx + x;
                let xp = if x + 1 < nx { phi[idx + 1] } else { phi[y * nx] };
                let xm = if x > 0 { phi[idx - 1] } else { phi[y * nx + nx - 1] };
                let yp = if y + 1 < ny { phi[(y + 1) * nx + x] } else { phi[x] };
                let ym = if y > 0 { phi[(y - 1) * nx + x] } else { phi[(ny - 1) * nx + x] };
                let gx = (xp - xm) * 0.5;
                let gy = (yp - ym) * 0.5;
                (gx * gx + gy * gy).sqrt()
            })
        }).collect();

        // Watershed segmentation
        let labels = watershed(phi, &grad_mag, nx, ny, centroids);

        // Cell mask and erosion for boundary ring
        let cell_mask: Vec<bool> = phi.iter().map(|&p| p > 0.5).collect();
        let mut interior = cell_mask.clone();
        for _ in 0..self.boundary_dilations {
            interior = erode_mask(&interior, nx, ny);
        }

        // Render pixels
        let mut rgb = vec![0u8; nx * ny * 3];
        for y in 0..ny {
            let iy = ny - 1 - y; // flip Y
            for x in 0..nx {
                let idx = y * nx + x;
                let dst = (iy * nx + x) * 3;

                if !cell_mask[idx] {
                    if self.color_by == ColorBy::None {
                        // When color_by=none, show phi heatmap for background too
                        let val = phi[idx].clamp(0.0, 1.0) as f64;
                        let (r, g, b) = phi_colormap(val);
                        rgb[dst] = r; rgb[dst + 1] = g; rgb[dst + 2] = b;
                    } else {
                        rgb[dst] = 31; rgb[dst + 1] = 31; rgb[dst + 2] = 38;
                    }
                    continue;
                }

                let label = labels[idx];
                if interior[idx] && label >= 0 {
                    // Interior
                    let cid = label as u32;
                    if self.shade_speed {
                        let speed = cell_speeds.get(&cid).copied().unwrap_or(0.0);
                        let intensity = (speed / self.speed_max).clamp(0.08, 1.0);
                        let brightness = (intensity * phi[idx].clamp(0.0, 1.0) * 255.0) as u8;
                        rgb[dst] = brightness; rgb[dst + 1] = brightness; rgb[dst + 2] = brightness;
                    } else {
                        // Default interior: muted phi heatmap
                        let val = phi[idx].clamp(0.0, 1.0) as f64;
                        let (r, g, b) = phi_colormap(val);
                        rgb[dst] = (r as f32 * 0.7) as u8;
                        rgb[dst + 1] = (g as f32 * 0.7) as u8;
                        rgb[dst + 2] = (b as f32 * 0.7) as u8;
                    }
                } else if label >= 0 {
                    // Boundary ring: color by selected property
                    let cid = label as u32;
                    let (r, g, b) = match self.color_by {
                        ColorBy::VA | ColorBy::Gamma => {
                            let val = if (cid as usize) < self.per_cell_color.len() { self.per_cell_color[cid as usize] } else { 0.0 };
                            let t = if self.color_range > 0.0 { ((val - self.color_min) / self.color_range).clamp(0.0, 1.0) } else { 0.5 };
                            coolwarm(t as f64)
                        }
                        ColorBy::CellId => {
                            // Distinct colors via golden-ratio hue rotation
                            let hue = (cid as f64 * 0.618033988749895) % 1.0;
                            hsv_to_rgb(hue, 0.7, 0.9)
                        }
                        ColorBy::None => {
                            let val = phi[idx].clamp(0.0, 1.0) as f64;
                            phi_colormap(val)
                        }
                    };
                    rgb[dst] = r; rgb[dst + 1] = g; rgb[dst + 2] = b;
                } else {
                    rgb[dst] = 31; rgb[dst + 1] = 31; rgb[dst + 2] = 38;
                }
            }
        }

        // Overlay cell labels if requested
        if label_cells && !centroids.is_empty() {
            // Build centroids vec with is_soft from per_cell_color (gamma mode)
            let mut cent_vec: Vec<(u32, f64, f64, bool)> = Vec::new();
            for (&cid, pos) in centroids {
                let is_soft = if self.color_by == ColorBy::Gamma && (cid as usize) < self.per_cell_color.len() {
                    // Same logic as checkpoint soft detection: compare to mode
                    let val = self.per_cell_color[cid as usize];
                    val < self.color_min + self.color_range * 0.5 && self.color_range > 0.0
                } else {
                    false
                };
                cent_vec.push((cid, pos[0] as f64, pos[1] as f64, is_soft));
            }
            // Wrap coordinates and draw labels using existing infrastructure
            let mut wrapped: Vec<(u32, i32, i32, bool)> = cent_vec.iter().map(|&(cid, cx, cy, soft)| {
                let wx = ((cx as i32 % nx as i32) + nx as i32) % nx as i32;
                let wy = ny as i32 - 1 - ((cy as i32 % ny as i32) + ny as i32) % ny as i32; // flip Y
                (cid, wx, wy, soft)
            }).collect();
            // Sort spatially for spatial index
            wrapped.sort_by(|a, b| {
                let row_a = a.2 / 80; let row_b = b.2 / 80;
                row_a.cmp(&row_b).then(a.1.cmp(&b.1))
            });
            let spatial_map: std::collections::HashMap<u32, usize> = wrapped.iter().enumerate()
                .map(|(i, &(cid, _, _, _))| (cid, i + 1)).collect();
            for &(cid, wx, wy, soft) in &wrapped {
                let sp = spatial_map.get(&cid).copied().unwrap_or(0);
                draw_label_with_spatial(&mut rgb, nx, ny, wx, wy,
                    &cid.to_string(), &sp.to_string(), "", soft);
            }
        }

        Ok(rgb)
    }
}

/// Coolwarm colormap: blue (low) → white (mid) → red (high)
fn coolwarm(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (59.0 + s * 196.0, 76.0 + s * 179.0, 192.0 + s * 63.0)
    } else {
        let s = (t - 0.5) * 2.0;
        (255.0, 255.0 - s * 190.0, 255.0 - s * 205.0)
    };
    (r as u8, g as u8, b as u8)
}

/// HSV to RGB conversion for per-cell_id coloring.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = (h % 1.0) * 6.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
        3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
    };
    (((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}

/// Watershed segmentation: flood-fill from centroids through low-gradient regions.
fn watershed(phi: &[f32], grad_mag: &[f32], nx: usize, ny: usize,
             centroids: &std::collections::HashMap<u32, [f32; 2]>) -> Vec<i32> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Copy, Clone)]
    struct Pixel { cost: f32, idx: usize, cell_id: i32 }
    impl PartialEq for Pixel { fn eq(&self, o: &Self) -> bool { self.idx == o.idx } }
    impl Eq for Pixel {}
    impl PartialOrd for Pixel { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
    impl Ord for Pixel { fn cmp(&self, o: &Self) -> Ordering { o.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal) } }

    let mut labels = vec![-1i32; nx * ny];
    let mut heap = BinaryHeap::new();

    for (&cid, pos) in centroids {
        let cx = ((pos[0] as isize % nx as isize) + nx as isize) as usize % nx;
        let cy = ((pos[1] as isize % ny as isize) + ny as isize) as usize % ny;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let sx = ((cx as i32 + dx) as usize) % nx;
                let sy = ((cy as i32 + dy) as usize) % ny;
                let idx = sy * nx + sx;
                if labels[idx] < 0 {
                    labels[idx] = cid as i32;
                    heap.push(Pixel { cost: grad_mag[idx], idx, cell_id: cid as i32 });
                }
            }
        }
    }

    while let Some(px) = heap.pop() {
        if labels[px.idx] != px.cell_id { continue; }
        let x = px.idx % nx;
        let y = px.idx / nx;
        let neighbors = [
            (if x + 1 < nx { x + 1 } else { 0 }, y),
            (if x > 0 { x - 1 } else { nx - 1 }, y),
            (x, if y + 1 < ny { y + 1 } else { 0 }),
            (x, if y > 0 { y - 1 } else { ny - 1 }),
        ];
        for (nx2, ny2) in neighbors {
            let nidx = ny2 * nx + nx2;
            if labels[nidx] >= 0 { continue; }
            if phi[nidx] < 0.1 { continue; }
            labels[nidx] = px.cell_id;
            heap.push(Pixel { cost: grad_mag[nidx], idx: nidx, cell_id: px.cell_id });
        }
    }
    labels
}

/// Erode a boolean mask by 1 pixel (4-connected, periodic).
fn erode_mask(src: &[bool], nx: usize, ny: usize) -> Vec<bool> {
    (0..ny).flat_map(|y| {
        (0..nx).map(move |x| {
            let idx = y * nx + x;
            if !src[idx] { return false; }
            let r = if x + 1 < nx { src[idx + 1] } else { src[y * nx] };
            let l = if x > 0 { src[idx - 1] } else { src[y * nx + nx - 1] };
            let u = if y + 1 < ny { src[(y + 1) * nx + x] } else { src[x] };
            let d = if y > 0 { src[(y - 1) * nx + x] } else { src[(ny - 1) * nx + x] };
            r && l && u && d
        })
    }).collect()
}

/// Render frames to PNGs (fallback when ffmpeg piping fails).
fn render_frames_to_png(selected: &[&PathBuf], frames_dir: &std::path::Path,
                        label_cells: bool, movie_ctx: Option<&MovieContext>) -> Result<()> {
    selected.par_iter().enumerate().for_each(|(idx, vtk_path)| {
        let result = if let Some(ctx) = movie_ctx {
            ctx.render_frame(vtk_path, idx, label_cells).map(|rgb| (rgb, ctx.nx, ctx.ny))
        } else {
            render_single_vtk(vtk_path, label_cells, None).map(|(data, nx, ny, _)| (data, nx, ny))
        };
        match result {
            Ok((img_data, nx, ny)) => {
                let frame_path = frames_dir.join(format!("frame_{:06}.png", idx));
                if let Err(e) = write_png(&frame_path, &img_data, nx, ny) {
                    eprintln!("Error writing {}: {}", frame_path.display(), e);
                }
            }
            Err(e) => eprintln!("Error rendering {}: {}", vtk_path.display(), e),
        }
    });
    eprintln!("Rendered {} frames to {}", selected.len(), frames_dir.display());
    Ok(())
}

/// Assemble movie from PNG frames using ffmpeg.
fn assemble_movie_from_png(frames_dir: &std::path::Path, fps: u32) -> Result<()> {
    let movie_path = frames_dir.join("movie.mp4");
    let ffmpeg_input = frames_dir.join("frame_%06d.png");
    eprintln!("Assembling movie from PNGs at {} fps...", fps);
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y", "-framerate", &fps.to_string(),
            "-i", &ffmpeg_input.to_string_lossy(),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            &movie_path.to_string_lossy().to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .status()?;
    if status.success() && movie_path.exists() {
        let size = std::fs::metadata(&movie_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("Movie saved: {} ({:.1} MB)", movie_path.display(), size as f64 / 1_048_576.0);
    } else {
        eprintln!("ERROR: ffmpeg failed (exit={})", status);
    }
    Ok(())
}
