//! Unified analysis CLI for cell-simulation trajectory data.
//!
//! Subcommands:
//!   `study`    — TOML-driven analysis pipeline (the canonical data path).
//!                Discovers runs, computes observables, aggregates, and
//!                renders figures / writes raw JSON. Add a new study by
//!                writing a TOML; no Rust changes needed.
//!   `snapshot` — render phase-field PNGs / movies from checkpoints or VTK.
//!   `check`    — validate trajectory + checkpoint integrity.
//!   `list`     — list available observables / aggregators / panels / templates.
//!
//! Library entry points live in `lib.rs`; this file is the bin only.

use cell_analyze::{analysis, vtk};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};

use analysis::io::load_trajectory;

#[derive(Parser)]
#[command(name = "cell_analyze")]
#[command(about = "TOML-driven analysis for cell-simulation trajectories.\n\
                   Reference TOMLs live in cpp/simulation/study/templates/.")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a TOML-defined study: discover runs, compute observables,
    /// aggregate, render figures. The TOML wires everything together.
    /// Reference templates live in cpp/simulation/study/templates/.
    /// `cell_analyze list` shows all available observables / aggregators / panels.
    /// Always writes <output_dir>/study_results.json next to any figures.
    Study {
        /// Path to the study TOML config file.
        config: PathBuf,
        /// Base directory containing simulation data.
        #[arg(long, short = 'd')]
        data_dir: PathBuf,
        /// Number of parallel threads (default: all available).
        #[arg(long)]
        threads: Option<usize>,
        /// Skip the per-run validation pre-pass. By default `study`
        /// validates each discovered run (same checks as
        /// `cell_analyze check`) before computing observables, and
        /// skips runs that fail. Pass this flag to bypass — useful
        /// when iterating on the analysis pipeline against runs you
        /// know are good.
        #[arg(long)]
        skip_validation: bool,
    },
    /// Render phase field snapshot(s) from checkpoint, VTK file, or directory of VTK frames.
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
        /// Skip the validation pre-pass on the input directory. By
        /// default `snapshot` validates the run that contains the input
        /// checkpoint/VTK before rendering (same checks as
        /// `cell_analyze check`). Pass this to bypass.
        #[arg(long)]
        skip_validation: bool,
    },
    /// List available observables, panels and aggregators (with descriptions).
    List {
        /// What to list. Default: all categories.
        #[arg(long, default_value = "all")]
        what: String,
    },
    /// Validate trajectory/checkpoint integrity AND run a sanity-check
    /// pass on observables (msd_palmieri, displacement, ln_perimeter)
    /// to catch NaN/Inf/empty outputs that would silently poison
    /// `study` runs. The observable pass runs by default; use `--fast`
    /// to skip it for a quick structural-only check. Always emits a
    /// full metadata report. Exits 0 on pass, 1 on any failure.
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
        /// Skip the observable pass (msd_palmieri, displacement,
        /// ln_perimeter). Saves a few seconds but cannot detect
        /// NaN/Inf outputs.
        #[arg(long)]
        fast: bool,
        /// Emit JSON report in addition to text
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Merge per-rank v8 checkpoint files into a single single-rank file.
    /// Pass the path to rank 0's checkpoint.bin (or the run directory);
    /// sibling rank{1..N-1}/checkpoint.bin files are discovered
    /// automatically. The merged file is a normal v8 checkpoint
    /// (num_ranks=1) that the simulator can resume with any `--gpus`.
    MergeCkpt {
        /// Path to rank-0 checkpoint.bin or its containing directory.
        input: PathBuf,
        /// Output path for the merged checkpoint. Defaults to
        /// `<input dir>/checkpoint_merged.bin`.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Find a pair of cells separated by approximately `distance` pixels.
    /// Useful for Phase 3A: pick two cells at a controlled separation, then
    /// resume with `--gamma <f>:nearest(x1,y1) --gamma <f>:nearest(x2,y2)`.
    /// Prints the chosen pair's ids, COMs, and actual separation. With
    /// `--format gamma-flags`, emits ready-to-paste `--gamma` arguments.
    ///
    /// IMPORTANT: the (x,y) coordinates printed are valid only for runs
    /// that resume from THIS checkpoint. If you re-equilibrate from t=0
    /// with a different `--seed`, the cells at those coordinates will be
    /// different (or absent). Workflow: equilibrate once → find-pair on
    /// that checkpoint → resume from the same checkpoint with the printed
    /// flags. Do NOT pass `--seed` on the resume.
    FindPair {
        /// Checkpoint file (single-rank or multi-rank; multi-rank auto-merges in memory).
        checkpoint: PathBuf,
        /// Target separation in pixels (periodic distance).
        #[arg(long)]
        distance: f64,
        /// Soft gamma value (only used with --format gamma-flags). Default 0.35.
        #[arg(long, default_value_t = 0.35)]
        soft_gamma: f64,
        /// Output format: "text" (human-readable) or "gamma-flags" (CLI snippet).
        #[arg(long, default_value = "text")]
        format: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Study {
            config,
            data_dir,
            threads,
            skip_validation,
        } => {
            if let Some(n) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
            }
            analysis::studies::run_study(&config, &data_dir, skip_validation)?;
        }
        Commands::Snapshot { input, output, width: _, label_cells, movie, skip, fps, color_by, shade_speed, speed_window, show_polarity, show_energy, emit_metadata, skip_validation } => {
            // Validation pre-pass: if the input is a checkpoint inside a
            // run directory, validate that directory first. We don't try
            // to validate VTK paths or movie directories — those have a
            // different shape and the snapshot path itself handles them.
            if !skip_validation && !movie {
                if let Some(run_dir) = input.parent() {
                    let traj = run_dir.join("trajectory.txt");
                    if traj.exists() {
                        use crate::analysis::precheck::{validate_run, print_report, Expectations};
                        let report = validate_run(run_dir, &Expectations::default(), true)?;
                        print_report(&report);
                        if !report.all_pass() {
                            eprintln!("\nERROR: validation failed for {}. \
                                       Re-run with --skip-validation to bypass.",
                                      run_dir.display());
                            std::process::exit(1);
                        }
                    }
                }
            }
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
        Commands::Check { dir, n_cells, expected_frames, t_start, t_end, fast, json } => {
            let exit_code = run_check(&dir, n_cells, expected_frames, t_start, t_end, !fast, json.as_deref())?;
            std::process::exit(exit_code);
        }
        Commands::List { what } => {
            list_what(&what);
        }
        Commands::MergeCkpt { input, output } => {
            let out = output.unwrap_or_else(|| {
                let dir = if input.is_dir() {
                    input.clone()
                } else {
                    input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
                };
                dir.join("checkpoint_merged.bin")
            });
            cell_analyze::analysis::merge_checkpoint::merge_checkpoints(&input, &out)?;
        }
        Commands::FindPair { checkpoint, distance, soft_gamma, format } => {
            run_find_pair(&checkpoint, distance, soft_gamma, &format)?;
        }
    }

    Ok(())
}

/// Find the cell pair whose periodic distance is closest to `target_d`.
/// Emits either human-readable text or `--gamma` CLI flags for the resume call.
fn run_find_pair(checkpoint: &Path, target_d: f64, soft_gamma: f64, format: &str) -> Result<()> {
    use cell_analyze::analysis::checkpoint::load_checkpoint;
    let ckpt = load_checkpoint(checkpoint)
        .with_context(|| format!("loading checkpoint {}", checkpoint.display()))?;
    let lx = ckpt.params.nx as f64;
    let ly = ckpt.params.ny as f64;
    let cells = &ckpt.cells;
    let n = cells.len();
    if n < 2 {
        anyhow::bail!("checkpoint has {} cells; need at least 2", n);
    }
    let wrap = |dx: f64, l: f64| {
        let m = dx.rem_euclid(l);
        if m > 0.5 * l { m - l } else { m }
    };
    // Find the (i, j) pair whose periodic distance is closest to target_d.
    // O(N²) — fine up to N≈10^4. The cell COM in the checkpoint is the
    // wrapped-to-domain value (centroid: (f32, f32) field).
    let mut best_i: usize = 0;
    let mut best_j: usize = 1;
    let mut best_diff = f64::INFINITY;
    let mut best_d = 0.0f64;
    for i in 0..n {
        let (xi, yi) = (cells[i].centroid.0 as f64, cells[i].centroid.1 as f64);
        for j in (i + 1)..n {
            let (xj, yj) = (cells[j].centroid.0 as f64, cells[j].centroid.1 as f64);
            let dx = wrap(xj - xi, lx);
            let dy = wrap(yj - yi, ly);
            let d = (dx * dx + dy * dy).sqrt();
            let diff = (d - target_d).abs();
            if diff < best_diff {
                best_diff = diff;
                best_i = i;
                best_j = j;
                best_d = d;
            }
        }
    }
    let ci = &cells[best_i];
    let cj = &cells[best_j];
    match format {
        "gamma-flags" => {
            // Emit ready-to-paste CLI fragment. nearest() resolves by COM
            // in the resumed sim, so we pass the saved centroid here.
            println!(
                "--gamma {g}:nearest({x1:.3},{y1:.3}) --gamma {g}:nearest({x2:.3},{y2:.3})",
                g = soft_gamma,
                x1 = ci.centroid.0, y1 = ci.centroid.1,
                x2 = cj.centroid.0, y2 = cj.centroid.1,
            );
        }
        _ => {
            // Default: human-readable.
            println!("found pair at target d = {:.3} px (domain {}×{})", target_d, lx as i32, ly as i32);
            println!("  cell {} at ({:.3}, {:.3})", ci.id, ci.centroid.0, ci.centroid.1);
            println!("  cell {} at ({:.3}, {:.3})", cj.id, cj.centroid.0, cj.centroid.1);
            println!("  actual periodic distance = {:.3} px (diff {:.3} px from target)", best_d, best_diff);
            println!("# resume FROM THIS CHECKPOINT only. Do not change --seed.");
            println!("# example: resume_simulation -c {} --gamma {:.2}:nearest({:.3},{:.3}) --gamma {:.2}:nearest({:.3},{:.3})",
                     checkpoint.display(), soft_gamma, ci.centroid.0, ci.centroid.1,
                     soft_gamma, cj.centroid.0, cj.centroid.1);
        }
    }
    Ok(())
}

/// Print available observables / panels / aggregators / templates.
fn list_what(what: &str) {
    let want = what.to_ascii_lowercase();
    let all = want == "all";
    if all || want == "observables" {
        println!("Observables (driven by registry — keep in sync with src/analysis/observables/mod.rs):");
        println!("  --- core (Phase 1 / Palmieri) ---");
        println!("  msd                        Ensemble MSD(Δt) + cell-0 MSD");
        println!("  msd_palmieri               Dense 0..8τ MSD/Δt + D_eff(8τ) read-off");
        println!("  ln_perimeter               L_n(t) shape index for tagged cell");
        println!("  displacement_velocities    (vx, vy) + speeds for tagged cell");
        println!("  velocity_distribution      P(v_x), σ, kurtosis (cell 0 + population)");
        println!("  bursts                     Speed-burst events |v| > μ + k·σ");
        println!("  --- MSD-derived ---");
        println!("  diffusion                  D_eff from long-time MSD slope + R²");
        println!("  msd_log_slope              Δ(t) = d log MSD / d log t");
        println!("  cage_length                L_c from min Δ(t) plateau near τ");
        println!("  per_cell_diffusion         Per-cell D_i + mean/std/CV");
        println!("  displacement               Net Δr first→last frame summary");
        println!("  --- glass / jamming ---");
        println!("  alpha2                     Non-Gaussian parameter α₂(Δt)");
        println!("  overlap_chi4               Self-overlap Q(t) + dynamic susceptibility χ₄");
        println!("  structure_factor           S(q) angularly averaged + q*");
        println!("  fs_qstar                   Self-intermediate scattering F_s(q*, t) + τ_α");
        println!("  van_hove                   G_s(Δx, t) histogram at {{0.1τ, τ, 3τ}}");
        println!("  --- spatial / correlation ---");
        println!("  spatial_correlation        C(r) of mobility + ξ at 1/e crossing");
        println!("  velocity_autocorrelation   C_v(τ) + τ_c");
        println!("  va_mobility_correlation    Pearson(inherent v_A, time-avg speed)");
        println!("  --- geometry / order ---");
        println!("  shape_index                p_eff = L_n × 2√π (vertex-model order param)");
        println!("  hexatic_order              ψ₆ + g₆(r)");
        println!("  voronoi_shape              q = P/√A from Voronoi polygons");
        println!("  --- polarity / energy ---");
        println!("  polarity_tau               Persistence τ from polarity autocorrelation");
        println!("  kinetic_energy             ½⟨v²⟩(t) per cell + total");
        println!();
    }
    if all || want == "aggregators" {
        println!("Aggregators (study TOML `[[aggregate]]` op):");
        println!("  groupby      group runs sharing variable values");
        println!("  mean_stderr  mean±stderr of metrics across replicates");
        println!("  sweep        order group summaries along a numeric axis");
        println!("  pair_ratio   ratio + propagated error between two groups");
        println!("  pair_runs    pair raw runs (e.g. soft vs ctrl) for diagnostic figures");
        println!("  single_run   pick exactly one run via filter");
        println!("  overlay      bundle N runs for layered plots");
        println!();
    }
    if all || want == "metrics" {
        println!("Scalar metric IDs (used in `mean_stderr.metrics`, `metric_vs_x.metric`):");
        println!("  msd_lag1            cell-0 MSD at Δt = first lag (from msd)");
        println!("  msd_pop_lag1        ensemble MSD at Δt = first lag (from msd)");
        println!("  deff_palmieri       cell-0 D_eff(8τ) (from msd_palmieri)");
        println!("  deff_pop_palmieri   population D_eff(8τ) (from msd_palmieri)");
        println!("  (extend in src/analysis/studies.rs::metric_registry)");
        println!();
    }
    if all || want == "panels" {
        println!("Panels (figure TOML `[[figure]].panels.type/subtype`):");
        println!("  metric_vs_x                       sweep panel: y vs x with stderr bars");
        println!("  pair / speed_bursts | gvi |        soft-vs-ctrl style 6/8-panel grid");
        println!("       ln_timeseries | ln_histogram |");
        println!("       msd_t | deff_bar | summary");
        println!("  single / msd | gvi |               one-run plots");
        println!("         ln_timeseries | speed_bursts");
        println!("  overlay / msd | gvi | ln_timeseries  N-run colored overlay");
        println!();
    }
    if all || want == "templates" {
        println!("Reference TOMLs in cpp/simulation/study/templates/:");
        println!("  single_run.toml          one run → 4-panel single-run grid");
        println!("  pair_compare.toml        soft vs ctrl → 8-panel diagnostic");
        println!("  overlay_cond_sweep.toml  N runs at different conditions overlaid");
        println!("  fss.toml                 sweep over N → metric_vs_x");
        println!("  phase3a_pairwise.toml    sweep over separation → metric_vs_x");
    }
}



/// Thin wrapper: delegates the full validation flow to
/// `analysis::precheck::validate_run`. The same code is invoked by
/// `study` and `snapshot` as a pre-pass — see the precheck module for
/// the full check catalog.
fn run_check(
    dir: &std::path::Path,
    expected_n_cells: Option<usize>,
    expected_frames: Option<usize>,
    expected_t_start: Option<f64>,
    expected_t_end: Option<f64>,
    with_observables: bool,
    json_out: Option<&std::path::Path>,
) -> Result<i32> {
    use crate::analysis::precheck::{validate_run, print_report, Expectations};
    let exp = Expectations {
        n_cells: expected_n_cells,
        frames: expected_frames,
        t_start: expected_t_start,
        t_end: expected_t_end,
    };
    let report = validate_run(dir, &exp, with_observables)?;
    print_report(&report);

    if let Some(p) = json_out {
        let json = serde_json::to_string_pretty(&serde_json::json!({
            "dir": report.dir.display().to_string(),
            "all_pass": report.all_pass(),
            "passed": report.passed_count(),
            "total": report.total(),
            "checks": report.findings.iter().map(|r| serde_json::json!({
                "name": r.name, "passed": r.passed, "detail": r.detail
            })).collect::<Vec<_>>(),
        }))?;
        std::fs::write(p, json)?;
    }

    Ok(if report.all_pass() { 0 } else { 1 })
}


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
