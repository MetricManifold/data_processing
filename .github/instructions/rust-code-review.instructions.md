---
applyTo: "rust/vtk_viewer/src/**"
---

# Rust cell_analyze — Code Review Instructions

## Purpose of This Document

This file provides context for conducting a code review of the `cell_analyze` Rust binary. It describes the design intent, expected behavior, and research context so a reviewer can evaluate whether the implementation serves its purpose correctly.

---

## 1. What This Tool Is

`cell_analyze` is the primary data analysis engine for a physics research project studying cell motility in phase-field simulations. It reads simulation trajectory files, computes physics observables, and produces structured output (JSON + SVG plots) that researchers use to draw conclusions about whether cancer cells move faster than normal cells in a tissue monolayer.

**Correctness of this tool directly determines the validity of published research results.** Any bug in observable computation, data loading, pairing logic, or aggregation propagates into the paper's figures and conclusions.

---

## 2. Design Intent

### 2.1 Single binary, multiple modes

The tool should work as ONE binary with subcommands, not multiple scripts. This ensures all analysis uses the same loading, unwrapping, and computation code. There should be no situation where different analysis paths produce different results from the same input.

### 2.2 TOML-driven studies

A study is defined by a TOML configuration file, not by code. The TOML specifies:
- **Discovery**: how to find trajectory files via path patterns with named captures
- **Observables**: what physics to compute per trajectory
- **Analysis**: how to group runs, pair conditions (e.g., soft vs ctrl), and what metrics to extract
- **Visualization**: what plots and multi-panel figures to generate

New experiments should require only a new TOML file, not code changes.

### 2.3 Source of truth

The Rust tool is the authoritative source for all quantitative results. No Python scripts, notebooks, or ad-hoc analysis should be trusted over Rust output. If Python produces different numbers, Rust is correct and Python needs fixing.

### 2.4 Paired comparison

The core analysis pattern is comparing a "tagged cell" (cell 0, the cancer cell) across two conditions: soft (γ=0.35) vs ctrl (γ=1.0), from the same equilibration checkpoint. The ratio `D_eff(soft) / D_eff(ctrl)` is the primary result. This must be computed as a paired comparison (same seed, different condition), NOT as an unpaired comparison (cell 0 vs population mean within one run).

### 2.5 D_eff extraction

D_eff is extracted from MSD(Δt)/(4Δt) evaluated at lag = 8τ (where τ=10000 is the persistence time). This follows the Palmieri et al. (2015) convention. The MSD is time-averaged using a sliding window over all starting times. Using a different lag or averaging method will produce different (potentially incorrect) results.

---

## 3. Expected Usage Patterns

### 3.1 Study pipeline (primary use case)
```bash
cell_analyze study fss.toml -d /data/fss_reps -o results.json --plot-dir plots/
```
This discovers all runs matching the TOML pattern, analyzes each, pairs soft/ctrl by (N, ρ), averages over seeds, computes ratios with propagated errors, and generates SVG plots.

### 3.2 Diagnostic comparison
```bash
cell_analyze compare /data/soft/run_01 /data/ctrl/run_01 -o diagnostic.svg
```
This produces a side-by-side multi-panel figure showing L_n time series, displacement speed, trajectory, and MSD/4t for the tagged cell in both runs.

### 3.3 Remote execution on cluster
```bash
# On HPC login node:
nohup ~/bin/cell_analyze study ~/fss.toml -d /scratch/data \
  --subsample 100 --threads 1 -o /tmp/results.json &
```
The binary deploys as a static Linux ELF. Subsampling reduces I/O for high-resolution trajectories. Results are a small JSON file that gets downloaded.

### 3.4 Agent-driven research workflow
An AI agent should be able to:
1. Submit simulation jobs via MCP tools
2. Run `cell_analyze study` on the cluster to get JSON results
3. Interpret the JSON (D_eff ratios, error bars, trends vs N)
4. Generate and view diagnostic comparisons for suspicious results
5. Update the study TOML to add new metrics or plots
6. Draw research conclusions from the data

---

## 4. Code Review Guidelines

### 4.1 Correctness
- **Physics correctness**: Are MSD, D_eff, L_n, velocity distribution computed correctly? Check formulas against Palmieri et al. (2015).
- **MSD at 8τ**: Verify that `per_cell_diffusion` evaluates MSD/(4Δt) at exactly lag = 8τ, not averaged over a range.
- **Periodic boundary handling**: Does `unwrap_trajectory` correctly detect and unwrap jumps > L/2?
- **Chain overlap handling**: Does `load_trajectory` correctly detect backward time jumps and keep the longest monotonic segment?
- **Subsampling**: Does `--subsample N` correctly keep every Nth frame (counted by cell-0 appearances)?
- **Paired comparison**: Does the study pipeline correctly pair soft/ctrl runs by (N, ρ) and compute ratios, not within-run comparisons?
- **Error propagation**: Is stderr computed as σ/√n (sample std / √n_seeds)? Is ratio error propagated as |a/b| × √((σa/a)² + (σb/b)²)?

### 4.2 Data integrity
- Is trajectory data loaded without corruption?
- Are positions properly unwrapped before MSD computation?
- Do chain-overlap segments get correctly identified and filtered?
- Does the subsampling maintain frame alignment (not interleaving chains)?
- Are all cells in a frame loaded atomically (no partial frames)?

### 4.3 Design quality
- **Separation of concerns**: Is I/O separate from computation separate from output?
- **Single responsibility**: Does each function do one thing?
- **DRY**: Is there duplicated logic between `run`, `batch`, `study`, and `compare`?
- **Extensibility**: Can new observables be added without modifying existing code?
- **TOML completeness**: Can all analysis that matters be expressed in the TOML, or are there hardcoded behaviors that should be configurable?
- **Compare should be TOML-driven**: The `compare` subcommand currently bypasses the TOML system. Should it be integrated?

### 4.4 Robustness
- What happens with empty trajectories, single-frame files, files with only headers?
- What happens if soft and ctrl have different numbers of frames?
- What happens if a run is still being written (active simulation)?
- Are there panics that should be Results?
- Is memory usage bounded (can it handle 200k-frame × 400-cell files)?

### 4.5 Research fitness
- Does the JSON output contain everything needed to reproduce a figure?
- Can an agent interpret the JSON without additional context?
- Are the SVG plots useful for quick assessment, or do they obscure issues?
- Are the right observables being computed for the research questions?
- Is the 8τ D_eff extraction the best choice, or should other methods be available?

### 4.6 Alternative approaches
- Would a different MSD computation (e.g., FFT-based) be faster or more accurate?
- Should the study pipeline retain per-run trajectory data for diagnostic figures?
- Should `compare` be a TOML section rather than a separate subcommand?
- Is the current plotting sufficient, or should the tool output data for an external plotter?
- Should metrics be composable expressions rather than a fixed set of named extractors?
