---
applyTo: "cpp/simulation/cluster/**"
---

# Cluster Operations - Copilot Instructions

> **When to consult this file:** You are working with simulation jobs on Alliance Canada HPC clusters. This file covers domain knowledge the MCP tool doesn't encode: physics context for equilibration, I/O budgets, MPI fallback, and analysis job submission. For simulation physics, CLI options, or local builds, see [cell-simulation.instructions.md](cell-simulation.instructions.md). For analyzing output data, see [postprocessing.instructions.md](postprocessing.instructions.md).

## Related Instructions

| Task | Instruction File |
|------|-----------------|
| **Building & running simulations locally** | [cell-simulation.instructions.md](cell-simulation.instructions.md) |
| **Post-processing output (visualization, MSD)** | [postprocessing.instructions.md](postprocessing.instructions.md) |
| **Developing analysis tools for cluster** | [cluster-postprocessing.instructions.md](cluster-postprocessing.instructions.md) |

---

## Primary Interface: `compute-canad` MCP Tool

**All cluster operations go through the `compute-canad` MCP tool.** The tool's schema is self-documenting — parameter descriptions, defaults, and validation are built into each tool. See `tools/compute_canada_mcp/DESIGN.md` for the full reference.

### Tool → Task Mapping

| Task | MCP Tool | NOT this |
|------|----------|----------|
| Check/establish SSH | `connect` | Manual `wsl ssh` commands |
| Build binary on cluster | `sync_and_build` | `build_on_cluster.sh`, manual tar/scp/make |
| Verify binary exists | `check_binary` | Manual `ls`, `ldd` commands |
| Submit fresh simulation | `start_simulation` | `submit_job.sh` |
| Resume from checkpoint | `resume_simulation` | `submit_job.sh --continue` |
| Check job queue | `list_jobs` | `squeue` via `run_command` |
| Check GPU availability | `check_queues` | `sinfo` via `run_command` |
| Monitor run progress | `check_progress` | `find`/`tail` via `run_command` |
| Read job logs | `get_job_logs` | `cat` via `run_command` |
| Cancel jobs | `cancel_jobs` | `scancel` via `run_command` |
| Find existing data | `discover` | `find`/`ls` via `run_command` |
| Download results | `download_results` | Manual `scp` |
| Estimate cost | `estimate_cost` | Manual calculation |
| Benchmark cluster perf | `benchmark_cluster` | Manual `sbatch` timing |
| Quick inspection | `run_command` | Shell scripts |
| Admin/diagnostic shell | `inspect_cluster` | N/A |
| Tag existing directory | `create_marker` | N/A |

---

### ⚠️ CRITICAL: Auto-Computed SLURM Resources — Do NOT Manually Override

**`start_simulation` and `resume_simulation` automatically compute ALL scheduling parameters.** Do NOT manually specify any of the following:

| Parameter | Auto-Selection Method | Why manual is worse |
|-----------|----------------------|---------------------|
| **account** | Tests ALL GPU accounts with `sbatch --test-only`, picks earliest start | You can't know scheduler state without probing it |
| **walltime** | Computed from `~/cell_sim_calibration.json` benchmark data × safety factor | Hand-picked walltimes are either too short (timeouts) or too long (scheduling delays) |
| **num_chains** | Computed from calibration rate × t_end | Over-chains waste scheduler slots; under-chains cause incomplete runs |
| **gres** | Matched to cluster GPU type from config | Already optimal per cluster |
| **memory** | Scaled from n_cells | Already optimal per system size |

**The only parameters you need to provide are physics parameters:**
- `n_cells`, `radius`, `confluence`, `t_end` (for start)
- `checkpoints`, `t_end`, `n_cells` hint (for resume)
- Physics overrides: `v_A`, `gamma`, `kappa`, `mu`, `xi`, `adhesion_J`, etc.
- I/O: `trajectory_samples`, `save_interval`
- `output_dir`, `study_tag`, `seed`

**Calibration:** Run `benchmark_cluster(cluster='nibi', action='submit')` then `benchmark_cluster(action='collect')` to measure actual performance. This writes `~/cell_sim_calibration.json` on the cluster, which all submission tools read automatically. If calibration is missing, the tools fall back to hardcoded estimates from `gpu_decision_rules`.

---

### ⚠️ `run_command` is Restricted

`run_command` **blocks** the following and returns an error:
- `sbatch`, `submit_job.sh` → use `start_simulation` / `resume_simulation`
- `scancel` → use `cancel_jobs`
- `squeue` → use `list_jobs`
- `sinfo` → use `check_queues`
- `cell_sim` → use `start_simulation` / `resume_simulation`
- Multi-line commands, heredocs, shell loops, commands over 500 characters

`run_command` is for **quick single-line inspection only**: sacctmgr, module list, df, stat, wc, ls.

For system administration tasks requiring unrestricted shell access, use `inspect_cluster` instead. It requires a `justification` parameter explaining why no dedicated tool covers the task.

### Auto-Reconnect Protocol

When `connect` returns `connected: false`:

1. **Immediately** call `connect` again with `reconnect: true`.
2. Tell the user: *"Reconnecting — please approve the Duo push on your phone."*
3. The tool auto-sends "1" (Duo Push) and waits up to 60 s for approval.
4. If it returns `connected: true`, proceed with the original task.
5. If it fails, tell the user to check their phone or try again.

**Do not** ask the user to run manual SSH commands — `reconnect` handles everything.

---

## ⚠️ CRITICAL: Never Run Compute on Login Nodes

**DO NOT run computationally intensive jobs on login nodes** — not even via `run_command`.

This includes:
- Processing large trajectory files
- Running simulations
- Heavy I/O operations
- Any job expected to take more than a few seconds

Login nodes are shared. Use them only for: file management, job submission, quick status checks. Submit compute work via SLURM (use `start_simulation` / `resume_simulation`, or `run_command` with `sbatch` for analysis jobs).

---

## Equilibration: Physics Context

**Equilibration = v_A=0 (no motility)** — cells relax into a static configuration from random initial conditions.

### Why Equilibrate?

Production runs should start from an equilibrated state:
1. Ensures fair comparison between different v_A values
2. Removes transient effects from random initialization
3. Allows studying the effect of "suddenly turning on" motility

### Packing Fraction & Domain Size

**⚠️ NEVER hardcode domain sizes. NEVER copy domain sizes from tables or examples.** The ONLY source of truth is the target packing fraction $\rho$ and the formula:

$$L = \lceil\sqrt{N \pi R^2 / \rho}\rceil$$

When using MCP tools, pass `confluence` as a parameter — the tool computes $L$ automatically. When running manually, always compute $L$ at submission time from $N$, $R$, and $\rho$.

The target confluence depends on the study:
- **Palmieri validation:** $\rho \in \{0.85, 0.90\}$
- **Adhesion / Griffiths:** $\rho = 0.89$
- **Palmieri extension:** $\rho \in \{0.70 \ldots 1.00\}$ (see study-specific instructions)

### Equilibration Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `v_A` | 0 | No self-propulsion |
| `t_end` | 80,000 | Sufficient for relaxation |

Physics parameters ($\gamma$, $\kappa$, $\mu$, $\xi$) depend on the study. Check the study-specific instruction file:
- **Palmieri extension:** Uses binary defaults (no overrides needed)
- **Adhesion study / Griffiths study:** Uses Bresler parameters — set `gamma=3.75`, `mu=0.5`, `xi=1000` via the MCP tool's native parameters (not `extra_cli_flags`). See the study-specific instruction files for details.

See [cell-simulation.instructions.md → Parameter Sets](cell-simulation.instructions.md) for the full comparison.

### Equilibration I/O

Equilibration needs **minimal output**:
- ~10 VTK frames to visually confirm convergence
- No trajectory or tracking data (`trajectory_samples=0`)
- Frequent checkpoints for chain reliability

#### ⚠️ Per-Cell v_A Regeneration on Resume

> See [cell-simulation.instructions.md → v_A Regeneration on Resume](cell-simulation.instructions.md) for full details.

When resuming from an equilibration checkpoint (where all v_A=0), you **must** specify `v_A` (and optionally `v_A_sigma`) to regenerate per-cell values. Without this, cells won't move. The MCP tool's `resume_simulation` passes these as override flags automatically.

---

## 3D Simulations

### 3D vs 2D Defaults

| Parameter | 2D | 3D |
|-----------|----|----|
| Cells | 288 | 100 (configurable) |
| Radius | 49 | 36 |
| Domain | Computed from confluence | Computed: L = ∛(N × (4/3)πR³ / φ) |
| Memory | 8–16 GB | 64 GB |
| Default t_end | 880,000 | 4,000 |
| Equilibration t_end | 80,000 | 60,000 |

### 3D Domain Reference (85% confluence)

| Cells | Domain (L³) |
|-------|-------------|
| 100 | 284³ |
| 200 | 358³ |
| 400 | 451³ |

---

## Multi-GPU (NCCL Domain Decomposition)

For large-N runs, `start_simulation` and `resume_simulation` accept a `gpus` parameter that enables NCCL-based 1D-slab domain decomposition across multiple GPUs on a single node.

### When to use it

| N_cells | Recommended `gpus` |
|---------|---:|
| < 3200 | 1 (default) — overhead dominates speedup |
| 3200 – 4000 | 2 |
| ≥ 4000 | 4 |

The MCP tool emits a warning whenever `n_cells ≥ 3200` is submitted with `gpus=1`. Resubmit with the `gpus` parameter set to follow the warning's recommendation.

### What `gpus=N` does

- Swaps to the multi-GPU build (`~/cell_simulation/build_mg/bin/cell_sim`)
- Requests `<gpu_type>:N` in the SLURM `--gpus`/`--gres` directive
- Loads the `nccl/2.29.7` module and exports `NCCL_P2P_LEVEL=NVL`
- Appends `--gpus N` to the cell_sim command line
- Scales `--mem` linearly by N (host-side scratch is per-process; **GPU VRAM is independent and per-rank**, typically fine on A100/H100)

### Measured performance (Narval A100×4, N=4608, t=1500)

| GPUs | ms/step | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 4.98 | 1.00× | — |
| 2 | 2.92 | 1.71× | 85% |
| 4 | 1.50 | 3.33× | **83%** |

### Checkpoint compatibility

- Multi-GPU runs write a **per-rank checkpoint set**: rank 0 in `<outdir>/checkpoint.bin`, rank K in `<outdir>/rankK/checkpoint.bin`. The format is v8 with a self-describing `(num_ranks, rank_id, num_cells_global)` header.
- `resume_simulation` requires `gpus` to **match the saved checkpoint's `num_ranks`** (the C++ loader bails with a clear error otherwise).
- To change `--gpus` between runs (e.g. resume a G=4 checkpoint as G=1 for analysis), first consolidate the per-rank files into a single-rank v8 checkpoint:

```bash
cell_analyze merge-ckpt <outdir>/checkpoint.bin -o <outdir>/merged.bin
```

The merged file is a normal v8 checkpoint that resumes with any `--gpus` value.

### Analysis tools

`cell_analyze` (snapshot, check, study, MSD, etc.) **auto-detects** multi-rank runs and unions all sibling `rankN/trajectory.txt` files transparently. No manual merging is needed for trajectory analysis. For checkpoint rendering, run `merge-ckpt` first.

### What `gpus` doesn't do

- **Doesn't validate** against your saved checkpoint at submission time. If you set `gpus=4` to resume a G=1 checkpoint, the cell_sim binary fails at startup with a clear error — but the SLURM job still queues. Run `dry_run=true` first if you're unsure.
- **Not auto-selected.** You opt in explicitly so the build/module costs are visible.
- **Requires `~/cell_simulation/build_mg/`** to exist on the target cluster. Build it with `sync_and_build` (the multi-GPU CMake target is set up under `build_mg/`).

---

## MPI/CPU Version (Fallback)

The MCP tool does not manage MPI builds or submissions. When GPU queues are heavily congested, the MPI/CPU version is a manual alternative.

### Key Facts

- Source: `~/cell_simulation_mpi/`, binary: `~/cell_simulation_mpi/build/cell_sim_mpi`
- Build: `module load cmake/3.27 gcc/12.3 openmpi/4.1 && cmake .. -DCMAKE_CXX_COMPILER=mpicxx && make -j8`
- **Checkpoint cross-compatible** with CUDA version (identical format v4) — can start on GPU, continue on CPU, or vice versa
- ~4× slower than GPU for small systems; use when queue wait > 24 hours

### MPI Job Template

```bash
#!/bin/bash
#SBATCH --account=<your-account>   # Use list_jobs or check_queues to find the correct account
#SBATCH --job-name=cell_sim_mpi
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --mem=32G --time=12:00:00
#SBATCH --output=%x_%j.out --error=%x_%j.err

module load gcc/12.3 openmpi/4.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Domain size MUST be computed at submission time: L = ceil(sqrt(N * pi * R^2 / rho))
# Example: python3 -c "import math; print(math.ceil(math.sqrt(72 * math.pi * 49**2 / 0.90)))"
L=$(python3 -c "import math; print(math.ceil(math.sqrt(72 * math.pi * 49**2 / 0.90)))")
~/cell_simulation_mpi/build/cell_sim_mpi -n 72 -N $L -r 49 -t 1000 --dt 0.01 \
    -o /scratch/ssilber/cell_sim_results_mpi/run_001
```

---

## ⚠️ I/O Intervals & Data Budget

**Trajectory files dominate storage.** Saving too frequently creates multi-TB datasets that exceed scratch quota. Always compute the data budget before running production.

### Current Production I/O Settings

| Parameter | Value (steps) | Time between saves | Purpose |
|-----------|--------------|--------------------|---------|
| `save_interval` | **0** | — | VTK disabled — **never save VTK in production** |
| `trajectory_samples` | **2000** | ~360 time units | ~2000 data points over production |
| `checkpoint_interval` | **50000–75000** | ~1000–1500 t.u. | Recovery checkpoints |

### Data Budget (288c, t=80k→880k = 720k production time)

| trajectory_samples | Interval (steps) | File size/run | Total (1000 runs) |
|----|----|----|---|
| 36,000 | 1000 | ~2.8 GB | **~2.8 TB** ❌ |
| 2,000 | 18000 | ~47 MB | **~47 GB** ✓ |
| 1,000 | 36000 | ~24 MB | ~24 GB ✓ |
| 500 | 72000 | ~12 MB | ~12 GB ✓ |

Formula: `saves = production_time / (interval × dt)`, `lines = saves × N_cells`, `bytes ≈ lines × 86`

### Guidelines

- **Never use VTK saves in production** — each frame for 288c on 1600² is ~10 MB
- **Target 500–2000 trajectory data points.** 2000 points gives excellent resolution for MSD at all timescales
- **Rule of thumb**: `trajectory_interval = total_steps / desired_data_points`
- **Scratch quota is 1 TB.** Budget trajectory + checkpoints + tracking within this
- For dense short-time data (ballistic regime), use a separate short diagnostic run

### Simulation Parameters (Paper Target)

> **Note:** These are the general jamming study targets. Each study has its own parameter requirements — see the study-specific instruction files. Binary defaults (dt, tau, kappa, etc.) do not need to be specified — run `cell_sim -h` for current values.

| Parameter | Value |
|-----------|-------|
| `T_END` | 880,000 |
| `v_A` values | 0.004–0.013 (10 points, step 0.001) |
| Cells | 288 (standard), 1152 / 4608 (finite-size) |
| Confluence | 89% |
| Replicates | 100 per condition (jamming), 3 per condition (Griffiths) |

---

## Analysis Jobs (Post-Processing on Cluster)

The MCP tool does not handle analysis job submission. For heavy post-processing, submit SLURM jobs manually.

### Available Analysis Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `msd_calculator.c` + `msd_job.sh` | MSD/diffusion analysis | `sbatch msd_job.sh 96000` |
| `energy_analyzer.c` + `energy_job.sh` | Kinetic energy analysis | `sbatch energy_job.sh` |
| `plot_jamming_transition.py` | Plot D vs v_A | Python script |

### Template for New Analysis Jobs

```bash
#!/bin/bash
#SBATCH --account=<your-account>   # Varies by cluster — check cluster_config.json or list_jobs
#SBATCH --job-name=my_analysis
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=8G --time=02:00:00
#SBATCH --output=%x_%j.out --error=%x_%j.err

module load gcc/12.3
./my_analysis_program /scratch/ssilber/data output.txt
```

**Workflow:** Test locally with small data first → upload via SCP (no MCP upload tool exists yet) → submit analysis job via `run_command` with `sbatch`.

> **Note on `run_command` for analysis jobs:** Submitting non-simulation SLURM jobs (C/Python analysis tools) via `run_command` + `sbatch` is acceptable — the MCP submission tools (`start_simulation`, `resume_simulation`) are only for the cell_sim binary. Do NOT use `run_command` for simulation submission, job monitoring, or file inspection — use the dedicated MCP tools for those.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `connect` returns false | Call `connect` with `reconnect: true`; approve Duo push |
| Job stuck in PENDING | Use `check_queues` to find less-loaded cluster |
| Job failed immediately | Use `get_job_logs` to read stderr |
| Out of disk space | Use `disk_usage` MCP tool (or `diskusage_report` via `run_command` for full Alliance report) |
| Checkpoint version mismatch | Binary is stale — use `sync_and_build` |
| Binary runs on CPU | Use `check_binary` to verify CUDA linkage; rebuild with `sync_and_build` |
| Mid-chain failure | Use `check_progress` to find last good time; resubmit with `resume_simulation` |
