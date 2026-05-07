# Rust analysis pipeline — architecture and migration plan

> Living document. Status of each phase is tracked at the bottom.
> See conversation thread on the architecture rationale (May 7 2026).

## Goal

Replace the monolithic `analysis/study.rs` (~2.9k LOC) with a small set of
composable, typed primitives that the study TOML wires together.

The user-facing contract becomes: **adding a new study type is just a TOML.**
Adding a new observable, panel, or aggregator is a small registered module.

## Layered design

```
+-------------------------------------------------------------------+
|  Layer 6: Studies (declarative TOML, no per-study Rust)           |
|    discover -> analyze -> aggregate -> render                     |
+-------------------------------------------------------------------+
|  Layer 5: Panels (one file per panel, registered)                 |
|    Panel trait, typed Data input, plotters backend                |
+-------------------------------------------------------------------+
|  Layer 4: Aggregators (compose: groupby, mean_stderr, sweep, ...) |
|    Aggregator trait, named ops the TOML invokes                   |
+-------------------------------------------------------------------+
|  Layer 3: Per-run analysis (analyze_run -> RunAnalysis)           |
|    Loads data once, runs requested observables, returns bag       |
+-------------------------------------------------------------------+
|  Layer 2: Observables (typed registry, pure functions)            |
|    Observable trait, ObservableBag, one module per observable     |
+-------------------------------------------------------------------+
|  Layer 1: Discovery (filesystem -> RunSpecs with typed variables) |
|    DiscoveryRule, VarSpec, discover()                             |
+-------------------------------------------------------------------+
|  Layer 0: I/O (UNCHANGED)                                         |
|    analysis::io  /  analysis::checkpoint  /  analysis::metadata   |
+-------------------------------------------------------------------+
```

## Module map (target state)

```
rust/vtk_viewer/src/
├── analysis/                 (legacy I/O, kept; rest gets emptied over time)
│   ├── io.rs                 KEEP
│   ├── checkpoint.rs         KEEP
│   ├── metadata.rs           KEEP
│   ├── batch.rs              KEEP
│   └── output.rs             KEEP (legacy RunResult JSON, slowly deprecated)
│
└── pipeline/                 (NEW, the v2 architecture)
    ├── mod.rs                pipeline executor + public API
    ├── observable.rs         Observable trait, ObservableBag, ObservableId
    ├── observables/          one module per observable (typed registry)
    │   ├── mod.rs            registration + ObservableSpec parsing
    │   ├── msd.rs
    │   ├── velocity.rs
    │   ├── bursts.rs
    │   ├── shape.rs
    │   ├── per_cell.rs
    │   ├── pairwise.rs       NEW (cross_correlation, relative_msd, burst_coincidence)
    │   ├── overlap.rs
    │   ├── structure.rs
    │   └── nongaussian.rs
    ├── discovery.rs          DiscoveryRule + discover()
    ├── analyze_run.rs        RunSpec + analyze_run -> RunAnalysis
    ├── aggregate.rs          Aggregator trait + GroupBy/MeanStderr/Sweep/PairRatio
    ├── panels/               one module per panel
    │   ├── mod.rs            Panel trait + registry + layout helpers
    │   ├── run/              one trajectory at a time
    │   ├── pair/             two runs/conditions
    │   ├── sweep/            axis vs metric (FSS, separation, percolation, ...)
    │   └── summary.rs
    └── studies.rs            declarative TOML schema + executor
```

During migration the new modules live under
`rust/vtk_viewer/src/analysis/v2/` so the old code keeps working untouched.
At the very end of the migration the v2 tree is promoted to top-level
`pipeline/` and the legacy `analysis/observables.rs`,
`analysis/study.rs`, `analysis/panels.rs` are deleted.

## Type contracts

### Observable (Layer 2)

```rust
pub trait Observable: Send + Sync {
    type Output: Serialize + DeserializeOwned + Send + Sync + 'static;
    fn id(&self) -> &'static str;
    fn requires(&self) -> Requirements;     // bitset: positions, trajectory, checkpoint
    fn compute(&self, ctx: &Context) -> anyhow::Result<Self::Output>;
}
```

`Context` carries the loaded data (positions, trajectory, optional
checkpoint) plus per-run config (tau, R, soft cell IDs). Compute
functions never re-read files.

`ObservableBag` is a typed map: insert by `Observable::Output` type,
look up the same way. No string keys at consume time.

### RunAnalysis (Layer 3)

```rust
pub struct RunAnalysis {
    pub spec: RunSpec,
    pub variables: BTreeMap<String, ScalarValue>,  // typed, not strings
    pub params: SimParams,
    pub bag: ObservableBag,
}
```

Serializable as `run_analysis.json`. The portable artifact across
machines: compute on Nibi, plot locally.

### Aggregator (Layer 4)

```rust
pub trait Aggregator {
    type Input;
    type Output;
    fn run(&self, input: Self::Input) -> anyhow::Result<Self::Output>;
}
```

Concrete: `GroupBy`, `MeanStderr`, `PairRatio`, `Sweep`, `Reference`.

### Panel (Layer 5)

```rust
pub trait Panel<B: DrawingBackend> {
    type Data;
    fn id(&self) -> &'static str;
    fn render(&self, area: &DrawingArea<B, Shift>, data: &Self::Data,
              opts: &PanelOpts) -> anyhow::Result<()>;
}
```

Panel modules split by what they consume (run / pair / sweep / summary).

### Study (Layer 6)

A study is a TOML file. No Rust per study type. Schema sketch:

```toml
[study]
name = "Phase 3A pairwise"
output_dir = "phase3a_results"

[discovery]
pattern   = "phase3a/d_{d:f64}R/run_{rep:int}"
trajectory_name  = "trajectory.txt"
checkpoint_name  = "checkpoint.bin"

[observables]
compute = ["msd", "diffusion(lag=8tau)", "velocity_distribution",
           "bursts(k_sigma=3, min_frames=1)",
           "pairwise_cross_correlation(soft=[0,1])",
           "burst_coincidence(soft=[0,1])",
           "in_run_reference(n=20, exclude=[0,1])"]
tau = 10000.0
cell_radius = 49.0

[[aggregate]]
op    = "groupby"
vars  = ["d"]
into  = "by_d"

[[aggregate]]
op      = "mean_stderr"
input   = "by_d"
metrics = ["soft_d_eff", "bg_d_eff", "c_12", "burst_coincidence_index"]
into    = "summary"

[[aggregate]]
op    = "sweep"
axis  = "d"
input = "summary"
into  = "phase3a_curve"

[[figure]]
output = "phase3a_summary.svg"
layout = [2, 2]
panels = [
  { type = "metric_vs_x", input = "phase3a_curve", x = "d",
    y = "soft_d_eff_minus_bg", title = "Excess D_eff" },
  { type = "metric_vs_x", input = "phase3a_curve", x = "d",
    y = "c_12", title = "Velocity correlation" },
  { type = "metric_vs_x", input = "phase3a_curve", x = "d",
    y = "burst_coincidence_index", title = "Burst coincidence" },
  { type = "summary", input = "by_d" },
]
```

The TOML refers to observables by ID, aggregators by op name,
panels by type. All three are typed registries discoverable via
`cell_analyze list-{observables,aggregators,panels}`.

## Migration phases

Each phase is independently shippable, builds + tests pass, no
behavior change to existing studies until phase 8 (cutover).

| # | Phase | What lands | Notes |
|---|-------|------------|-------|
| 1 | Skeleton + Observable trait | `analysis/v2/{mod, observable, observables/mod}.rs`, registry stub | builds, tests pass |
| 2 | First port: MSD vertical slice | `observables/msd.rs` ports `compute_msd` to the trait, ObservableBag round-trip test | proves the design |
| 3 | Discovery + analyze_run | `v2/discovery.rs`, `v2/analyze_run.rs`, RunAnalysis JSON schema | end-to-end "load + compute MSD + serialize" |
| 4 | More observables + Aggregator | port velocity, bursts, per_cell, pairwise; introduce GroupBy + MeanStderr | feature parity for soft-vs-hard |
| 5 | Panel trait + sweep_metric panel | `v2/panels/mod.rs` + `panels/sweep/metric_vs_x.rs` | first new SVG |
| 6 | Pair panels + summary panel | port the soft-vs-hard 6/8-panel layout panels into `panels/pair/` | parity with current diagnostic |
| 7 | Declarative study TOML executor | `v2/studies.rs` parses + dispatches the new schema, FSS + soft-vs-hard rewritten as TOMLs | both old and new paths still work |
| 8 | Phase 3A study TOML | First *new* study uses the v2 path end-to-end | scientific deliverable |
| 9 | Cutover | delete legacy `analysis/{observables,study,panels}.rs`, promote `v2/` to top-level `pipeline/` | breaking but contained |

Migration phases 2–6 can land in any order once 1 is done; phase 7 needs
4+5 done; phase 8 needs 7. Phase 9 is the very end.

## Status

| phase | status | landed in |
|-------|--------|-----------|
| 1     | ✅ done | `analysis/v2/{mod, observable}.rs` |
| 2     | ✅ done | `analysis/v2/observables/{mod, msd}.rs` |
| 3     | ✅ done | `analysis/v2/{discovery, analyze_run}.rs` |
| 4     | ✅ done | `analysis/v2/aggregate.rs` |
| 5     | ✅ done | `analysis/v2/panels/{mod, layout, sweep}.rs` |
| 6     | ✅ done | `analysis/v2/studies.rs` |
| 7     | ✅ done | `cell_analyze study2` subcommand |
| 8     | ✅ done | observables {ln_perimeter, displacement_velocities, msd_palmieri, velocity_distribution, bursts}, pair panels, `pair_runs` aggregator |
| 9     | ✅ done | inlined gvi/msd/bursts/velocity_distribution; added single + overlay aggregators and panels; rewrote CLI; deleted legacy `analysis/{study,observables,panels,output,batch}.rs` (~5300 LOC); shipped 5 reference TOMLs in `cpp/simulation/study/templates/` |
