---
applyTo: "rust/vtk_viewer/**"
---

# VTK Viewer - Copilot Instructions

> **When to consult this file:** You are building, modifying, or debugging the Rust-based VTK viewer (`rust/vtk_viewer/`). This covers the rendering pipeline, colormap system, glow effects, layer compositing, preset system, and UI patterns. For the Python visualization alternative, see [postprocessing.instructions.md](postprocessing.instructions.md). For simulation parameters that affect VTK output, see [cell-simulation.instructions.md](cell-simulation.instructions.md).

## ⚠️ Related Instructions - Read Before Proceeding

| Task | Instruction File |
|------|-----------------|
| Cell simulation parameters/physics | [cell-simulation.instructions.md](cell-simulation.instructions.md) |
| Python visualization (alternative) | [postprocessing.instructions.md](postprocessing.instructions.md) |
| Viewing cluster-generated VTK files | [cluster-operations.instructions.md](cluster-operations.instructions.md) |

**Note**: This viewer loads VTK files from the cell simulation. Ensure output format compatibility when making simulation changes.

---

This is a high-performance VTK frame viewer written in Rust using `eframe`/`egui` for the GUI. It visualizes 2D structured grid VTK files with multi-layer compositing, colormaps, and glow effects.

**Primary Use Case**: Visualizing time-series simulation data (phase fields, stress tensors, etc.) from the cell simulation project.

## Project Location

- **Source code**: `rust/vtk_viewer/src/`
- **Executable**: `rust/vtk_viewer/target/release/vtk_viewer.exe`
- **Runtime files** (generated next to exe): `presets.json`, `recents.txt`

## Architecture

```
rust/vtk_viewer/
├── src/
│   ├── main.rs      # Application entry, GUI, state management, presets
│   ├── vtk.rs       # VTK file parsing
│   ├── colormap.rs  # Colormap definitions and application
│   └── render.rs    # Frame rendering with layer compositing and glow
├── Cargo.toml
└── .github/         # (deprecated, use root .github/instructions/)
```

## Module Responsibilities

### `vtk.rs` - VTK Parsing
- Parses ASCII VTK structured grid files
- Extracts dimensions and scalar field data
- **Key types**: `VtkData`, `Dimensions`
- **Key functions**:
  - `parse_vtk(path) -> Result<VtkData>` - parse single file
  - `find_vtk_frames(dir) -> Result<Vec<PathBuf>>` - find all `frame_*.vtk` files sorted by number

```rust
pub struct VtkData {
    pub dims: Dimensions,                      // nx, ny grid dimensions
    pub scalars: HashMap<String, Vec<f32>>,    // field_name -> values
}
```

### `colormap.rs` - Color Mapping
- Defines `Colormap` enum with 14 colormaps
- **Key functions**:
  - `apply_colormap(colormap, t) -> [u8; 4]` - normalized value to RGBA
  - `generate_preview(colormap, width, height) -> Vec<u8>` - UI thumbnail
  - `power_normalize()`, `symmetric_power_normalize()` - value normalization
- **Colormap types**:
  - Sequential: VonMises, Plasma, Viridis, Inferno, Magma, Turbo, Ocean, Thermal, Grayscale
  - Diverging: Pressure, Coolwarm, Spectral (symmetric around zero)

### `render.rs` - Rendering Pipeline
- **Key types**:
  - `LayerConfig`: Per-layer settings (field, colormap, opacity, gamma, glow, range)
  - `RenderConfig`: Collection of layers + background color (serializable for presets)
  - `GlowConfig`: Glow effect with `GlowNode` system
  - `GlowNode`: Defines glow position on value scale with falloff
  - `GlowColor`: Glow color options (MatchColormap, White, Cyan, etc.)
- **Key functions**:
  - `render_frame(vtk, config) -> Vec<u8>` - main rendering
  - `compute_derived_fields(vtk)` - calculates von_mises, tau_max from stress tensors
  - `apply_glow()` - post-process glow effect

### `main.rs` - Application
- `App` struct holds all application state
- **Features**:
  - Async frame loading with progress bar (uses `mpsc` channels)
  - Native file browser (`rfd` crate)
  - Recent directories tracking (persisted to `recents.txt`)
  - Preset save/load system (persisted to `presets.json`)
  - Playback controls (play/pause, FPS slider, frame scrubbing)
  - Layer configuration UI with colormap previews

## Key Data Structures

### LayerConfig
```rust
pub struct LayerConfig {
    pub name: String,
    pub field: String,          // VTK scalar field name
    pub colormap: Colormap,
    pub enabled: bool,
    pub opacity: f32,           // 0.0 - 1.0
    pub gamma: f32,             // 0.1 - 2.0 (contrast adjustment)
    pub glow: GlowConfig,
    pub auto_range: bool,       // Auto min/max vs manual
    pub manual_min: f32,
    pub manual_max: f32,
}
```

### GlowConfig & GlowNode
```rust
pub struct GlowConfig {
    pub enabled: bool,
    pub intensity: f32,         // 0.1 - 2.0
    pub radius: f32,            // Blur radius in pixels (1-10)
    pub color: GlowColor,       // MatchColormap, White, Cyan, Magenta, Gold, Red, Green, Blue
    pub nodes: Vec<GlowNode>,   // Where glow appears on value scale
}

pub struct GlowNode {
    pub enabled: bool,
    pub position: f32,          // 0.0 (min) to 1.0 (max) on normalized scale
    pub falloff: f32,           // 0.01-0.5, sharpness of glow dropoff
    pub name: String,
}
```

## Rendering Pipeline

1. **Field Selection**: Get scalar field from `vtk.scalars` by name
2. **Masking**: Use `phi > 0.1` as cell interior mask
3. **Normalization**: Map field values to 0.0-1.0 (auto percentile or manual range)
4. **Gamma Correction**: Apply `t.powf(gamma)` for contrast
5. **Colormap Application**: Convert normalized value to RGBA
6. **Layer Compositing**: Alpha-blend layers back-to-front onto background
7. **Glow Effect**: Apply post-process glow ON TOP of rendered pixels

### Glow System
- Glow intensity at value `t` computed from enabled `GlowNode`s
- Uses atan-based falloff: `1 - (2/π) * atan(dist / falloff)`
- Glow rendered additively on top of pixels
- Presets available: `GlowConfig::sequential()` (max only), `GlowConfig::diverging()` (both extremes)

## Adding New Features

### Adding a New Colormap
1. Add variant to `Colormap` enum in `colormap.rs`
2. Add to `Colormap::all()` array (controls UI order)
3. Add name in `Colormap::name()` match arm
4. Implement color function in `apply_colormap()` match arm
5. Mark as diverging in `is_diverging()` if symmetric around zero

### Adding a New Derived Field
1. In `render.rs`, find `compute_derived_fields()`
2. Check required input fields exist in `vtk.scalars`
3. Calculate new field values
4. Insert into `vtk.scalars` HashMap

### Adding New Layer Settings
1. Add field to `LayerConfig` struct (include `Serialize, Deserialize` derives)
2. Update `Default` impl for `LayerConfig`
3. Add UI controls in `main.rs` layer panel (inside the `for (i, layer)` loop)
4. Use the setting in `render_frame()` in `render.rs`

### Adding New UI Panels
- Use `egui::SidePanel`, `TopBottomPanel`, or `CentralPanel`
- Add state to `App` struct
- Add UI code in `eframe::App::update()` impl

## Coding Conventions

### Error Handling
- Use `anyhow::Result` for fallible operations
- Parse errors include file path context
- UI displays errors via `self.error: Option<String>`

### Performance
- Use `rayon` for parallel iteration (`.par_iter()`, `.into_par_iter()`)
- Cache rendered pixels in `self.cached_pixels: Option<Vec<u8>>`
- Set `self.cached_pixels = None` when config changes to trigger re-render
- Async loading uses `mpsc` channels to not block UI

### Serialization
- Config types derive `Serialize, Deserialize` for preset system
- Presets stored as JSON in `presets.json` next to executable
- Use `serde_json::to_string_pretty()` for human-readable output

### UI Patterns
```rust
// Trigger re-render on any change
let mut changed = false;
if ui.add(egui::Slider::new(&mut value, range)).changed() {
    changed = true;
}
// ... at end of panel:
if changed { self.cached_pixels = None; }
```

```rust
// Combo box pattern
egui::ComboBox::from_id_salt("unique_id")
    .selected_text(current.name())
    .show_ui(ui, |ui| {
        for item in Items::all() {
            if ui.selectable_value(&mut current, *item, item.name()).changed() {
                changed = true;
            }
        }
    });
```

## File Format: VTK Structured Points (ASCII)

Expected format for input files:
```
# vtk DataFile Version 3.0
Description
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS nx ny 1
ORIGIN x0 y0 z0
SPACING dx dy dz
POINT_DATA n
SCALARS field_name float 1
LOOKUP_TABLE default
value1 value2 value3 ...
SCALARS another_field float 1
LOOKUP_TABLE default
...
```

File naming convention: `frame_NNNNNN.vtk` (6-digit zero-padded frame number)

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `eframe` | 0.29 | Cross-platform GUI framework |
| `egui` | 0.29 | Immediate-mode UI library |
| `image` | 0.25 | Image encoding (if needed) |
| `rayon` | 1.8 | Parallel iteration |
| `anyhow` | 1.0 | Error handling |
| `rfd` | 0.15 | Native file dialogs |
| `serde` | 1.0 | Serialization framework (with `derive` feature) |
| `serde_json` | 1.0 | JSON preset storage |

## Build and Run

```powershell
cd rust/vtk_viewer
cargo build --release
.\target\release\vtk_viewer.exe
```

## Testing Changes

1. Build: `cargo build --release`
2. Run: `.\target\release\vtk_viewer.exe`
3. Click Browse, select VTK output directory (e.g., `cpp/simulation/agent_test_runs/my_sim/`)
4. Verify:
   - Frames load with progress bar
   - Layer controls work (enable/disable, colormap, opacity)
   - Glow effects render correctly
   - Preset save/load cycle works
   - Playback controls function

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No VTK frames found" | Ensure directory contains `frame_*.vtk` files |
| Fields not appearing | Check VTK file has expected SCALARS sections |
| Rendering artifacts | Try manual range instead of auto; check colormap type matches data |
| Glow not visible | Increase intensity; check node positions match data extremes |
| Slow performance | Reduce glow radius; disable unused layers; check frame size |
| Presets not saving | Check write permissions in executable directory |

## Related Projects

This viewer is designed to work with output from the cell simulation project:
- **Simulation code**: `cpp/simulation/`
- **VTK output location**: `cpp/simulation/agent_test_runs/*/frame_*.vtk`
- **Stress field visualization**: Enable stress fields with `-DENABLE_STRESS_FIELDS=ON` in simulation build
