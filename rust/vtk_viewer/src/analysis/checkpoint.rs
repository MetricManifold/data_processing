//! Checkpoint reader for the cell simulation binary.
//!
//! Reads the v4 binary checkpoint format and extracts:
//! - Header (step, time, num_cells)
//! - SimParams (Nx, Ny, dt, R, v_A, etc.)
//! - Per-cell bounding boxes, centroids, and phi fields
//! - Optional per-cell v_A, gamma, and radius arrays

use anyhow::{Context, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Checkpoint header (mirrors C++ CheckpointHeader).
#[derive(Debug, Clone)]
pub struct CheckpointHeader {
    pub magic: u32,
    pub version: u32,
    pub step: i32,
    pub time: f64,
    pub num_cells: i32,
    pub save_interval: i32,
    pub checkpoint_interval: i32,
    pub trajectory_samples: i32,
    pub save_vtk: bool,
    pub save_tracking: bool,
    pub compute_diagnostics: bool,
    pub save_individual_fields: bool,
    pub sim_params_size: u32,
}

/// Subset of SimParams we care about.
#[derive(Debug, Clone)]
pub struct SimParams {
    pub nx: i32,
    pub ny: i32,
    pub dx: f32,
    pub dy: f32,
    pub dt: f32,
    pub target_radius: f32,
    pub v_a: f32,
    pub tau: f32,
    pub halo_width: i32,
    pub lambda: f32,
}

/// Bounding box (inner, without halo).
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl BBox {
    pub fn width(&self) -> i32 { self.x1 - self.x0 }
    pub fn height(&self) -> i32 { self.y1 - self.y0 }
}

/// Per-cell data from checkpoint.
#[derive(Debug, Clone)]
pub struct CellData {
    pub id: i32,
    pub bbox: BBox,
    pub centroid: (f32, f32),
    pub velocity: (f32, f32),
    pub volume: f32,
    /// Phi field stored in row-major order, size = (bbox_h + 2*halo) * (bbox_w + 2*halo).
    pub phi: Vec<f32>,
    /// Width of phi field including halo.
    pub phi_w: i32,
    /// Height of phi field including halo.
    pub phi_h: i32,
}

/// Full checkpoint data.
#[derive(Debug)]
pub struct Checkpoint {
    pub header: CheckpointHeader,
    pub params: SimParams,
    pub cells: Vec<CellData>,
    pub per_cell_v_a: Vec<f32>,
    pub per_cell_gamma: Vec<f32>,
    pub per_cell_radius: Vec<f32>,
}

impl Checkpoint {
    /// Composite all cell phi fields into a single Nx×Ny image using max blending.
    pub fn composite_phi(&self) -> Vec<f32> {
        let nx = self.params.nx as usize;
        let ny = self.params.ny as usize;
        let halo = self.params.halo_width;
        let mut phi = vec![0.0f32; ny * nx];

        for cell in &self.cells {
            // The phi data is stored in a pool-slot-sized grid.
            // The cell's inner bbox defines where the cell sits in global coords.
            // The phi grid is centered on the inner bbox with halo.
            let bx0 = cell.bbox.x0 - halo;
            let by0 = cell.bbox.y0 - halo;
            let inner_w = cell.bbox.width() + 2 * halo;
            let inner_h = cell.bbox.height() + 2 * halo;
            let fw = cell.phi_w as usize;
            let fh = cell.phi_h as usize;

            // Only iterate over the actual bbox region (inner_w × inner_h),
            // not the full pool slot
            let use_h = (inner_h as usize).min(fh);
            let use_w = (inner_w as usize).min(fw);

            for ly in 0..use_h {
                let gy = ((by0 + ly as i32) % ny as i32 + ny as i32) as usize % ny;
                for lx in 0..use_w {
                    let gx = ((bx0 + lx as i32) % nx as i32 + nx as i32) as usize % nx;
                    let val = cell.phi[ly * fw + lx];
                    let idx = gy * nx + gx;
                    if val > phi[idx] {
                        phi[idx] = val;
                    }
                }
            }
        }
        phi
    }
}

/// Load a checkpoint file.
pub fn load_checkpoint(path: &Path) -> Result<Checkpoint> {
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("Opening checkpoint: {}", path.display()))?;

    // Read header (40 bytes for v4 new format)
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Magic
    f.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != 0x43454C4C {
        anyhow::bail!("Invalid checkpoint magic: 0x{:08X} (expected 0x43454C4C)", magic);
    }

    // Version
    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version < 2 || version > 6 {
        anyhow::bail!("Unsupported checkpoint version {} (expected 2-6)", version);
    }

    // Step
    f.read_exact(&mut buf4)?;
    let step = i32::from_le_bytes(buf4);

    // Time: v5+ stores f64 (sim_v2 fix for float32 precision wall at t=2^18).
    //       v2-v4 stores f32.
    let time: f64 = if version >= 5 {
        f.read_exact(&mut buf8)?;
        f64::from_le_bytes(buf8)
    } else {
        f.read_exact(&mut buf4)?;
        f32::from_le_bytes(buf4) as f64
    };

    // num_cells
    f.read_exact(&mut buf4)?;
    let num_cells = i32::from_le_bytes(buf4);

    // save_interval, checkpoint_interval, trajectory_samples
    f.read_exact(&mut buf4)?;
    let save_interval = i32::from_le_bytes(buf4);
    f.read_exact(&mut buf4)?;
    let checkpoint_interval = i32::from_le_bytes(buf4);
    f.read_exact(&mut buf4)?;
    let trajectory_samples = i32::from_le_bytes(buf4);

    // 4 bools (1 byte each, packed into 4 bytes)
    let mut flags = [0u8; 4];
    f.read_exact(&mut flags)?;

    // Detect old vs new v4 format. v5+ has a clean layout (no padding guess).
    let sim_params_size;
    if version <= 3 {
        sim_params_size = 76u32; // approximate old size
    } else if version >= 5 {
        // v5+: sim_params_size immediately follows flags (44-byte header: time is f64).
        f.read_exact(&mut buf4)?;
        sim_params_size = u32::from_le_bytes(buf4);
    } else {
        // Check for old format: read bytes at offset 36
        let _pos = f.stream_position()?;
        f.read_exact(&mut buf4)?;
        let val_at_36 = u32::from_le_bytes(buf4);
        f.read_exact(&mut buf4)?;
        let val_at_40 = u32::from_le_bytes(buf4);

        if val_at_36 == 0 && val_at_40 >= 64 && val_at_40 <= 512 {
            // Old format with _padding field
            sim_params_size = val_at_40;
            // Already past the padding + size, at start of SimParams (offset 44)
        } else {
            // New format: val_at_36 IS sim_params_size
            sim_params_size = val_at_36;
            // Seek back to offset 40 (start of SimParams)
            f.seek(SeekFrom::Start(40))?;
        }
    }

    let header = CheckpointHeader {
        magic, version, step, time, num_cells,
        save_interval, checkpoint_interval, trajectory_samples,
        save_vtk: flags[0] != 0,
        save_tracking: flags[1] != 0,
        compute_diagnostics: flags[2] != 0,
        save_individual_fields: flags[3] != 0,
        sim_params_size,
    };

    // Read SimParams — we only need a few fields.
    // Three layouts coexist:
    //   baseline (sp_size=72 or 92): lambda@28, gamma@32, target_radius@40 (f32)
    //   sim_v2 v5 (sp_size=88):      lambda@24, gamma@28, target_radius@36 (f32)
    //   sim_v2 v6 (sp_size=144):     scalars as f64; Nx/Ny i32, then 13 f64, then ints
    let _sp_start = f.stream_position()?;
    let mut sp_buf = vec![0u8; sim_params_size as usize];
    f.read_exact(&mut sp_buf)?;

    let (nx, ny, dx, dy, dt, lambda, target_radius, v_a, tau, halo_width);
    if sim_params_size == 144 {
        // sim_v2 v6 layout (f64 scalars)
        nx = i32::from_le_bytes(sp_buf[0..4].try_into()?);
        ny = i32::from_le_bytes(sp_buf[4..8].try_into()?);
        dx = f64::from_le_bytes(sp_buf[8..16].try_into()?) as f32;
        dy = f64::from_le_bytes(sp_buf[16..24].try_into()?) as f32;
        dt = f64::from_le_bytes(sp_buf[24..32].try_into()?) as f32;
        // t_end@32, lambda@40, gamma@48, kappa@56, target_radius@64, mu@72,
        // v_A@80, xi@88, tau@96, subdomain_padding@104, halo_width@112
        lambda = f64::from_le_bytes(sp_buf[40..48].try_into()?) as f32;
        target_radius = f64::from_le_bytes(sp_buf[64..72].try_into()?) as f32;
        v_a = f64::from_le_bytes(sp_buf[80..88].try_into()?) as f32;
        tau = f64::from_le_bytes(sp_buf[96..104].try_into()?) as f32;
        halo_width = i32::from_le_bytes(sp_buf[112..116].try_into()?);
    } else if sim_params_size == 88 {
        // sim_v2 v5 layout (f32 scalars, sim_v2's own field order)
        nx = i32::from_le_bytes(sp_buf[0..4].try_into()?);
        ny = i32::from_le_bytes(sp_buf[4..8].try_into()?);
        dx = f32::from_le_bytes(sp_buf[8..12].try_into()?);
        dy = f32::from_le_bytes(sp_buf[12..16].try_into()?);
        dt = f32::from_le_bytes(sp_buf[16..20].try_into()?);
        // t_end@20, lambda@24, gamma@28, kappa@32, target_radius@36, mu@40,
        // v_A@44, xi@48, tau@52, subdomain_padding@56, halo@60
        lambda = f32::from_le_bytes(sp_buf[24..28].try_into()?);
        target_radius = f32::from_le_bytes(sp_buf[36..40].try_into()?);
        v_a = f32::from_le_bytes(sp_buf[44..48].try_into()?);
        tau = f32::from_le_bytes(sp_buf[52..56].try_into()?);
        halo_width = i32::from_le_bytes(sp_buf[60..64].try_into()?);
    } else {
        // baseline sim layout:
        // 0: Nx, 4: Ny, 8: dx, 12: dy, 16: dt, 20: t_end, 24: save_interval,
        // 28: lambda, 32: gamma, 36: kappa, 40: target_radius, 44: mu,
        // 48: v_A, 52: xi, 56: tau, 60: halo_width
        nx = i32::from_le_bytes(sp_buf[0..4].try_into()?);
        ny = i32::from_le_bytes(sp_buf[4..8].try_into()?);
        dx = f32::from_le_bytes(sp_buf[8..12].try_into()?);
        dy = f32::from_le_bytes(sp_buf[12..16].try_into()?);
        dt = f32::from_le_bytes(sp_buf[16..20].try_into()?);
        lambda = if sp_buf.len() > 32 { f32::from_le_bytes(sp_buf[28..32].try_into()?) } else { 7.0 };
        target_radius = if sp_buf.len() > 44 { f32::from_le_bytes(sp_buf[40..44].try_into()?) } else { 49.0 };
        v_a = if sp_buf.len() > 52 { f32::from_le_bytes(sp_buf[48..52].try_into()?) } else { 0.0 };
        tau = if sp_buf.len() > 60 { f32::from_le_bytes(sp_buf[56..60].try_into()?) } else { 10000.0 };
        halo_width = if sp_buf.len() > 64 { i32::from_le_bytes(sp_buf[60..64].try_into()?) } else { 4 };
    }

    let params = SimParams { nx, ny, dx, dy, dt, target_radius, v_a, tau, halo_width, lambda };

    eprintln!("Checkpoint: v={}, step={}, t={:.0} ({:.1}τ), cells={}, domain={}×{}",
              version, step, time, time / 10000.0, num_cells, nx, ny);

    // Read cells
    let halo = halo_width;
    let mut cells = Vec::with_capacity(num_cells as usize);
    for _ in 0..num_cells {
        // id
        f.read_exact(&mut buf4)?;
        let id = i32::from_le_bytes(buf4);

        // BoundingBox: x0, y0, x1, y1 (inner, no halo)
        let mut bb = [0u8; 16];
        f.read_exact(&mut bb)?;
        let x0 = i32::from_le_bytes(bb[0..4].try_into()?);
        let y0 = i32::from_le_bytes(bb[4..8].try_into()?);
        let x1 = i32::from_le_bytes(bb[8..12].try_into()?);
        let y1 = i32::from_le_bytes(bb[12..16].try_into()?);
        let bbox = BBox { x0, y0, x1, y1 };

        // centroid (Vec2: f32, f32)
        f.read_exact(&mut buf8)?;
        let cx = f32::from_le_bytes(buf8[0..4].try_into()?);
        let cy = f32::from_le_bytes(buf8[4..8].try_into()?);

        // velocity
        f.read_exact(&mut buf8)?;
        let vx = f32::from_le_bytes(buf8[0..4].try_into()?);
        let vy = f32::from_le_bytes(buf8[4..8].try_into()?);

        // volume
        f.read_exact(&mut buf4)?;
        let volume = f32::from_le_bytes(buf4);

        // phi field: (bbox_w + 2*halo) * (bbox_h + 2*halo) floats
        // This is cell->field_size = bbox_with_halo.size()
        let phi_w = bbox.width() + 2 * halo;
        let phi_h = bbox.height() + 2 * halo;
        let field_size = (phi_w * phi_h) as usize;

        if field_size == 0 || field_size > 500_000 {
            anyhow::bail!("Cell {} has suspicious field_size={} (bbox={}x{}, halo={})",
                          id, field_size, bbox.width(), bbox.height(), halo);
        }

        let mut phi_bytes = vec![0u8; field_size * 4];
        f.read_exact(&mut phi_bytes)
            .with_context(|| format!("Reading phi for cell {} ({}×{} = {} floats)", id, phi_w, phi_h, field_size))?;

        let phi: Vec<f32> = phi_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        cells.push(CellData {
            id, bbox, centroid: (cx, cy), velocity: (vx, vy), volume,
            phi, phi_w, phi_h,
        });
    }

    // Read optional per-cell v_A
    let mut per_cell_v_a = Vec::new();
    if let Ok(()) = f.read_exact(&mut buf4) {
        let m = u32::from_le_bytes(buf4);
        if m == 0x56415F41 { // "VA_A"
            f.read_exact(&mut buf4)?;
            let count = i32::from_le_bytes(buf4) as usize;
            let mut data = vec![0u8; count * 4];
            f.read_exact(&mut data)?;
            per_cell_v_a = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
        }
    }

    // Read optional per-cell gamma
    let mut per_cell_gamma = Vec::new();
    if let Ok(()) = f.read_exact(&mut buf4) {
        let m = u32::from_le_bytes(buf4);
        if m == 0x47414D41 { // "GAMA"
            f.read_exact(&mut buf4)?;
            let count = i32::from_le_bytes(buf4) as usize;
            let mut data = vec![0u8; count * 4];
            f.read_exact(&mut data)?;
            per_cell_gamma = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
        }
    }

    // Read optional per-cell radius
    let mut per_cell_radius = Vec::new();
    if let Ok(()) = f.read_exact(&mut buf4) {
        let m = u32::from_le_bytes(buf4);
        if m == 0x52414449 { // "RADI"
            f.read_exact(&mut buf4)?;
            let count = i32::from_le_bytes(buf4) as usize;
            let mut data = vec![0u8; count * 4];
            f.read_exact(&mut data)?;
            per_cell_radius = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
        }
    }

    Ok(Checkpoint {
        header, params, cells,
        per_cell_v_a, per_cell_gamma, per_cell_radius,
    })
}

/// Lightweight header-only read. Returns (time, num_cells, version).
/// Used by `cell_analyze check` for fast validation without loading phi fields.
pub fn load_checkpoint_header_only(path: &Path) -> Result<(f64, i32, u32)> {
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("Opening checkpoint: {}", path.display()))?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != 0x43454C4C {
        anyhow::bail!("bad magic 0x{:08X}", magic);
    }
    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    f.read_exact(&mut buf4)?;
    let _step = i32::from_le_bytes(buf4);
    let time: f64 = if version >= 5 {
        f.read_exact(&mut buf8)?;
        f64::from_le_bytes(buf8)
    } else {
        f.read_exact(&mut buf4)?;
        f32::from_le_bytes(buf4) as f64
    };
    f.read_exact(&mut buf4)?;
    let num_cells = i32::from_le_bytes(buf4);
    Ok((time, num_cells, version))
}

/// Read mean bounding box width and subdomain_padding from a checkpoint.
pub fn read_bbox_stats(path: &Path) -> Result<(f64, Option<f64>)> {
    let mut f = std::fs::File::open(path)?;
    let mut buf4 = [0u8; 4];

    // Magic + version
    f.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != 0x43454C4C { anyhow::bail!("bad magic"); }
    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);

    // step, time, num_cells. v5+: time is f64 (8 bytes); earlier: f32 (4 bytes).
    f.read_exact(&mut buf4)?;  // step
    if version >= 5 {
        let mut t8 = [0u8; 8];
        f.read_exact(&mut t8)?;
    } else {
        f.read_exact(&mut buf4)?;  // time (f32)
    }
    f.read_exact(&mut buf4)?;
    let num_cells = i32::from_le_bytes(buf4) as usize;

    // runtime opts (12 bytes) + 4 bools
    f.seek(SeekFrom::Current(16))?;

    // sim_params_size
    let sp_size = if version >= 4 {
        f.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) as usize
    } else { 76 };

    // Read SimParams to get halo and subdomain_padding
    let mut sp_buf = vec![0u8; sp_size];
    f.read_exact(&mut sp_buf)?;
    let halo = if sp_buf.len() > 64 {
        i32::from_le_bytes(sp_buf[60..64].try_into().unwrap_or([0;4]))
    } else { 4 };
    let subdomain_padding = if sp_buf.len() > 72 {
        Some(f32::from_le_bytes(sp_buf[68..72].try_into().unwrap_or([0;4])) as f64)
    } else { None };

    // Read cell bboxes, skip phi
    let mut total_w = 0i64;
    for _ in 0..num_cells {
        f.read_exact(&mut buf4)?; // id
        let mut bb = [0u8; 16];
        f.read_exact(&mut bb)?;
        let x0 = i32::from_le_bytes(bb[0..4].try_into().unwrap());
        let x1 = i32::from_le_bytes(bb[8..12].try_into().unwrap());
        let y0 = i32::from_le_bytes(bb[4..8].try_into().unwrap());
        let y1 = i32::from_le_bytes(bb[12..16].try_into().unwrap());
        let w = (x1 - x0) + 2 * halo;
        let h = (y1 - y0) + 2 * halo;
        total_w += w as i64;
        // Skip centroid(8) + velocity(8) + volume(4) + phi(w*h*4)
        let skip = 20 + (w as i64) * (h as i64) * 4;
        f.seek(SeekFrom::Current(skip))?;
    }

    let bbox_mean = total_w as f64 / num_cells as f64;
    Ok((bbox_mean, subdomain_padding))
}
