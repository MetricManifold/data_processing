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
    pub time: f32,
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
    if version < 2 || version > 5 {
        anyhow::bail!("Unsupported checkpoint version {} (expected 2-5)", version);
    }

    // Step
    f.read_exact(&mut buf4)?;
    let step = i32::from_le_bytes(buf4);

    // Time (float, not double!)
    f.read_exact(&mut buf4)?;
    let time = f32::from_le_bytes(buf4);

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

    // Detect old vs new v4 format
    let sim_params_size;
    if version <= 3 {
        sim_params_size = 76u32; // approximate old size
    } else {
        // Check for old format: read bytes at offset 36
        let pos = f.stream_position()?;
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

    // Read SimParams — we only need a few fields
    let sp_start = f.stream_position()?;
    let mut sp_buf = vec![0u8; sim_params_size as usize];
    f.read_exact(&mut sp_buf)?;

    // SimParams layout (first fields): Nx(i32), Ny(i32), dx(f32), dy(f32), dt(f32), ...
    let nx = i32::from_le_bytes(sp_buf[0..4].try_into()?);
    let ny = i32::from_le_bytes(sp_buf[4..8].try_into()?);
    let dx = f32::from_le_bytes(sp_buf[8..12].try_into()?);
    let dy = f32::from_le_bytes(sp_buf[12..16].try_into()?);
    let dt = f32::from_le_bytes(sp_buf[16..20].try_into()?);

    // SimParams layout (from types.cuh):
    // 0: Nx(i32), 4: Ny(i32), 8: dx(f32), 12: dy(f32), 16: dt(f32),
    // 20: t_end(f32), 24: save_interval(i32), 28: lambda(f32), 32: gamma(f32),
    // 36: kappa(f32), 40: target_radius(f32), 44: mu(f32),
    // 48: v_A(f32), 52: xi(f32), 56: tau(f32),
    // 60: halo_width(i32), 64: min_subdomain_size(i32), 68: subdomain_padding(f32)
    let v_a = if sp_buf.len() > 52 {
        f32::from_le_bytes(sp_buf[48..52].try_into().unwrap_or([0;4]))
    } else { 0.0 };
    let tau = if sp_buf.len() > 60 {
        f32::from_le_bytes(sp_buf[56..60].try_into().unwrap_or([0;4]))
    } else { 10000.0 };
    let target_radius = if sp_buf.len() > 44 {
        f32::from_le_bytes(sp_buf[40..44].try_into().unwrap_or([0;4]))
    } else { 49.0 };
    let halo_width = if sp_buf.len() > 64 {
        i32::from_le_bytes(sp_buf[60..64].try_into().unwrap_or([0;4]))
    } else { 4 };
    let lambda = if sp_buf.len() > 32 {
        f32::from_le_bytes(sp_buf[28..32].try_into().unwrap_or([0;4]))
    } else { 7.0 };

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

    // step, time, num_cells
    f.read_exact(&mut buf4)?;
    f.read_exact(&mut buf4)?;
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
