//! Read v7 (sim_v3) checkpoint files written by `cell_sim`.
//!
//! Layout (all little-endian):
//!   u32 magic = 0x43454C4C
//!   u32 version (>= 7)
//!   i32 step
//!   f64 cur_time
//!   i32 num_cells
//!   i32 save_interval, i32 checkpoint_interval, i32 trajectory_samples,
//!   4 bytes flags
//!   u32 sp_size, then sp_size bytes of SimParams
//!   i32 T  (tile size)
//!   per cell: i32 id, i32 ox, i32 oy, f32 cx, f32 cy, f32 vx, f32 vy,
//!             f32 volume, f32 phi[T*T]
//!
//! The SimParams blob has three known layouts (see Python conftest.py):
//!   sp_size = 72 / 92  → baseline (f32 scalars; Nx,Ny@0,4; lambda@28; gamma@32; ...)
//!   sp_size = 88       → sim_v2 v5 (f32 scalars; lambda@24; gamma@28; ...)
//!   sp_size = 144      → sim_v2 v6 (f64 scalars; lambda@40; gamma@48; ...)

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Debug, Clone)]
pub struct CkptParams {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub dt: f64,
    pub lambd: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub mu: f64,
    pub xi: f64,
    pub target_radius: f64,
}

#[derive(Debug, Clone)]
pub struct CkptCell {
    pub id: i32,
    pub ox: i32,
    pub oy: i32,
    pub cx: f32,
    pub cy: f32,
    pub vx: f32,
    pub vy: f32,
    pub volume: f32,
    /// Tile of size T*T, row-major (y, x).
    pub phi_tile: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct Sidecars {
    /// Per-cell tumble angles theta (radians). One f32 per cell.
    pub polr: Option<Vec<f32>>,
    pub va_a: Option<Vec<f32>>,
    pub gama: Option<Vec<f32>>,
    pub radi: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub t: f64,
    pub params: CkptParams,
    pub tile_t: usize,
    pub cells: Vec<CkptCell>,
    pub sidecars: Sidecars,
}

fn rd_u32_at(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn rd_i32_at(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn rd_f32_at(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn rd_f64_at(buf: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

fn parse_params(sp: &[u8]) -> Result<CkptParams> {
    let n = sp.len();
    let p = match n {
        144 => CkptParams {
            // v6: f64 scalars, ints @0,4; doubles @8.. ; ints @112..
            nx: rd_i32_at(sp, 0) as usize,
            ny: rd_i32_at(sp, 4) as usize,
            dx: rd_f64_at(sp, 8),
            dy: rd_f64_at(sp, 16),
            dt: rd_f64_at(sp, 24),
            // t_end @ 32
            lambd: rd_f64_at(sp, 40),
            gamma: rd_f64_at(sp, 48),
            kappa: rd_f64_at(sp, 56),
            target_radius: rd_f64_at(sp, 64),
            mu: rd_f64_at(sp, 72),
            // v_A @ 80
            xi: rd_f64_at(sp, 88),
            // tau @ 96
        },
        88 => CkptParams {
            // sim_v2 v5: f32 scalars
            nx: rd_i32_at(sp, 0) as usize,
            ny: rd_i32_at(sp, 4) as usize,
            dx: rd_f32_at(sp, 8) as f64,
            dy: rd_f32_at(sp, 12) as f64,
            dt: rd_f32_at(sp, 16) as f64,
            lambd: rd_f32_at(sp, 24) as f64,
            gamma: rd_f32_at(sp, 28) as f64,
            kappa: rd_f32_at(sp, 32) as f64,
            target_radius: rd_f32_at(sp, 36) as f64,
            mu: rd_f32_at(sp, 40) as f64,
            xi: rd_f32_at(sp, 48) as f64,
        },
        72 | 92 => CkptParams {
            // baseline: f32 scalars, lambda @ 28
            nx: rd_i32_at(sp, 0) as usize,
            ny: rd_i32_at(sp, 4) as usize,
            dx: rd_f32_at(sp, 8) as f64,
            dy: rd_f32_at(sp, 12) as f64,
            dt: rd_f32_at(sp, 16) as f64,
            lambd: rd_f32_at(sp, 28) as f64,
            gamma: rd_f32_at(sp, 32) as f64,
            kappa: rd_f32_at(sp, 36) as f64,
            target_radius: rd_f32_at(sp, 40) as f64,
            mu: rd_f32_at(sp, 44) as f64,
            xi: rd_f32_at(sp, 52) as f64,
        },
        _ => bail!("unsupported SimParams sp_size={}", n),
    };
    Ok(p)
}

pub fn read(path: &Path) -> Result<Checkpoint> {
    let mut f = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let mut c = std::io::Cursor::new(&buf);

    let magic = c.read_u32::<LittleEndian>()?;
    if magic != 0x4345_4C4C {
        bail!("bad magic 0x{:x}", magic);
    }
    let version = c.read_u32::<LittleEndian>()?;
    if version < 7 {
        bail!("only checkpoint version >= 7 supported (got {})", version);
    }
    let _step = c.read_i32::<LittleEndian>()?;
    let t = c.read_f64::<LittleEndian>()?;
    let num_cells = c.read_i32::<LittleEndian>()? as usize;

    let _save_interval = c.read_i32::<LittleEndian>()?;
    let _checkpoint_interval = c.read_i32::<LittleEndian>()?;
    let _trajectory_samples = c.read_i32::<LittleEndian>()?;
    let mut flags = [0u8; 4];
    c.read_exact(&mut flags)?;

    let sp_size = c.read_u32::<LittleEndian>()? as usize;
    let mut sp = vec![0u8; sp_size];
    c.read_exact(&mut sp)?;
    let params = parse_params(&sp)?;

    let tile_t = c.read_i32::<LittleEndian>()? as usize;
    let tt = tile_t * tile_t;

    let mut cells = Vec::with_capacity(num_cells);
    for _ in 0..num_cells {
        let id = c.read_i32::<LittleEndian>()?;
        let ox = c.read_i32::<LittleEndian>()?;
        let oy = c.read_i32::<LittleEndian>()?;
        let cx = c.read_f32::<LittleEndian>()?;
        let cy = c.read_f32::<LittleEndian>()?;
        let vx = c.read_f32::<LittleEndian>()?;
        let vy = c.read_f32::<LittleEndian>()?;
        let volume = c.read_f32::<LittleEndian>()?;
        let mut phi_tile = vec![0f32; tt];
        for v in phi_tile.iter_mut() {
            *v = c.read_f32::<LittleEndian>()?;
        }
        cells.push(CkptCell {
            id, ox, oy, cx, cy, vx, vy, volume, phi_tile,
        });
    }

    Ok(Checkpoint { t, params, tile_t, cells, sidecars: read_sidecars(&mut c) })
}

fn read_sidecars(c: &mut std::io::Cursor<&Vec<u8>>) -> Sidecars {
    // Tagged sidecars at file tail: u32 magic + i32 count + count*f32 data.
    // Magic values are little-endian-encoded ASCII; reading as u32 LE returns
    // the tag in REVERSED character order.
    //   on disk "RLOP" = u32 0x504F4C52 = ASCII "POLR"
    //   on disk "A_AV" = u32 0x56415F41 = ASCII "VA_A"
    //   on disk "AMAG" = u32 0x47414D41 = ASCII "GAMA"
    //   on disk "IDAR" = u32 0x52414449 = ASCII "RADI"
    let mut sc = Sidecars::default();
    loop {
        let pos = c.position();
        let m = match c.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(_) => break,
        };
        let count = match c.read_i32::<LittleEndian>() {
            Ok(v) => v,
            Err(_) => { c.set_position(pos); break; }
        };
        if count <= 0 || count > 1_000_000 {
            c.set_position(pos); break;
        }
        let mut data = vec![0f32; count as usize];
        let mut ok = true;
        for v in data.iter_mut() {
            match c.read_f32::<LittleEndian>() {
                Ok(x) => *v = x,
                Err(_) => { ok = false; break; }
            }
        }
        if !ok { c.set_position(pos); break; }
        match m {
            0x504F4C52 => sc.polr = Some(data),    // 'POLR'
            0x56415F41 => sc.va_a = Some(data),    // 'VA_A'
            0x47414D41 => sc.gama = Some(data),    // 'GAMA'
            0x52414449 => sc.radi = Some(data),    // 'RADI'
            _ => { c.set_position(pos); break; }   // unknown tag, stop
        }
    }
    sc
}
