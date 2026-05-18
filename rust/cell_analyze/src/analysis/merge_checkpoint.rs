//! v8 checkpoint merger.
//!
//! Combines per-rank `checkpoint.bin` files (`<dir>/checkpoint.bin`,
//! `<dir>/rank1/checkpoint.bin`, ...) into a single v8 checkpoint with
//! `num_ranks=1, rank_id=0`. The merged file is a normal v8 checkpoint
//! that the C++ loader can resume with any `--gpus` value.
//!
//! The merge is byte-level — per-cell records and all sidecar arrays
//! (GAMA, RADI, VA_A, POLR, RNGS) are concatenated in global-id order so
//! no precision is lost through deserialize/reserialize.

use anyhow::{anyhow, bail, Context, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: u32 = 0x4345_4C4C;
const VERSION_V8: u32 = 8;

/// One sidecar block (magic, count, raw payload bytes).
#[derive(Clone)]
struct Sidecar {
    magic: u32,
    /// Bytes per element. 4 for VA_A/GAMA/RADI/POLR (f32). Variable for RNGS.
    bytes_per_elem: usize,
    /// `count` elements; each is `bytes_per_elem` bytes.
    payload: Vec<u8>,
}

/// What we extract from a per-rank file: the header (verbatim, up to and
/// including the v8 trailer), the per-cell record bytes (count × record_sz),
/// and the trailing sidecars.
struct RankFile {
    cell_ids: Vec<i32>,
    /// Per-cell raw bytes (one per cell, fixed size = `record_size`).
    cell_records: Vec<Vec<u8>>,
    sidecars: Vec<Sidecar>,
    /// Header bytes from offset 0 up to (but not including) the per-cell
    /// records. Used so we can rewrite num_ranks/rank_id/n_global in the
    /// merged output without re-parsing SimParams.
    header_bytes: Vec<u8>,
    /// Byte offset within `header_bytes` of `num_ranks` (i32).
    num_ranks_offset: usize,
    /// `num_cells` field offset inside header_bytes (needs rewriting).
    num_cells_offset: usize,
    /// (num_ranks, rank_id, num_cells_global) read from the v8 trailer.
    num_ranks: i32,
    rank_id: i32,
    num_cells_global: i32,
    /// Bytes per per-cell record (id+origin+centroid+velocity+volume + phi).
    record_size: usize,
}

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn read_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn parse_rank_file(path: &Path) -> Result<RankFile> {
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut whole = Vec::new();
    f.read_to_end(&mut whole)
        .with_context(|| format!("reading {}", path.display()))?;
    if whole.len() < 64 {
        bail!("{}: file too short ({} bytes)", path.display(), whole.len());
    }
    let magic = read_u32(&whole, 0);
    if magic != MAGIC {
        bail!("{}: bad magic 0x{:08x}", path.display(), magic);
    }
    let version = read_u32(&whole, 4);
    if version != VERSION_V8 {
        bail!(
            "{}: only v8 checkpoints can be merged (got v{}). For v7, no \
             merge is needed — it is already single-rank.",
            path.display(),
            version
        );
    }
    // Header layout up to sp_sz:
    //   magic(4) + ver(4) + step(4) + cur_time(8) + nc(4) + si(4)
    //   + reserved(4) + ts(4) + bools(4) + sp_sz(4)
    //   = 44 bytes; nc is at offset 20, sp_sz at offset 40.
    let num_cells = read_i32(&whole, 20);
    let sp_sz = read_u32(&whole, 40) as usize;
    let sp_start = 44;
    let sp_end = sp_start + sp_sz;
    if sp_end + 16 > whole.len() {
        bail!("{}: header truncated", path.display());
    }
    // After SimParams: tile_t(4), num_ranks(4), rank_id(4), n_global(4).
    let tile_t_off = sp_end;
    let tile_t = read_i32(&whole, tile_t_off);
    let num_ranks_offset = tile_t_off + 4;
    let num_ranks = read_i32(&whole, num_ranks_offset);
    let rank_id = read_i32(&whole, num_ranks_offset + 4);
    let num_cells_global = read_i32(&whole, num_ranks_offset + 8);
    let records_start = num_ranks_offset + 12;

    // Per-cell record: id(4) + origin(8) + centroid(8) + velocity(8) +
    //   volume(4) + phi(tile_t^2 * 4).
    let record_size = 32 + (tile_t as usize) * (tile_t as usize) * 4;
    let records_end = records_start + (num_cells as usize) * record_size;
    if records_end > whole.len() {
        bail!(
            "{}: per-cell records truncated (need {} bytes, have {})",
            path.display(),
            records_end,
            whole.len()
        );
    }

    let mut cell_ids = Vec::with_capacity(num_cells as usize);
    let mut cell_records = Vec::with_capacity(num_cells as usize);
    for i in 0..num_cells as usize {
        let off = records_start + i * record_size;
        cell_ids.push(read_i32(&whole, off));
        cell_records.push(whole[off..off + record_size].to_vec());
    }

    // Sidecars: each is magic(4) + count(4) + count * bytes_per_elem.
    // VA_A/GAMA/RADI/POLR use f32 (4 bytes/elem); RNGS uses curandState
    // raw bytes (variable; we infer size from total payload / count).
    let mut sidecars = Vec::new();
    let mut cur = records_end;
    while cur + 8 <= whole.len() {
        let m = read_u32(&whole, cur);
        let count = read_i32(&whole, cur + 4) as usize;
        let bytes_per_elem = match m {
            0x56415F41 | 0x47414D41 | 0x52414449 | 0x504F4C52 => 4,
            0x53474E52 => {
                // RNGS: payload bytes = remaining size if this is the last
                // block, otherwise we'd need to know curandState size. For
                // robustness, accept any payload size; downstream merges
                // require equal per-cell stride across rank files.
                let remaining = whole.len() - cur - 8;
                if count == 0 {
                    1 // avoid div-by-zero
                } else {
                    let bpe = remaining / count;
                    if bpe == 0 {
                        bail!("{}: RNGS sidecar count={} but only {} bytes left", path.display(), count, remaining);
                    }
                    bpe
                }
            }
            _ => break, // unknown magic — stop here
        };
        let payload_bytes = count * bytes_per_elem;
        if cur + 8 + payload_bytes > whole.len() {
            bail!(
                "{}: sidecar 0x{:08x} truncated (need {} bytes, have {})",
                path.display(), m, payload_bytes, whole.len() - cur - 8
            );
        }
        let payload = whole[cur + 8..cur + 8 + payload_bytes].to_vec();
        sidecars.push(Sidecar { magic: m, bytes_per_elem, payload });
        cur += 8 + payload_bytes;
    }

    // Header bytes: everything from 0 up to records_start. We'll rewrite
    // num_cells (offset 16) and (num_ranks, rank_id, n_global) at the
    // trailer when writing the merged file.
    let header_bytes = whole[..records_start].to_vec();
    Ok(RankFile {
        cell_ids,
        cell_records,
        sidecars,
        header_bytes,
        num_ranks_offset,
        num_cells_offset: 20,
        num_ranks,
        rank_id,
        num_cells_global,
        record_size,
    })
}

/// Per-rank header summary used by validators. Avoids loading per-cell
/// tile data — only reads the fixed-size header + v8 trailer (< 300 bytes
/// per file). Returned in rank-id order.
pub struct RankCounts {
    pub path: PathBuf,
    pub rank_id: i32,
    pub num_ranks: i32,
    pub num_cells: i32,
    pub num_cells_global: i32,
}

/// Cheap header-only peek across all rank siblings of `input` (which may be
/// a rank-0 `checkpoint.bin` or its containing directory). Each file is
/// read up to and including the v8 trailer, then closed. Errors out for
/// non-v8 files since this is the only multi-rank format that exposes
/// `num_cells_global`.
pub fn peek_rank_counts(input: &Path) -> Result<Vec<RankCounts>> {
    let paths = discover_rank_files(input)?;
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        let mut f = std::fs::File::open(&p)
            .with_context(|| format!("opening {}", p.display()))?;
        // Read the fixed prefix (44 bytes) so we know sp_sz, then slurp
        // through the v8 trailer in a single follow-up read.
        let mut prefix = [0u8; 44];
        f.read_exact(&mut prefix)
            .with_context(|| format!("reading header of {}", p.display()))?;
        let magic = read_u32(&prefix, 0);
        if magic != MAGIC {
            bail!("{}: bad magic 0x{:08x}", p.display(), magic);
        }
        let version = read_u32(&prefix, 4);
        if version != VERSION_V8 {
            bail!(
                "{}: peek_rank_counts only supports v8 (got v{})",
                p.display(), version
            );
        }
        let num_cells = read_i32(&prefix, 20);
        let sp_sz = read_u32(&prefix, 40) as usize;
        // SimParams payload + tile_t(4) + num_ranks(4) + rank_id(4) + n_global(4)
        let mut rest = vec![0u8; sp_sz + 16];
        f.read_exact(&mut rest)
            .with_context(|| format!("reading v8 trailer of {}", p.display()))?;
        let num_ranks = read_i32(&rest, sp_sz + 4);
        let rank_id = read_i32(&rest, sp_sz + 8);
        let num_cells_global = read_i32(&rest, sp_sz + 12);
        out.push(RankCounts {
            path: p,
            rank_id,
            num_ranks,
            num_cells,
            num_cells_global,
        });
    }
    Ok(out)
}

/// Find all rank files under `input_dir`. Looks for
/// `<dir>/checkpoint.bin` (rank 0) and `<dir>/rankK/checkpoint.bin` for
/// K = 1..num_ranks-1. Returns paths in rank order. Requires that rank 0
/// exists and has a v8 multi-rank header; uses its `num_ranks` to
/// determine how many rank files to expect.
fn discover_rank_files(input: &Path) -> Result<Vec<PathBuf>> {
    // Accept either a directory containing checkpoint.bin or a direct
    // file path; in both cases we resolve to the rank-0 file's directory.
    let (dir, base) = if input.is_dir() {
        (input.to_path_buf(), "checkpoint.bin".to_string())
    } else {
        let parent = input.parent()
            .ok_or_else(|| anyhow!("{}: cannot determine parent dir", input.display()))?
            .to_path_buf();
        let base = input.file_name()
            .ok_or_else(|| anyhow!("{}: no filename component", input.display()))?
            .to_string_lossy()
            .into_owned();
        (parent, base)
    };

    let rank0 = dir.join(&base);
    if !rank0.exists() {
        bail!("rank 0 checkpoint not found at {}", rank0.display());
    }
    let r0 = parse_rank_file(&rank0)?;
    if r0.rank_id != 0 {
        bail!(
            "{}: expected rank_id=0 in supplied file, got {}",
            rank0.display(), r0.rank_id
        );
    }
    let num_ranks = r0.num_ranks;
    if num_ranks <= 1 {
        bail!(
            "{}: checkpoint already has num_ranks={}, nothing to merge.",
            rank0.display(), num_ranks
        );
    }

    let mut paths = vec![rank0];
    for k in 1..num_ranks {
        let p = dir.join(format!("rank{}", k)).join(&base);
        if !p.exists() {
            bail!(
                "expected rank-{} file at {} but it is missing. Cannot merge \
                 an incomplete multi-rank checkpoint.",
                k, p.display()
            );
        }
        paths.push(p);
    }
    Ok(paths)
}

/// Merge per-rank checkpoint files into a single v8 single-rank file.
///
/// `input` is either the path to rank 0's `checkpoint.bin` or its parent
/// directory. `output` is the merged file path. Returns the total cell
/// count written.
pub fn merge_checkpoints(input: &Path, output: &Path) -> Result<usize> {
    let paths = discover_rank_files(input)?;
    eprintln!(
        "merge-ckpt: combining {} rank files -> {}",
        paths.len(), output.display()
    );

    // Parse all ranks.
    let mut ranks: Vec<RankFile> = Vec::with_capacity(paths.len());
    for p in &paths {
        let r = parse_rank_file(p)?;
        eprintln!(
            "  rank {}/{}: {} cells (n_global={})",
            r.rank_id, r.num_ranks, r.cell_ids.len(), r.num_cells_global
        );
        ranks.push(r);
    }

    // Validate consistency.
    let expected_global = ranks[0].num_cells_global;
    let expected_num_ranks = ranks[0].num_ranks;
    let expected_record_size = ranks[0].record_size;
    let mut seen_ranks = vec![false; expected_num_ranks as usize];
    let mut total_cells = 0usize;
    for r in &ranks {
        if r.num_cells_global != expected_global {
            bail!(
                "inconsistent num_cells_global across rank files: rank 0 \
                 has {}, rank {} has {}",
                expected_global, r.rank_id, r.num_cells_global
            );
        }
        if r.num_ranks != expected_num_ranks {
            bail!(
                "inconsistent num_ranks: rank 0 says {}, rank {} says {}",
                expected_num_ranks, r.rank_id, r.num_ranks
            );
        }
        if r.record_size != expected_record_size {
            bail!(
                "rank {} record_size {} disagrees with rank 0's {} \
                 (tile_t mismatch)",
                r.rank_id, r.record_size, expected_record_size
            );
        }
        let ri = r.rank_id as usize;
        if ri >= seen_ranks.len() || seen_ranks[ri] {
            bail!("duplicate or out-of-range rank_id {}", r.rank_id);
        }
        seen_ranks[ri] = true;
        total_cells += r.cell_ids.len();
    }
    if total_cells != expected_global as usize {
        bail!(
            "sum of per-rank cell counts ({}) != num_cells_global ({})",
            total_cells, expected_global
        );
    }

    // Concatenate cells across ranks, then sort by global id.
    let mut by_gid: Vec<(i32, Vec<u8>)> = Vec::with_capacity(total_cells);
    for r in ranks.iter_mut() {
        let recs = std::mem::take(&mut r.cell_records);
        for (gid, rec) in r.cell_ids.iter().zip(recs.into_iter()) {
            by_gid.push((*gid, rec));
        }
    }
    by_gid.sort_by_key(|(gid, _)| *gid);
    // Detect duplicates and -1 sentinels.
    for w in by_gid.windows(2) {
        if w[0].0 == w[1].0 {
            bail!("duplicate global_id {} across rank files", w[0].0);
        }
    }
    if let Some((gid, _)) = by_gid.first() {
        if *gid < 0 {
            bail!(
                "cell has invalid global_id={} (probable migration bookkeeping bug)",
                gid
            );
        }
    }

    // Concatenate sidecars: order matters. We respect rank-0's sidecar
    // ordering and require every other rank to have the same magics in
    // the same order. Per-cell sidecars are reordered by gid; per-cell
    // ordering inside a rank's sidecar is the same as that rank's cell
    // order (verified by index).
    let s0_magics: Vec<u32> = ranks[0].sidecars.iter().map(|s| s.magic).collect();
    for r in &ranks[1..] {
        let mags: Vec<u32> = r.sidecars.iter().map(|s| s.magic).collect();
        if mags != s0_magics {
            bail!(
                "rank {} has different sidecar set/order than rank 0 \
                 ({:?} vs {:?})",
                r.rank_id, mags, s0_magics
            );
        }
    }

    // Build merged sidecars in gid order. For each sidecar slot s,
    // gather (gid, payload_slice) across ranks, sort by gid, write.
    let mut merged_sidecars: Vec<Sidecar> = Vec::with_capacity(s0_magics.len());
    for (s_idx, &m) in s0_magics.iter().enumerate() {
        let bpe = ranks[0].sidecars[s_idx].bytes_per_elem;
        // Validate per-rank counts match per-rank cell counts (otherwise
        // we can't map sidecar entries to cells).
        for r in &ranks {
            let sc = &r.sidecars[s_idx];
            let count = sc.payload.len() / sc.bytes_per_elem;
            if count != r.cell_ids.len() {
                bail!(
                    "rank {} sidecar 0x{:08x} has {} entries but {} cells",
                    r.rank_id, m, count, r.cell_ids.len()
                );
            }
            if sc.bytes_per_elem != bpe {
                bail!(
                    "rank {} sidecar 0x{:08x} bytes_per_elem={} disagrees \
                     with rank 0's {}",
                    r.rank_id, m, sc.bytes_per_elem, bpe
                );
            }
        }
        // Gather (gid, slice) and sort. Use index-into-payload to avoid
        // copying twice.
        let mut indexed: Vec<(i32, &[u8])> = Vec::with_capacity(total_cells);
        for r in &ranks {
            let sc = &r.sidecars[s_idx];
            for (i, gid) in r.cell_ids.iter().enumerate() {
                let start = i * bpe;
                indexed.push((*gid, &sc.payload[start..start + bpe]));
            }
        }
        indexed.sort_by_key(|(gid, _)| *gid);
        let mut payload = Vec::with_capacity(total_cells * bpe);
        for (_, slc) in indexed {
            payload.extend_from_slice(slc);
        }
        merged_sidecars.push(Sidecar {
            magic: m,
            bytes_per_elem: bpe,
            payload,
        });
    }

    // Write the merged file. Start from rank 0's header bytes; rewrite
    // num_cells and the (num_ranks, rank_id, n_global) trailer.
    let r0 = &ranks[0];
    let mut header = r0.header_bytes.clone();
    header[r0.num_cells_offset..r0.num_cells_offset + 4]
        .copy_from_slice(&(total_cells as i32).to_le_bytes());
    header[r0.num_ranks_offset..r0.num_ranks_offset + 4]
        .copy_from_slice(&1i32.to_le_bytes());
    header[r0.num_ranks_offset + 4..r0.num_ranks_offset + 8]
        .copy_from_slice(&0i32.to_le_bytes());
    header[r0.num_ranks_offset + 8..r0.num_ranks_offset + 12]
        .copy_from_slice(&(total_cells as i32).to_le_bytes());

    // Atomic write via .tmp + rename.
    let tmp = output.with_extension("bin.tmp");
    {
        let mut out = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        out.write_all(&header)?;
        for (_, rec) in &by_gid {
            out.write_all(rec)?;
        }
        for sc in &merged_sidecars {
            out.write_all(&sc.magic.to_le_bytes())?;
            let count = (sc.payload.len() / sc.bytes_per_elem) as i32;
            out.write_all(&count.to_le_bytes())?;
            out.write_all(&sc.payload)?;
        }
        out.flush()?;
    }
    std::fs::rename(&tmp, output)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), output.display()))?;
    eprintln!(
        "merge-ckpt: wrote {} cells to {}",
        total_cells, output.display()
    );
    Ok(total_cells)
}
