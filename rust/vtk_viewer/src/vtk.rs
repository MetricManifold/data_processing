//! VTK file parser for cell simulation output

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Copy, Default)]
pub struct Dimensions {
    pub nx: usize,
    pub ny: usize,
}

#[derive(Debug, Clone)]
pub struct VtkData {
    pub dims: Dimensions,
    pub scalars: HashMap<String, Vec<f32>>,
}

impl VtkData {
    pub fn field_names(&self) -> Vec<&str> {
        self.scalars.keys().map(|s| s.as_str()).collect()
    }
}

pub fn parse_vtk<P: AsRef<Path>>(path: P) -> Result<VtkData> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let mut dims = Dimensions::default();
    let mut n_points = 0usize;
    let mut is_binary = false;
    let mut scalars: HashMap<String, Vec<f32>> = HashMap::new();

    // Parse header (always ASCII in VTK legacy format)
    let mut header_line = String::new();
    let mut current_field_name: Option<String> = None;

    loop {
        header_line.clear();
        let bytes_read = reader.read_line(&mut header_line)?;
        if bytes_read == 0 {
            break;
        }
        let line = header_line.trim();

        if line.starts_with("DIMENSIONS") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                dims.nx = parts[1].parse()?;
                dims.ny = parts[2].parse()?;
            }
        } else if line == "BINARY" {
            is_binary = true;
        } else if line.starts_with("POINT_DATA") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                n_points = parts[1].parse()?;
            }
        } else if line.starts_with("SCALARS") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                current_field_name = Some(parts[1].to_string());
            }
        } else if line.starts_with("LOOKUP_TABLE") {
            // Data starts right after this line
            if is_binary {
                // Read binary data: big-endian f32 values
                if let Some(name) = current_field_name.take() {
                    let mut buf = vec![0u8; n_points * 4];
                    reader.read_exact(&mut buf)?;
                    let data: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|chunk| {
                            f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        })
                        .collect();
                    scalars.insert(name, data);
                }
            }
        } else if !is_binary {
            // ASCII data lines
            if let Some(ref name) = current_field_name {
                if !line.is_empty() && !line.starts_with("VECTORS") {
                    let entry = scalars.entry(name.clone()).or_insert_with(|| Vec::with_capacity(n_points));
                    for val_str in line.split_whitespace() {
                        if let Ok(val) = val_str.parse::<f32>() {
                            entry.push(val);
                        }
                    }
                    if entry.len() >= n_points {
                        current_field_name = None;
                    }
                }
            }
        }
    }

    if n_points == 0 {
        return Err(anyhow!("No POINT_DATA found"));
    }

    Ok(VtkData { dims, scalars })
}

pub fn find_vtk_frames<P: AsRef<Path>>(dir: P) -> Result<Vec<std::path::PathBuf>> {
    let dir = dir.as_ref();
    let mut frames: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map(|e| e == "vtk").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("frame_"))
                    .unwrap_or(false)
        })
        .collect();
    
    frames.sort_by(|a, b| {
        let num_a = extract_frame_number(a);
        let num_b = extract_frame_number(b);
        num_a.cmp(&num_b)
    });
    
    Ok(frames)
}

fn extract_frame_number(path: &Path) -> u32 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("frame_"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}
