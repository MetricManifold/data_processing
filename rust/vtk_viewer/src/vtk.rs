//! VTK file parser for cell simulation output

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
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
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    let mut dims = Dimensions::default();
    let mut n_points = 0usize;
    let mut scalars: HashMap<String, Vec<f32>> = HashMap::new();
    
    // Parse header
    while let Some(line) = lines.next() {
        let line = line?;
        let line = line.trim();
        
        if line.starts_with("DIMENSIONS") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                dims.nx = parts[1].parse()?;
                dims.ny = parts[2].parse()?;
            }
        } else if line.starts_with("POINT_DATA") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                n_points = parts[1].parse()?;
            }
            break;
        }
    }
    
    if n_points == 0 {
        return Err(anyhow!("No POINT_DATA found"));
    }
    
    // Parse scalar fields
    let mut current_field: Option<String> = None;
    let mut current_data: Vec<f32> = Vec::new();
    
    for line in lines {
        let line = line?;
        let line = line.trim();
        
        if line.starts_with("SCALARS") {
            if let Some(name) = current_field.take() {
                if current_data.len() == n_points {
                    scalars.insert(name, std::mem::take(&mut current_data));
                }
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                current_field = Some(parts[1].to_string());
                current_data = Vec::with_capacity(n_points);
            }
        } else if line.starts_with("LOOKUP_TABLE") {
            continue;
        } else if line.starts_with("VECTORS") {
            if let Some(name) = current_field.take() {
                if current_data.len() == n_points {
                    scalars.insert(name, std::mem::take(&mut current_data));
                }
            }
            current_field = None;
        } else if current_field.is_some() && !line.is_empty() {
            for val_str in line.split_whitespace() {
                if let Ok(val) = val_str.parse::<f32>() {
                    current_data.push(val);
                }
            }
        }
    }
    
    if let Some(name) = current_field {
        if current_data.len() == n_points {
            scalars.insert(name, current_data);
        }
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
