//! Frame rendering

use crate::colormap::{Colormap, apply_colormap, power_normalize, symmetric_power_normalize};
use crate::vtk::VtkData;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GlowColor {
    MatchColormap,
    White,
    Cyan,
    Magenta,
    Gold,
    Red,
    Green,
    Blue,
}

impl GlowColor {
    pub fn all() -> &'static [GlowColor] {
        &[
            GlowColor::MatchColormap,
            GlowColor::White,
            GlowColor::Cyan,
            GlowColor::Magenta,
            GlowColor::Gold,
            GlowColor::Red,
            GlowColor::Green,
            GlowColor::Blue,
        ]
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            GlowColor::MatchColormap => "Match Colormap",
            GlowColor::White => "White",
            GlowColor::Cyan => "Cyan",
            GlowColor::Magenta => "Magenta",
            GlowColor::Gold => "Gold",
            GlowColor::Red => "Red",
            GlowColor::Green => "Green",
            GlowColor::Blue => "Blue",
        }
    }
    
    pub fn rgb(&self) -> Option<[u8; 3]> {
        match self {
            GlowColor::MatchColormap => None,
            GlowColor::White => Some([255, 255, 255]),
            GlowColor::Cyan => Some([0, 255, 255]),
            GlowColor::Magenta => Some([255, 0, 255]),
            GlowColor::Gold => Some([255, 200, 50]),
            GlowColor::Red => Some([255, 80, 80]),
            GlowColor::Green => Some([80, 255, 80]),
            GlowColor::Blue => Some([80, 150, 255]),
        }
    }
}

#[derive(Clone)]
pub struct GlowConfig {
    pub enabled: bool,
    pub intensity: f32,  // 0.0 - 2.0
    pub radius: f32,     // blur radius in pixels (1-10)
    pub color: GlowColor,
}

impl Default for GlowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            radius: 3.0,
            color: GlowColor::MatchColormap,
        }
    }
}

#[derive(Clone)]
pub struct LayerConfig {
    pub name: String,
    pub field: String,
    pub colormap: Colormap,
    pub enabled: bool,
    pub opacity: f32,
    pub gamma: f32,
    pub glow: GlowConfig,
    // Range controls
    pub auto_range: bool,
    pub manual_min: f32,
    pub manual_max: f32,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            name: "Layer".into(),
            field: "phi".into(),
            colormap: Colormap::Grayscale,
            enabled: true,
            opacity: 1.0,
            gamma: 1.0,
            glow: GlowConfig::default(),
            auto_range: true,
            manual_min: 0.0,
            manual_max: 1.0,
        }
    }
}

pub struct RenderConfig {
    pub layers: Vec<LayerConfig>,
    pub background: [u8; 4],
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            layers: vec![
                LayerConfig {
                    name: "Phase Field".into(),
                    field: "phi".into(),
                    colormap: Colormap::Grayscale,
                    enabled: true,
                    opacity: 0.3,
                    gamma: 1.0,
                    glow: GlowConfig::default(),
                    auto_range: true,
                    manual_min: 0.0,
                    manual_max: 1.0,
                },
                LayerConfig {
                    name: "Von Mises".into(),
                    field: "von_mises".into(),
                    colormap: Colormap::VonMises,
                    enabled: true,
                    opacity: 1.0,
                    gamma: 0.5,
                    glow: GlowConfig {
                        enabled: true,
                        intensity: 1.0,
                        radius: 4.0,
                        color: GlowColor::MatchColormap,
                    },
                    auto_range: true,
                    manual_min: 0.0,
                    manual_max: 1.0,
                },
            ],
            background: [26, 26, 46, 255],
        }
    }
}

pub fn render_frame(vtk: &VtkData, config: &RenderConfig) -> Vec<u8> {
    let width = vtk.dims.nx;
    let height = vtk.dims.ny;
    let n = width * height;
    
    let mut pixels = vec![0u8; n * 4];
    for i in 0..n {
        pixels[i * 4] = config.background[0];
        pixels[i * 4 + 1] = config.background[1];
        pixels[i * 4 + 2] = config.background[2];
        pixels[i * 4 + 3] = 255;
    }
    
    let phi_mask: Vec<bool> = vtk.scalars.get("phi")
        .map(|phi| phi.iter().map(|&v| v > 0.1).collect())
        .unwrap_or_else(|| vec![true; n]);
    
    for layer in &config.layers {
        if !layer.enabled { continue; }
        let Some(data) = vtk.scalars.get(&layer.field) else { continue; };
        
        // Use auto or manual range
        let (min, max) = if layer.auto_range {
            percentile_range(data, &phi_mask)
        } else {
            (layer.manual_min, layer.manual_max)
        };
        let is_div = layer.colormap.is_diverging();
        let gamma = layer.gamma;
        let cmap = layer.colormap;
        
        // Compute normalized values for glow intensity
        let normalized: Vec<f32> = data.par_iter().enumerate().map(|(i, &v)| {
            if !phi_mask[i] { return 0.0; }
            if is_div {
                symmetric_power_normalize(v, min, max, gamma)
            } else {
                power_normalize(v, min, max, gamma)
            }
        }).collect();
        
        let layer_px: Vec<[u8; 4]> = normalized.par_iter().enumerate().map(|(i, &t)| {
            if !phi_mask[i] { return [0, 0, 0, 0]; }
            apply_colormap(cmap, t)
        }).collect();
        
        // Blend layer first
        let opacity = (layer.opacity * 255.0) as u8;
        for i in 0..n {
            let src = layer_px[i];
            if src[3] == 0 { continue; }
            let alpha = ((src[3] as u32 * opacity as u32) / 255) as f32 / 255.0;
            let inv = 1.0 - alpha;
            let idx = i * 4;
            pixels[idx] = (src[0] as f32 * alpha + pixels[idx] as f32 * inv) as u8;
            pixels[idx + 1] = (src[1] as f32 * alpha + pixels[idx + 1] as f32 * inv) as u8;
            pixels[idx + 2] = (src[2] as f32 * alpha + pixels[idx + 2] as f32 * inv) as u8;
        }
        
        // Apply glow ON TOP after layer blend (so it glows over the cells)
        if layer.glow.enabled {
            let glow_px = apply_glow(&layer_px, &normalized, width, height, &layer.glow, is_div);
            let glow_opacity = (layer.opacity * layer.glow.intensity * 255.0).min(255.0) as u8;
            blend_additive(&mut pixels, &glow_px, glow_opacity);
        }
    }
    
    // Flip Y
    let mut flipped = vec![0u8; n * 4];
    for y in 0..height {
        let src = (height - 1 - y) * width * 4;
        let dst = y * width * 4;
        flipped[dst..dst + width * 4].copy_from_slice(&pixels[src..src + width * 4]);
    }
    flipped
}

fn apply_glow(layer_px: &[[u8; 4]], normalized: &[f32], width: usize, height: usize, glow: &GlowConfig, is_diverging: bool) -> Vec<[u8; 4]> {
    let n = width * height;
    let radius = glow.radius as i32;
    let kernel_size = (radius * 2 + 1) as usize;
    
    // Build Gaussian kernel
    let sigma = glow.radius / 2.0;
    let mut kernel = vec![0.0f32; kernel_size];
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = (i as i32 - radius) as f32;
        let g = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel[i] = g;
        sum += g;
    }
    for k in &mut kernel { *k /= sum; }
    
    // Weight pixels by how extreme their values are (for glow intensity)
    // Only glow at extremes: top 20% for sequential, both ends for diverging
    let weighted_px: Vec<[f32; 4]> = layer_px.iter().zip(normalized.iter()).map(|(px, &t)| {
        let intensity = if is_diverging {
            // Distance from center (0.5), scaled to 0-1
            let dist_from_center = (t - 0.5).abs() * 2.0; // 0 at center, 1 at extremes
            // Only glow in top 20% of distance (dist > 0.80)
            let threshold = 0.80;
            if dist_from_center > threshold {
                (dist_from_center - threshold) / (1.0 - threshold)
            } else {
                0.0
            }
        } else {
            // Only glow in top 20% of values (t > 0.80)
            let threshold = 0.80;
            if t > threshold {
                (t - threshold) / (1.0 - threshold)
            } else {
                0.0
            }
        };
        // Boost for visible glow effect
        let boost = intensity * 3.0;
        [
            px[0] as f32 * boost,
            px[1] as f32 * boost,
            px[2] as f32 * boost,
            px[3] as f32 * boost,
        ]
    }).collect();
    
    // Horizontal blur
    let mut temp = vec![[0.0f32; 4]; n];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32; 4];
            for k in 0..kernel_size {
                let kx = (x as i32 + k as i32 - radius).clamp(0, width as i32 - 1) as usize;
                let src = weighted_px[y * width + kx];
                let w = kernel[k];
                acc[0] += src[0] * w;
                acc[1] += src[1] * w;
                acc[2] += src[2] * w;
                acc[3] += src[3] * w;
            }
            temp[y * width + x] = acc;
        }
    }
    
    // Vertical blur
    let mut result = vec![[0u8; 4]; n];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32; 4];
            for k in 0..kernel_size {
                let ky = (y as i32 + k as i32 - radius).clamp(0, height as i32 - 1) as usize;
                let src = temp[ky * width + x];
                let w = kernel[k];
                acc[0] += src[0] * w;
                acc[1] += src[1] * w;
                acc[2] += src[2] * w;
                acc[3] += src[3] * w;
            }
            // Apply glow color if not matching colormap
            let color = if let Some(rgb) = glow.color.rgb() {
                // Tint with glow color based on original brightness
                let brightness = (acc[0] + acc[1] + acc[2]) / (3.0 * 255.0);
                [
                    (rgb[0] as f32 * brightness).min(255.0) as u8,
                    (rgb[1] as f32 * brightness).min(255.0) as u8,
                    (rgb[2] as f32 * brightness).min(255.0) as u8,
                    acc[3].min(255.0) as u8,
                ]
            } else {
                [
                    acc[0].min(255.0) as u8,
                    acc[1].min(255.0) as u8,
                    acc[2].min(255.0) as u8,
                    acc[3].min(255.0) as u8,
                ]
            };
            result[y * width + x] = color;
        }
    }
    result
}

fn blend_additive(dst: &mut [u8], glow: &[[u8; 4]], opacity: u8) {
    let scale = opacity as f32 / 255.0;
    for (i, g) in glow.iter().enumerate() {
        if g[3] == 0 { continue; }
        let idx = i * 4;
        let alpha = g[3] as f32 / 255.0 * scale;
        // Additive blend for glow effect
        dst[idx] = (dst[idx] as f32 + g[0] as f32 * alpha).min(255.0) as u8;
        dst[idx + 1] = (dst[idx + 1] as f32 + g[1] as f32 * alpha).min(255.0) as u8;
        dst[idx + 2] = (dst[idx + 2] as f32 + g[2] as f32 * alpha).min(255.0) as u8;
    }
}

fn percentile_range(data: &[f32], mask: &[bool]) -> (f32, f32) {
    let mut vals: Vec<f32> = data.iter().enumerate()
        .filter(|(i, v)| mask[*i] && v.is_finite())
        .map(|(_, &v)| v)
        .collect();
    if vals.is_empty() { return (0.0, 1.0); }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = vals[(vals.len() as f32 * 0.01) as usize];
    let hi = vals[((vals.len() as f32 * 0.99) as usize).min(vals.len() - 1)];
    (lo, hi)
}

pub fn compute_derived_fields(vtk: &mut VtkData) {
    let has = vtk.scalars.contains_key("sigma_xx")
        && vtk.scalars.contains_key("sigma_yy")
        && vtk.scalars.contains_key("sigma_xy");
    if !has { return; }
    
    let n = vtk.scalars.get("sigma_xx").unwrap().len();
    
    // Compute von_mises
    if !vtk.scalars.contains_key("von_mises") {
        let sxx = vtk.scalars.get("sigma_xx").unwrap();
        let syy = vtk.scalars.get("sigma_yy").unwrap();
        let sxy = vtk.scalars.get("sigma_xy").unwrap();
        let vm: Vec<f32> = (0..n).into_par_iter().map(|i| {
            let mean = 0.5 * (sxx[i] + syy[i]);
            let diff = sxx[i] - syy[i];
            let disc = (0.25 * diff * diff + sxy[i] * sxy[i]).sqrt();
            let s1 = mean + disc;
            let s2 = mean - disc;
            (s1 * s1 - s1 * s2 + s2 * s2).sqrt()
        }).collect();
        vtk.scalars.insert("von_mises".into(), vm);
    }
    
    // Compute tau_max
    if !vtk.scalars.contains_key("tau_max") {
        let sxx = vtk.scalars.get("sigma_xx").unwrap();
        let syy = vtk.scalars.get("sigma_yy").unwrap();
        let sxy = vtk.scalars.get("sigma_xy").unwrap();
        let tm: Vec<f32> = (0..n).into_par_iter().map(|i| {
            let diff = sxx[i] - syy[i];
            (0.25 * diff * diff + sxy[i] * sxy[i]).sqrt()
        }).collect();
        vtk.scalars.insert("tau_max".into(), tm);
    }
}
