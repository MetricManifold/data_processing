//! Colormap definitions

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    VonMises,
    Shear,
    Pressure,
    Plasma,
    Viridis,
    Inferno,
    Magma,
    Turbo,
    Spectral,
    Coolwarm,
    Rainbow,
    Ocean,
    Thermal,
    Grayscale,
}

impl Colormap {
    pub fn all() -> &'static [Colormap] {
        &[
            Colormap::VonMises,
            Colormap::Shear,
            Colormap::Plasma,
            Colormap::Viridis,
            Colormap::Inferno,
            Colormap::Magma,
            Colormap::Turbo,
            Colormap::Spectral,
            Colormap::Coolwarm,
            Colormap::Rainbow,
            Colormap::Ocean,
            Colormap::Thermal,
            Colormap::Pressure,
            Colormap::Grayscale,
        ]
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Colormap::VonMises => "Von Mises",
            Colormap::Shear => "Shear",
            Colormap::Pressure => "Pressure",
            Colormap::Plasma => "Plasma",
            Colormap::Viridis => "Viridis",
            Colormap::Inferno => "Inferno",
            Colormap::Magma => "Magma",
            Colormap::Turbo => "Turbo",
            Colormap::Spectral => "Spectral",
            Colormap::Coolwarm => "Coolwarm",
            Colormap::Rainbow => "Rainbow",
            Colormap::Ocean => "Ocean",
            Colormap::Thermal => "Thermal",
            Colormap::Grayscale => "Grayscale",
        }
    }
    
    pub fn is_diverging(&self) -> bool {
        matches!(self, Colormap::Pressure | Colormap::Coolwarm | Colormap::Spectral)
    }
}

/// Apply colormap to normalized value [0, 1] -> [R, G, B, A]
pub fn apply_colormap(cmap: Colormap, t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    match cmap {
        Colormap::VonMises => von_mises_color(t),
        Colormap::Shear => shear_color(t),
        Colormap::Pressure => pressure_color(t),
        Colormap::Plasma => plasma_color(t),
        Colormap::Viridis => viridis_color(t),
        Colormap::Inferno => inferno_color(t),
        Colormap::Magma => magma_color(t),
        Colormap::Turbo => turbo_color(t),
        Colormap::Spectral => spectral_color(t),
        Colormap::Coolwarm => coolwarm_color(t),
        Colormap::Rainbow => rainbow_color(t),
        Colormap::Ocean => ocean_color(t),
        Colormap::Thermal => thermal_color(t),
        Colormap::Grayscale => grayscale_color(t),
    }
}

/// Generate a colormap preview strip (width x height pixels)
pub fn generate_preview(cmap: Colormap, width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 4];
    for x in 0..width {
        let t = x as f32 / (width - 1) as f32;
        let color = apply_colormap(cmap, t);
        for y in 0..height {
            let idx = (y * width + x) * 4;
            pixels[idx] = color[0];
            pixels[idx + 1] = color[1];
            pixels[idx + 2] = color[2];
            pixels[idx + 3] = 255;
        }
    }
    pixels
}

fn lerp_color(colors: &[[u8; 3]], positions: &[f32], t: f32) -> [u8; 4] {
    for i in 0..positions.len() - 1 {
        if t >= positions[i] && t <= positions[i + 1] {
            let local_t = (t - positions[i]) / (positions[i + 1] - positions[i]);
            let c0 = colors[i];
            let c1 = colors[i + 1];
            return [
                (c0[0] as f32 + (c1[0] as f32 - c0[0] as f32) * local_t) as u8,
                (c0[1] as f32 + (c1[1] as f32 - c0[1] as f32) * local_t) as u8,
                (c0[2] as f32 + (c1[2] as f32 - c0[2] as f32) * local_t) as u8,
                255,
            ];
        }
    }
    let c = colors.last().unwrap();
    [c[0], c[1], c[2], 255]
}

fn von_mises_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 12] = [
        [5, 5, 16], [26, 10, 48], [45, 27, 78], [30, 58, 95],
        [0, 102, 204], [0, 180, 216], [0, 204, 102], [153, 230, 0],
        [255, 204, 0], [255, 102, 0], [255, 0, 0], [255, 255, 255],
    ];
    let positions: [f32; 12] = [0.0, 0.05, 0.12, 0.2, 0.3, 0.4, 0.5, 0.62, 0.72, 0.82, 0.92, 1.0];
    lerp_color(&colors, &positions, t)
}

fn shear_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 14] = [
        [0, 0, 8], [15, 0, 48], [26, 0, 96], [32, 0, 160],
        [0, 64, 255], [0, 160, 255], [0, 224, 224], [0, 255, 128],
        [128, 255, 0], [255, 255, 0], [255, 128, 0], [255, 96, 160],
        [255, 192, 224], [255, 255, 255],
    ];
    let positions: [f32; 14] = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.94, 1.0];
    lerp_color(&colors, &positions, t)
}

fn pressure_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 9] = [
        [8, 48, 107], [33, 113, 181], [66, 146, 198], [107, 174, 214],
        [248, 248, 248],
        [253, 174, 107], [253, 141, 60], [217, 72, 1], [139, 0, 0],
    ];
    let positions: [f32; 9] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    lerp_color(&colors, &positions, t)
}

fn plasma_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 9] = [
        [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
        [204, 71, 120], [232, 107, 84], [248, 149, 64], [252, 194, 36], [240, 249, 33],
    ];
    let positions: [f32; 9] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    lerp_color(&colors, &positions, t)
}

fn viridis_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 9] = [
        [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
        [41, 120, 142], [32, 146, 140], [53, 183, 121], [109, 205, 89], [253, 231, 37],
    ];
    let positions: [f32; 9] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    lerp_color(&colors, &positions, t)
}

fn inferno_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 9] = [
        [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
        [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 255, 164],
    ];
    let positions: [f32; 9] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    lerp_color(&colors, &positions, t)
}

fn magma_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 9] = [
        [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
        [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
    ];
    let positions: [f32; 9] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    lerp_color(&colors, &positions, t)
}

fn turbo_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 11] = [
        [48, 18, 59], [70, 68, 172], [62, 137, 236], [30, 192, 208],
        [53, 224, 138], [147, 244, 78], [213, 226, 45], [254, 188, 43],
        [253, 121, 36], [215, 48, 31], [122, 4, 3],
    ];
    let positions: [f32; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    lerp_color(&colors, &positions, t)
}

fn spectral_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 11] = [
        [94, 79, 162], [50, 136, 189], [102, 194, 165], [171, 221, 164],
        [230, 245, 152], [255, 255, 191], [254, 224, 139], [253, 174, 97],
        [244, 109, 67], [213, 62, 79], [158, 1, 66],
    ];
    let positions: [f32; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    lerp_color(&colors, &positions, t)
}

fn coolwarm_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 9] = [
        [59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249],
        [247, 247, 247],
        [253, 199, 178], [244, 143, 117], [216, 82, 82], [180, 4, 38],
    ];
    let positions: [f32; 9] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    lerp_color(&colors, &positions, t)
}

fn rainbow_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 7] = [
        [128, 0, 255], [0, 0, 255], [0, 255, 255],
        [0, 255, 0], [255, 255, 0], [255, 128, 0], [255, 0, 0],
    ];
    let positions: [f32; 7] = [0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0];
    lerp_color(&colors, &positions, t)
}

fn ocean_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 10] = [
        [0, 8, 32], [0, 24, 64], [0, 48, 96], [0, 80, 128],
        [0, 128, 160], [32, 176, 176], [64, 208, 192], [128, 224, 208],
        [192, 240, 224], [248, 255, 248],
    ];
    let positions: [f32; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0];
    lerp_color(&colors, &positions, t)
}

fn thermal_color(t: f32) -> [u8; 4] {
    let colors: [[u8; 3]; 10] = [
        [0, 0, 0], [16, 0, 48], [48, 0, 96], [96, 0, 128],
        [160, 32, 96], [208, 64, 48], [240, 128, 16], [255, 192, 32],
        [255, 240, 128], [255, 255, 255],
    ];
    let positions: [f32; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0];
    lerp_color(&colors, &positions, t)
}

fn grayscale_color(t: f32) -> [u8; 4] {
    let v = (t * 255.0) as u8;
    [v, v, v, 255]
}

pub fn power_normalize(value: f32, min: f32, max: f32, gamma: f32) -> f32 {
    if max <= min { return 0.5; }
    let normalized = (value - min) / (max - min);
    normalized.clamp(0.0, 1.0).powf(gamma)
}

pub fn symmetric_power_normalize(value: f32, min: f32, max: f32, gamma: f32) -> f32 {
    if max <= min { return 0.5; }
    let normalized = 2.0 * (value - min) / (max - min) - 1.0;
    let sign = normalized.signum();
    let result = sign * normalized.abs().powf(gamma);
    ((result + 1.0) / 2.0).clamp(0.0, 1.0)
}
