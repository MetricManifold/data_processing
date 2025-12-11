//! VTK Viewer - High-performance visualization

mod vtk;
mod colormap;
mod render;

use std::collections::HashMap;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use std::path::PathBuf;
use std::io::{BufRead, Write};
use eframe::egui;
use egui::{Color32, ColorImage, TextureHandle, TextureOptions};

use crate::colormap::{Colormap, generate_preview};
use crate::render::{LayerConfig, RenderConfig, GlowColor, compute_derived_fields, render_frame};
use crate::vtk::{VtkData, find_vtk_frames, parse_vtk};

const PREVIEW_WIDTH: usize = 150;
const PREVIEW_HEIGHT: usize = 16;
const MAX_RECENTS: usize = 10;

fn get_recents_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_default()
        .join("recents.txt")
}

fn load_recents() -> Vec<String> {
    let path = get_recents_path();
    if let Ok(file) = std::fs::File::open(&path) {
        std::io::BufReader::new(file)
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .take(MAX_RECENTS)
            .collect()
    } else {
        Vec::new()
    }
}

fn save_recents(recents: &[String]) {
    let path = get_recents_path();
    if let Ok(mut file) = std::fs::File::create(&path) {
        for r in recents.iter().take(MAX_RECENTS) {
            let _ = writeln!(file, "{}", r);
        }
    }
}

fn add_to_recents(recents: &mut Vec<String>, path: &str) {
    // Remove if already present
    recents.retain(|r| r != path);
    // Add to front
    recents.insert(0, path.to_string());
    // Trim to max
    recents.truncate(MAX_RECENTS);
    save_recents(recents);
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("VTK Viewer"),
        ..Default::default()
    };
    eframe::run_native("VTK Viewer", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

/// Message from loader thread
enum LoaderMsg {
    Frame(Arc<VtkData>),
    Done,
    Error(String),
    Progress { loaded: usize, total: usize },
}

struct App {
    frames: Vec<Arc<VtkData>>,
    current_frame: usize,
    playing: bool,
    fps: f32,
    last_frame_time: Instant,
    config: RenderConfig,
    cached_pixels: Option<Vec<u8>>,
    texture: Option<TextureHandle>,
    directory: String,
    error: Option<String>,
    available_fields: Vec<String>,
    frame_dims: (usize, usize),
    colormap_previews: HashMap<String, TextureHandle>,
    // Async loading
    loader_rx: Option<mpsc::Receiver<LoaderMsg>>,
    loading_progress: Option<(usize, usize)>, // (loaded, total)
    // Recents
    recents: Vec<String>,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            frames: Vec::new(),
            current_frame: 0,
            playing: false,
            fps: 30.0,
            last_frame_time: Instant::now(),
            config: RenderConfig::default(),
            cached_pixels: None,
            texture: None,
            directory: String::new(),
            error: None,
            available_fields: Vec::new(),
            frame_dims: (0, 0),
            colormap_previews: HashMap::new(),
            loader_rx: None,
            loading_progress: None,
            recents: load_recents(),
        }
    }
    
    fn ensure_colormap_previews(&mut self, ctx: &egui::Context) {
        for cmap in Colormap::all() {
            let key = cmap.name().to_string();
            if !self.colormap_previews.contains_key(&key) {
                let pixels = generate_preview(*cmap, PREVIEW_WIDTH, PREVIEW_HEIGHT);
                let image = ColorImage::from_rgba_unmultiplied([PREVIEW_WIDTH, PREVIEW_HEIGHT], &pixels);
                let tex = ctx.load_texture(&key, image, TextureOptions::LINEAR);
                self.colormap_previews.insert(key, tex);
            }
        }
    }
    
    fn is_loading(&self) -> bool {
        self.loader_rx.is_some()
    }
    
    fn start_loading(&mut self, path: String) {
        self.error = None;
        self.frames.clear();
        self.cached_pixels = None;
        self.texture = None;
        self.available_fields.clear();
        self.loading_progress = Some((0, 0));
        
        // Add to recents
        add_to_recents(&mut self.recents, &path);
        
        let (tx, rx) = mpsc::channel();
        self.loader_rx = Some(rx);
        
        thread::spawn(move || {
            let files = match find_vtk_frames(&path) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.send(LoaderMsg::Error(format!("{}", e)));
                    return;
                }
            };
            
            if files.is_empty() {
                let _ = tx.send(LoaderMsg::Error("No VTK frames found".to_string()));
                return;
            }
            
            let total = files.len();
            let _ = tx.send(LoaderMsg::Progress { loaded: 0, total });
            
            for (i, vtk_path) in files.iter().enumerate() {
                match parse_vtk(vtk_path) {
                    Ok(mut vtk) => {
                        compute_derived_fields(&mut vtk);
                        if tx.send(LoaderMsg::Frame(Arc::new(vtk))).is_err() {
                            return; // Receiver dropped
                        }
                        let _ = tx.send(LoaderMsg::Progress { loaded: i + 1, total });
                    }
                    Err(e) => {
                        let _ = tx.send(LoaderMsg::Error(format!("Frame {}: {}", i, e)));
                        return;
                    }
                }
            }
            let _ = tx.send(LoaderMsg::Done);
        });
    }
    
    fn poll_loader(&mut self) {
        let Some(rx) = &self.loader_rx else { return };
        
        // Process all available messages
        loop {
            match rx.try_recv() {
                Ok(LoaderMsg::Frame(vtk)) => {
                    // Update fields from first frame
                    if self.frames.is_empty() {
                        self.available_fields = vtk.field_names()
                            .iter().map(|s| s.to_string()).collect();
                        self.available_fields.sort();
                        self.frame_dims = (vtk.dims.nx, vtk.dims.ny);
                    }
                    self.frames.push(vtk);
                    self.cached_pixels = None; // Re-render with new frame
                }
                Ok(LoaderMsg::Progress { loaded, total }) => {
                    self.loading_progress = Some((loaded, total));
                }
                Ok(LoaderMsg::Done) => {
                    self.loader_rx = None;
                    self.loading_progress = None;
                    break;
                }
                Ok(LoaderMsg::Error(e)) => {
                    self.error = Some(e);
                    self.loader_rx = None;
                    self.loading_progress = None;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.loader_rx = None;
                    self.loading_progress = None;
                    break;
                }
            }
        }
    }
    
    fn browse_directory(&mut self) {
        // Start in exe directory, or current directory if that fails
        let start_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Select VTK Directory")
            .set_directory(&start_dir)
            .pick_folder()
        {
            self.directory = path.to_string_lossy().to_string();
            self.start_loading(self.directory.clone());
        }
    }
    
    fn render_current(&mut self) {
        if self.frames.is_empty() { return; }
        let vtk = &self.frames[self.current_frame];
        self.cached_pixels = Some(render_frame(vtk, &self.config));
        self.frame_dims = (vtk.dims.nx, vtk.dims.ny);
    }
    
    fn update_texture(&mut self, ctx: &egui::Context) {
        let Some(pixels) = &self.cached_pixels else { return; };
        let (w, h) = self.frame_dims;
        let image = ColorImage::from_rgba_unmultiplied([w, h], pixels);
        self.texture = Some(ctx.load_texture("frame", image, TextureOptions::LINEAR));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Ensure colormap preview textures exist
        self.ensure_colormap_previews(ctx);
        
        // Poll async loader
        self.poll_loader();
        
        // Request repaint while loading
        if self.is_loading() {
            ctx.request_repaint();
        }
        
        // Playback
        if self.playing && !self.frames.is_empty() {
            let dur = Duration::from_secs_f32(1.0 / self.fps);
            if self.last_frame_time.elapsed() >= dur {
                self.current_frame = (self.current_frame + 1) % self.frames.len();
                self.cached_pixels = None;
                self.last_frame_time = Instant::now();
            }
            ctx.request_repaint();
        }
        
        if self.cached_pixels.is_none() && !self.frames.is_empty() {
            self.render_current();
            self.update_texture(ctx);
        }
        
        // Top panel
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Browse button
                let browse_enabled = !self.is_loading();
                if ui.add_enabled(browse_enabled, egui::Button::new("📂 Browse")).clicked() {
                    self.browse_directory();
                }
                
                // Recents dropdown
                if !self.recents.is_empty() {
                    let recents_enabled = !self.is_loading();
                    ui.add_enabled_ui(recents_enabled, |ui| {
                        egui::ComboBox::from_id_salt("recents")
                            .selected_text("📋 Recent")
                            .width(100.0)
                            .show_ui(ui, |ui| {
                                let mut selected = None;
                                for recent in &self.recents {
                                    // Show just the last folder name for brevity
                                    let display = std::path::Path::new(recent)
                                        .file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_else(|| recent.clone());
                                    if ui.selectable_label(false, &display)
                                        .on_hover_text(recent)
                                        .clicked() 
                                    {
                                        selected = Some(recent.clone());
                                    }
                                }
                                if let Some(path) = selected {
                                    self.directory = path.clone();
                                    self.start_loading(path);
                                }
                            });
                    });
                }
                
                ui.separator();
                ui.label("Path:");
                let text_enabled = !self.is_loading();
                ui.add_enabled(text_enabled, egui::TextEdit::singleline(&mut self.directory).desired_width(350.0));
                
                let load_enabled = !self.is_loading() && !self.directory.is_empty();
                if ui.add_enabled(load_enabled, egui::Button::new("Load")).clicked() {
                    let p = self.directory.clone();
                    self.start_loading(p);
                }
                
                // Loading progress
                if let Some((loaded, total)) = self.loading_progress {
                    ui.separator();
                    ui.spinner();
                    ui.label(format!("Loading {}/{}", loaded, total));
                } else if !self.frames.is_empty() {
                    ui.separator();
                    ui.label(format!("{} frames", self.frames.len()));
                }
                
                if let Some(e) = &self.error {
                    ui.separator();
                    ui.colored_label(Color32::RED, e);
                }
            });
        });
        
        // Left panel - layers
        egui::SidePanel::left("layers").min_width(250.0).show(ctx, |ui| {
            ui.heading("Layers");
            ui.separator();
            
            let mut changed = false;
            for (i, layer) in self.config.layers.iter_mut().enumerate() {
                ui.push_id(i, |ui| {
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut layer.enabled, "").changed() { changed = true; }
                        ui.strong(&layer.name);
                    });
                    if layer.enabled {
                        ui.indent(i, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Field:");
                                egui::ComboBox::from_id_salt(format!("f{}", i))
                                    .selected_text(&layer.field)
                                    .show_ui(ui, |ui| {
                                        for f in &self.available_fields {
                                            if ui.selectable_value(&mut layer.field, f.clone(), f).changed() {
                                                changed = true;
                                            }
                                        }
                                    });
                            });
                            ui.horizontal(|ui| {
                                ui.label("Colormap:");
                                egui::ComboBox::from_id_salt(format!("c{}", i))
                                    .selected_text(layer.colormap.name())
                                    .width(180.0)
                                    .show_ui(ui, |ui| {
                                        for c in Colormap::all() {
                                            let response = ui.horizontal(|ui| {
                                                // Show preview
                                                if let Some(tex) = self.colormap_previews.get(c.name()) {
                                                    ui.image((tex.id(), egui::vec2(PREVIEW_WIDTH as f32, PREVIEW_HEIGHT as f32)));
                                                }
                                                ui.selectable_value(&mut layer.colormap, *c, c.name())
                                            }).inner;
                                            if response.changed() {
                                                changed = true;
                                            }
                                        }
                                    });
                            });
                            // Show current colormap preview
                            if let Some(tex) = self.colormap_previews.get(layer.colormap.name()) {
                                ui.horizontal(|ui| {
                                    ui.add_space(50.0);
                                    ui.image((tex.id(), egui::vec2(PREVIEW_WIDTH as f32, PREVIEW_HEIGHT as f32)));
                                });
                            }
                            ui.horizontal(|ui| {
                                ui.label("Opacity:");
                                if ui.add(egui::Slider::new(&mut layer.opacity, 0.0..=1.0)).changed() {
                                    changed = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Gamma:");
                                if ui.add(egui::Slider::new(&mut layer.gamma, 0.1..=2.0)).changed() {
                                    changed = true;
                                }
                            });
                            
                            // Range controls
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut layer.auto_range, "").changed() { changed = true; }
                                ui.label("Auto Range");
                            });
                            if !layer.auto_range {
                                ui.indent(format!("range{}", i), |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Min:");
                                        if ui.add(egui::DragValue::new(&mut layer.manual_min).speed(0.001)).changed() {
                                            changed = true;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Max:");
                                        if ui.add(egui::DragValue::new(&mut layer.manual_max).speed(0.001)).changed() {
                                            changed = true;
                                        }
                                    });
                                });
                            }
                            
                            // Glow controls
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut layer.glow.enabled, "").changed() { changed = true; }
                                ui.label("Glow");
                            });
                            if layer.glow.enabled {
                                ui.indent(format!("glow{}", i), |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Intensity:");
                                        if ui.add(egui::Slider::new(&mut layer.glow.intensity, 0.1..=2.0)).changed() {
                                            changed = true;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Radius:");
                                        if ui.add(egui::Slider::new(&mut layer.glow.radius, 1.0..=10.0)).changed() {
                                            changed = true;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Color:");
                                        egui::ComboBox::from_id_salt(format!("gc{}", i))
                                            .selected_text(layer.glow.color.name())
                                            .show_ui(ui, |ui| {
                                                for c in GlowColor::all() {
                                                    if ui.selectable_value(&mut layer.glow.color, *c, c.name()).changed() {
                                                        changed = true;
                                                    }
                                                }
                                            });
                                    });
                                });
                            }
                        });
                    }
                    ui.add_space(8.0);
                });
            }
            
            if changed { self.cached_pixels = None; }
            
            ui.separator();
            if ui.button("+ Add Layer").clicked() {
                self.config.layers.push(LayerConfig::default());
            }
        });
        
        // Bottom - timeline
        egui::TopBottomPanel::bottom("timeline").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button(if self.playing { "⏸" } else { "▶" }).clicked() {
                    self.playing = !self.playing;
                    self.last_frame_time = Instant::now();
                }
                
                let max = self.frames.len().saturating_sub(1);
                let old = self.current_frame;
                ui.add(egui::Slider::new(&mut self.current_frame, 0..=max).text("Frame"));
                if self.current_frame != old { self.cached_pixels = None; }
                
                ui.separator();
                ui.label("FPS:");
                ui.add(egui::Slider::new(&mut self.fps, 1.0..=120.0).logarithmic(true));
                
                if !self.frames.is_empty() {
                    ui.separator();
                    ui.label(format!("{}/{}", self.current_frame + 1, self.frames.len()));
                }
            });
        });
        
        // Center - image
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tex) = &self.texture {
                let avail = ui.available_size();
                let tsz = tex.size_vec2();
                let scale = (avail.x / tsz.x).min(avail.y / tsz.y);
                let dsz = tsz * scale;
                let off = (avail - dsz) / 2.0;
                ui.add_space(off.y);
                ui.horizontal(|ui| {
                    ui.add_space(off.x);
                    ui.image((tex.id(), dsz));
                });
            } else if self.is_loading() {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.spinner();
                        if let Some((loaded, total)) = self.loading_progress {
                            ui.label(format!("Loading frames: {}/{}", loaded, total));
                            if total > 0 {
                                let progress = loaded as f32 / total as f32;
                                ui.add(egui::ProgressBar::new(progress).show_percentage());
                            }
                        }
                    });
                });
            } else if self.frames.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.label("Click Browse to select a VTK directory");
                        ui.add_space(10.0);
                        ui.label("Or paste a path and click Load");
                    });
                });
            }
        });
    }
}
