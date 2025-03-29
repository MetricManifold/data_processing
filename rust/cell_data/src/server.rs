use base64::encode;
use regex::Regex;
use warp::reject::Reject;
use warp::Filter;
use std::sync::{ Arc, Mutex };
use std::fs;
use std::path::PathBuf;
use serde::{ Deserialize, Serialize };
use std::fmt;

use crate::read::{
    read_checkpoint,
    read_data,
    read_data_slice_3d,
    read_full_checkpoint,
    read_data_dimensions,
    RegionData,
    RegionParams,
    SystemAxis,
};

#[derive(Serialize, Deserialize)]
struct DirectoryListing {
    directories: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct DimensionListing {
    len_x: i32,
    len_y: i32,
    len_z: i32,
}

#[derive(Serialize, Deserialize)]
struct CheckpointImageData {
    index: u32,
    encodings_full: Vec<String>,
    encodings: Vec<String>,
    encodings_border: Vec<String>,
}

#[derive(Debug)]
struct WarpDataError {
    message: &'static str,
}

impl fmt::Display for WarpDataError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Reject for WarpDataError {}
impl Error for WarpDataError {}

impl WarpDataError {
    const LOCK_ERROR: WarpDataError = WarpDataError { message: "Failed to lock current directory" };
    const DIR_READ_ERROR: WarpDataError = WarpDataError { message: "Failed to read the directory" };
    const FILE_READ_ERROR: WarpDataError = WarpDataError {
        message: "Failed to read the data file",
    };
    const DATA_PARSE_ERROR: WarpDataError = WarpDataError { message: "Failed to parse data file" };
}

#[derive(Debug)]
struct CustomSyncError {
    message: &'static str,
}

impl fmt::Display for CustomSyncError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for CustomSyncError {}

#[derive(Serialize, Deserialize)]
struct SetDirectory {
    path: String,
}

#[derive(Clone)]
struct AppState {
    current_directory: Arc<Mutex<PathBuf>>,
}

async fn set_directory(
    new_dir: SetDirectory,
    state: AppState
) -> Result<impl warp::Reply, warp::Rejection> {
    let mut current_directory = state.current_directory.lock().unwrap();
    let msg = format!("Directory set successfully to {:?}", new_dir.path);
    *current_directory = PathBuf::from(new_dir.path);
    Ok(warp::reply::json(&msg))
}

async fn list_directory(state: AppState) -> Result<impl warp::Reply, warp::Rejection> {
    let current_directory = state.current_directory.lock().unwrap();

    println!("Listing directory: {:?}", current_directory);
    let mut directories = Vec::new();

    if current_directory.is_dir() {
        if let Ok(entries) = fs::read_dir(&*current_directory) {
            for entry in entries {
                if let Ok(entry) = entry {
                    if entry.file_type().unwrap().is_dir() {
                        directories.push(entry.file_name().into_string().unwrap());
                    }
                }
            }
        }
    }

    // let listing = DirectoryListing { directories };
    Ok(warp::reply::json(&directories))
}

use ndarray::Array2;
use plotters::prelude::*;
use std::error::Error;
use std::path::Path;

async fn plot_cell(
    data: &Array2<f64>,
    domain: (u32, u32),
    interval: ((u32, u32), (u32, u32)),
    width: u32,
    height: u32
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let (domain_width, domain_height) = domain;
    let ((start_x, end_x), (start_y, end_y)) = interval;
    let (region_width, region_height) = (end_x - start_x + 1, end_y - start_y + 1);

    let mut buffer_full = vec![0u8; (domain_width * domain_height * 4) as usize]; // Initialize buffer with the correct size for RGBA
    let mut buffer_border = vec![0u8; (domain_width * domain_height * 4) as usize]; // Initialize buffer with the correct size for RGBA
    let mut buffer = vec![0u8; (region_width * region_height * 4) as usize]; // Initialize buffer with the correct size for RGBA

    let max_value = data.iter().cloned().fold(f64::MIN, f64::max);
    let min_value = data.iter().cloned().fold(f64::MAX, f64::min);

    for ((y0, x0), &value) in data.indexed_iter() {
        let x = (start_x + (x0 as u32)) % domain_width;
        let y = (start_y + (y0 as u32)) % domain_height;

        let opacity = if value < 0.1 { 0.0 } else if value >= 0.5 { 1.0 } else { value * 2.0 };
        let h_value: f64 = ((value - min_value) as f64) / ((max_value - min_value) as f64);
        let color = HSLColor((1.0 - h_value) * 0.5 + 0.2, 1.0, 0.5).to_rgba();

        let pixel_index_full = ((y * domain_width + x) * 4) as usize;
        buffer_full[pixel_index_full] = ((color.0 as f32) * 255.0) as u8;
        buffer_full[pixel_index_full + 1] = ((color.1 as f32) * 255.0) as u8;
        buffer_full[pixel_index_full + 2] = ((color.2 as f32) * 255.0) as u8;
        buffer_full[pixel_index_full + 3] = (opacity * 255.0) as u8;
        let pixel_index = (y0 * (region_width as usize) + x0) * 4;
        buffer[pixel_index] = ((color.0 as f32) * 255.0) as u8;
        buffer[pixel_index + 1] = ((color.1 as f32) * 255.0) as u8;
        buffer[pixel_index + 2] = ((color.2 as f32) * 255.0) as u8;
        buffer[pixel_index + 3] = (opacity * 255.0) as u8;
    }

    let red = [255u8, 0u8, 0u8, 200u8]; // RGBA for red color
    for x in start_x..=end_x {
        let top_pixel_index = (((start_y % domain_height) * domain_width + (x % domain_width)) *
            4) as usize;
        let bottom_pixel_index = (((end_y % domain_height) * domain_width + (x % domain_width)) *
            4) as usize;
        buffer_border[top_pixel_index..top_pixel_index + 4].copy_from_slice(&red);
        buffer_border[bottom_pixel_index..bottom_pixel_index + 4].copy_from_slice(&red);
    }
    for y in start_y..=end_y {
        let left_pixel_index = (((y % domain_height) * domain_width + (start_x % domain_width)) *
            4) as usize;
        let right_pixel_index = (((y % domain_height) * domain_width + (end_x % domain_width)) *
            4) as usize;
        buffer_border[left_pixel_index..left_pixel_index + 4].copy_from_slice(&red);
        buffer_border[right_pixel_index..right_pixel_index + 4].copy_from_slice(&red);
    }

    let img_full: RgbaImage = ImageBuffer::from_raw(domain_width, domain_height, buffer_full).ok_or(
        "Failed to create image buffer for full sized image"
    )?;
    let mut png_data_full = Vec::new();
    {
        let dimg = image::DynamicImage::ImageRgba8(img_full);
        dimg.write_to(&mut png_data_full, image::ImageOutputFormat::Png)?;
    }

    let img_border: RgbaImage = ImageBuffer::from_raw(
        domain_width,
        domain_height,
        buffer_border
    ).ok_or("Failed to create image buffer for full sized image")?;
    let mut png_data_border = Vec::new();
    {
        let dimg = image::DynamicImage::ImageRgba8(img_border);
        dimg.write_to(&mut png_data_border, image::ImageOutputFormat::Png)?;
    }

    let img: RgbaImage = ImageBuffer::from_raw(region_width, region_height, buffer).ok_or(
        "Failed to create image buffer"
    )?;
    let mut png_data = Vec::new();
    {
        let dimg = image::DynamicImage::ImageRgba8(img);
        dimg.write_to(&mut png_data, image::ImageOutputFormat::Png)?;
    }

    Ok((png_data_full, png_data, png_data_border))
}

async fn plot_heatmap(
    data: &Array2<f32>,
    width: u32,
    height: u32
) -> Result<Vec<u8>, Box<dyn Error>> {
    // let (rows, cols) = data.dim();
    let mut buffer = vec![0u8; (width * height * 3) as usize]; // Initialize buffer with the correct size

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;

        let (rows, cols) = data.dim();
        let max_value = data.iter().cloned().fold(f32::MIN, f32::max);
        let min_value = data.iter().cloned().fold(f32::MAX, f32::min);

        let mut chart = ChartBuilder::on(&root)
            .caption("Heatmap", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(((height as f32) * 0.05).round() as u32)
            .y_label_area_size(((width as f32) * 0.05).round() as u32)
            .top_x_label_area_size(((height as f32) * 0.05).round() as u32)
            .right_y_label_area_size(((width as f32) * 0.05).round() as u32)
            .build_cartesian_2d(0..cols as u32, 0..rows as u32)?;

        chart
            .configure_mesh()
            .x_labels(8)
            .x_label_style(("sans-serif", 20).into_font())
            .y_labels(8)
            .y_label_style(("sans-serif", 20).into_font())
            .draw()?;

        for row in 0..rows {
            for col in 0..cols {
                let value = data[[row, col]];
                let h_value = ((value - min_value) as f64) / ((max_value - min_value) as f64);
                let color = HSLColor((1.0 - h_value) * 0.6, 1.0, 0.5);

                chart.draw_series(
                    std::iter::once(
                        Rectangle::new(
                            [
                                (col as u32, row as u32),
                                ((col as u32) + 1, (row as u32) + 1),
                            ],
                            color.filled()
                        )
                    )
                )?;
            }
        }

        root.present()?;
    }
    Ok(buffer)
}

use tokio::task;
use futures::future;

async fn generate_data_plots(
    file_path: &Path,
    width: u32,
    height: u32
) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let data_list = read_data(file_path)?;
    let mut tasks = Vec::new();
    for data in data_list {
        let data_clone = data.clone();
        let task = task::spawn(async move {
            match plot_heatmap(&data_clone, width, height).await {
                Ok(buffer) => Ok(buffer),
                Err(_) => { Err(Box::new(CustomSyncError { message: "Issue plotting heatmap" })) }
            }
        });
        tasks.push(task);
    }

    let results = future::join_all(tasks).await;

    let mut image_list = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(join_result) =>
                match join_result {
                    Ok(buffer) => image_list.push(buffer),
                    Err(e) => {
                        return Err(e);
                    }
                }
            Err(e) => {
                return Err(Box::new(e));
            }
        }
    }

    Ok(image_list)
}

async fn generate_data_slice(
    file_path: &Path,
    dimension: &SystemAxis,
    slice_index: usize,
    width: u32,
    height: u32
) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let data_list = read_data_slice_3d(file_path, dimension, slice_index)?;
    println!("Data list length: {}", data_list.len());
    let mut tasks = Vec::new();
    for data in data_list {
        let data_clone = data.clone();
        let task = task::spawn(async move {
            match plot_heatmap(&data_clone, width, height).await {
                Ok(buffer) => Ok(buffer),
                Err(_) => { Err(Box::new(CustomSyncError { message: "Issue plotting heatmap" })) }
            }
        });
        tasks.push(task);
    }

    let results = future::join_all(tasks).await;

    let mut image_list = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(join_result) =>
                match join_result {
                    Ok(buffer) => image_list.push(buffer),
                    Err(e) => {
                        return Err(e);
                    }
                }
            Err(e) => {
                return Err(Box::new(e));
            }
        }
    }

    Ok(image_list)
}

use image::{ ImageBuffer, RgbImage, RgbaImage };

async fn get_data_images(
    selected_dir: String,
    selected_field_index: u32,
    state: AppState
) -> Result<impl warp::Reply, warp::Rejection> {
    let path = {
        let current_directory = match state.current_directory.lock() {
            Ok(current_directory) => current_directory,
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::LOCK_ERROR));
            }
        };

        current_directory.join(selected_dir).join("data")
    };

    let path_str = path.display().to_string();
    println!("Path: {}", path_str);

    let regex_pattern = format!(r"data_{}(_\d+)?\.txt", selected_field_index);
    let re = Regex::new(&regex_pattern).map_err(|_|
        warp::reject::custom(WarpDataError::FILE_READ_ERROR)
    )?;

    let plot_height = 800;
    let plot_width = 800;
    if path.exists() {
        let path_name = path.display().to_string();
        let image_data = match fs::read_dir(path) {
            Ok(entries) => {
                let mut path_matches: Vec<(String, PathBuf)> = Vec::new();

                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            let file_name = match entry.file_name().into_string() {
                                Ok(file_name) => file_name,
                                Err(_) => {
                                    return Err(
                                        warp::reject::custom(WarpDataError::FILE_READ_ERROR)
                                    );
                                }
                            };

                            if !re.is_match(&file_name) {
                                continue;
                            }

                            path_matches.push((file_name.clone(), entry.path()));
                        }
                        Err(_) => {
                            return Err(warp::reject::custom(WarpDataError::DIR_READ_ERROR));
                        }
                    }
                }

                if path_matches.is_empty() {
                    return Err(warp::reject::not_found());
                }

                println!("Found {} files matching the pattern", path_matches.len());
                let image_list_of_paths = path_matches.into_iter().map(|(file_name, path)| {
                    async move {
                        // println!("Reading file {file_name} in {path_name}");
                        match generate_data_plots(&path, plot_width, plot_height).await {
                            Ok(buffers) => {
                                let result = buffers
                                    .into_iter()
                                    .map(|buffer| {
                                        let mut png_data = Vec::new();
                                        let buf: RgbImage = ImageBuffer::from_raw(
                                            plot_width,
                                            plot_height,
                                            buffer
                                        ).unwrap();
                                        let dynamic_image = image::DynamicImage::ImageRgb8(buf);
                                        dynamic_image
                                            .write_to(&mut png_data, image::ImageOutputFormat::Png)
                                            .unwrap();
                                        encode(&png_data)
                                    })
                                    .collect::<Vec<_>>();
                                Ok(result)
                            }
                            Err(_) => { Err(warp::reject::custom(WarpDataError::FILE_READ_ERROR)) }
                        }
                    }
                });

                let image_list = future
                    ::join_all(image_list_of_paths).await
                    .into_iter()
                    .map(|result| {
                        match result {
                            Ok(buffers) => Ok(buffers),
                            Err(_) => Err(warp::reject::custom(WarpDataError::FILE_READ_ERROR)),
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                image_list
            }
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::DIR_READ_ERROR));
            }
        };
        if !image_data.is_empty() {
            println!("Returning image data");
            Ok(warp::reply::json(&image_data))
        } else {
            Err(warp::reject::not_found())
        }
    } else {
        Err(warp::reject::not_found())
    }
}

fn get_matching_data_files(
    entries: fs::ReadDir,
    data_file_regex: Regex
) -> Result<Vec<(String, PathBuf)>, warp::Rejection> {
    let mut first_match: Vec<(String, PathBuf)> = Vec::new();

    for entry in entries {
        match entry {
            Ok(entry) => {
                let file_name = match entry.file_name().into_string() {
                    Ok(file_name) => file_name,
                    Err(_) => {
                        return Err(warp::reject::custom(WarpDataError::FILE_READ_ERROR));
                    }
                };

                if !data_file_regex.is_match(&file_name) {
                    continue;
                }

                first_match.push((file_name.clone(), entry.path()));
            }
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::DIR_READ_ERROR));
            }
        }
    }
    Ok(first_match)
}

async fn get_data_slices_of_3d(
    selected_dir: String,
    selected_field_index: u32,
    dimension: SystemAxis,
    slice_index: usize,
    state: AppState
) -> Result<impl warp::Reply, warp::Rejection> {
    let path = {
        let current_directory = match state.current_directory.lock() {
            Ok(current_directory) => current_directory,
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::LOCK_ERROR));
            }
        };

        current_directory.join(selected_dir).join("data")
    };

    let path_str = path.display().to_string();
    println!("Path: {}", path_str);

    let data_file_pat = format!(r"data_{}(_\d+)\.txt", selected_field_index);
    let data_file_regex = Regex::new(&data_file_pat).map_err(|_|
        warp::reject::custom(WarpDataError::FILE_READ_ERROR)
    )?;

    let plot_height = 800;
    let plot_width = 800;
    if path.exists() {
        let path_name = path.display().to_string();
        let image_data = match fs::read_dir(path) {
            Ok(entries) => {
                let data_files = get_matching_data_files(entries, data_file_regex)?;
                let system_dimension = dimension.clone();
                println!("Found {} files matching the pattern", data_files.len());
                let image_list_of_paths = data_files.into_iter().map(|(file_name, path)| {
                    async move {
                        match
                            generate_data_slice(
                                &path,
                                &system_dimension,
                                slice_index,
                                plot_width,
                                plot_height
                            ).await
                        {
                            Ok(buffers) => {
                                let result = buffers
                                    .into_iter()
                                    .map(|buffer| {
                                        let mut png_data = Vec::new();
                                        let buf: RgbImage = ImageBuffer::from_raw(
                                            plot_width,
                                            plot_height,
                                            buffer
                                        ).unwrap();
                                        let dynamic_image = image::DynamicImage::ImageRgb8(buf);
                                        dynamic_image
                                            .write_to(&mut png_data, image::ImageOutputFormat::Png)
                                            .unwrap();
                                        encode(&png_data)
                                    })
                                    .collect::<Vec<_>>();
                                Ok(result)
                            }
                            Err(_) => {
                                return Err(warp::reject::custom(WarpDataError::FILE_READ_ERROR));
                            }
                        }
                    }
                });

                let image_list = future
                    ::join_all(image_list_of_paths).await
                    .into_iter()
                    .map(|result| {
                        match result {
                            Ok(buffers) => Ok(buffers),
                            Err(_) => Err(warp::reject::custom(WarpDataError::FILE_READ_ERROR)),
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                image_list
            }
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::DIR_READ_ERROR));
            }
        };
        if !image_data.is_empty() {
            println!("Returning image data");
            Ok(warp::reply::json(&image_data))
        } else {
            Err(warp::reject::not_found())
        }
    } else {
        Err(warp::reject::not_found())
    }
}

async fn get_system_size(
    selected_dir: String,
    selected_field_index: u32,
    state: AppState
) -> Result<impl warp::Reply, warp::Rejection> {
    let path = {
        let current_directory = match state.current_directory.lock() {
            Ok(current_directory) => current_directory,
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::LOCK_ERROR));
            }
        };

        current_directory.join(selected_dir).join("data")
    };

    let path_str = path.display().to_string();
    println!("Path: {}", path_str);

    let data_file_pat = format!(r"\w+{}\.txt", selected_field_index);
    let data_file_regex = Regex::new(&data_file_pat).map_err(|_|
        warp::reject::custom(WarpDataError::FILE_READ_ERROR)
    )?;

    if path.exists() {
        let path_name = path.display().to_string();
        let dimension_data = match fs::read_dir(path) {
            Ok(entries) => {
                let data_files = get_matching_data_files(entries, data_file_regex)?;
                let dimension_data_list = data_files
                    .into_iter()
                    .map(|(file_name, path)| {
                        println!("Reading file {file_name} in {path_name}");
                        match read_data_dimensions(&path) {
                            Ok((x, y, z)) => {
                                Ok(DimensionListing { len_x: x, len_y: y, len_z: z })
                            }
                            Err(_) => { Err(warp::reject::custom(WarpDataError::FILE_READ_ERROR)) }
                        }
                    })
                    .collect::<Vec<_>>();

                let dimension_data = match dimension_data_list.first() {
                    Some(Ok(dimension_data)) => dimension_data,
                    Some(Err(_)) => {
                        return Err(warp::reject::custom(WarpDataError::FILE_READ_ERROR));
                    }
                    None => {
                        return Err(warp::reject::not_found());
                    }
                };
                DimensionListing {
                    len_x: dimension_data.len_x,
                    len_y: dimension_data.len_y,
                    len_z: dimension_data.len_z,
                }
            }
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::DIR_READ_ERROR));
            }
        };
        println!("Returning dimension data");
        Ok(warp::reply::json(&dimension_data))
    } else {
        Err(warp::reject::not_found())
    }
}

async fn get_checkpoint_images(
    selected_dir: String,
    state: AppState
) -> Result<impl warp::Reply, warp::Rejection> {
    let path = {
        let current_directory = match state.current_directory.lock() {
            Ok(current_directory) => current_directory,
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::LOCK_ERROR));
            }
        };

        current_directory.join(selected_dir).join("checkpoint")
    };

    let path_str = path.display().to_string();
    println!("Path: {}", path_str);

    let re = Regex::new(&r"data(\d+)").map_err(|_|
        warp::reject::custom(WarpDataError::FILE_READ_ERROR)
    )?;

    let plot_height = 800;
    let plot_width = 800;
    if path.exists() {
        let image_data = match fs::read_dir(path) {
            Ok(entries) => {
                let mut tasks = Vec::new();
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            let file_name = entry.file_name();
                            let file_name_str = match file_name.to_str() {
                                Some(file_name_str) => file_name_str,
                                None => {
                                    continue;
                                }
                            };

                            let index = match re.captures(file_name_str) {
                                Some(capture) =>
                                    match capture.get(1).unwrap().as_str().parse::<u32>() {
                                        Ok(index) => index,
                                        Err(_) => {
                                            continue;
                                        }
                                    }
                                None => {
                                    continue;
                                }
                            };

                            let task = task::spawn(async move {
                                let (dp, checkpoint_data) = match
                                    read_full_checkpoint(&entry.path())
                                {
                                    Ok((dp, checkpoint_data)) => (dp, checkpoint_data),
                                    Err(_) => {
                                        return Err(
                                            Box::new(CustomSyncError {
                                                message: "Issue reading checkpoint data",
                                            })
                                        );
                                    }
                                };
                                let widths: Vec<f64> = dp.dims
                                    .iter()
                                    .zip(&dp.intervals)
                                    .map(|(&dim, &(a, b))| (b - a) / ((dim - 1) as f64))
                                    .collect();

                                let mut image_list = Vec::with_capacity(checkpoint_data.len());
                                for rd in checkpoint_data {
                                    let pos = rd.intervals
                                        .iter()
                                        .zip(dp.intervals.iter())
                                        .map(|(&(ra, rb), &(da, db))| (ra - da, rb - da))
                                        .zip(widths.iter())
                                        .map(|((a, b), &width)| (
                                            (a / width) as u32,
                                            (b / width) as u32,
                                        ))
                                        .collect::<Vec<_>>();

                                    let domain = ((dp.dims[0] - 6) as u32, (dp.dims[1] - 6) as u32);
                                    let interval = ((pos[0].0, pos[0].1), (pos[1].0, pos[1].1));
                                    let image_data = match
                                        plot_cell(
                                            &rd.data,
                                            domain,
                                            interval,
                                            plot_width,
                                            plot_height
                                        ).await
                                    {
                                        Ok((buffer_full, buffer, buffer_border)) =>
                                            (
                                                encode(buffer_full),
                                                encode(buffer),
                                                encode(buffer_border),
                                            ),
                                        Err(_) => {
                                            return Err(
                                                Box::new(CustomSyncError {
                                                    message: "Issue plotting heatmap",
                                                })
                                            );
                                        }
                                    };
                                    image_list.push(image_data);
                                }
                                Ok((index, image_list))
                            });
                            tasks.push(task);
                        }
                        Err(_) => {
                            return Err(warp::reject::custom(WarpDataError::DIR_READ_ERROR));
                        }
                    }
                }

                let results = future::join_all(tasks).await;

                let mut image_list = Vec::with_capacity(results.len());
                for result in results {
                    match result {
                        Ok(join_result) =>
                            match join_result {
                                Ok((index, all_encodings)) => {
                                    let (encodings_full, encodings, encodings_border): (
                                        Vec<_>,
                                        Vec<_>,
                                        Vec<_>,
                                    ) = all_encodings
                                        .into_iter()
                                        .fold(
                                            (Vec::new(), Vec::new(), Vec::new()),
                                            |(mut a, mut b, mut c), (x, y, z)| {
                                                a.push(x);
                                                b.push(y);
                                                c.push(z);
                                                (a, b, c)
                                            }
                                        );
                                    image_list.push(CheckpointImageData {
                                        index,
                                        encodings_full,
                                        encodings,
                                        encodings_border,
                                    });
                                }
                                Err(e) => {
                                    return Err(
                                        warp::reject::custom(WarpDataError::DATA_PARSE_ERROR)
                                    );
                                }
                            }
                        Err(e) => {
                            return Err(warp::reject::custom(WarpDataError::DATA_PARSE_ERROR));
                        }
                    }
                }
                image_list
            }
            Err(_) => {
                return Err(warp::reject::custom(WarpDataError::DIR_READ_ERROR));
            }
        };
        if !image_data.is_empty() {
            println!("Returning checkpoint data");
            Ok(warp::reply::json(&image_data))
        } else {
            Err(warp::reject::not_found())
        }
    } else {
        Err(warp::reject::not_found())
    }
}

pub fn build_server() -> impl warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> +
    Clone {
    let state = AppState {
        current_directory: Arc::new(Mutex::new(PathBuf::from("."))),
    };

    let state_filter = warp::any().map(move || state.clone());

    let set_directory_route = warp
        ::path("set_directory")
        .and(warp::post())
        .and(warp::body::json::<SetDirectory>())
        .and(state_filter.clone())
        .and_then(set_directory);

    let list_directory_route = warp
        ::path("list_directory")
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(list_directory);

    let get_system_size_route = warp
        ::path("get_system_size")
        .and(warp::path::param())
        .and(warp::path::param())
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(get_system_size);

    let get_image_route = warp
        ::path("get_image")
        .and(warp::path::param())
        .and(warp::path::param())
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(get_data_images);

    let get_slice_route = warp
        ::path("get_slice")
        .and(warp::path::param())
        .and(warp::path::param())
        .and(warp::path::param())
        .and(warp::path::param())
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(get_data_slices_of_3d);

    let get_checkpoint_route = warp
        ::path("get_fields")
        .and(warp::path::param())
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(get_checkpoint_images);

    // let routes = set_directory_route.or(list_directory_route).or(get_image_route);
    let hi = warp
        ::path("hello")
        .and(warp::path::param())
        .and(warp::header("user-agent"))
        .map(|param: String, agent: String| {
            format!("Hello {}, whose agent is {}", param, agent)
        });

    let routes = hi
        .or(set_directory_route)
        .or(list_directory_route)
        .or(get_system_size_route)
        .or(get_image_route)
        .or(get_slice_route)
        .or(get_checkpoint_route);

    let cors = warp
        ::cors()
        .allow_any_origin()
        .allow_methods(vec!["GET", "POST"])
        .allow_headers(vec!["Content-Type"]);

    return routes.with(cors);
}
