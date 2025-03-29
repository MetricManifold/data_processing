use ndarray::Array2;
use std::error::Error;
use std::fs::File;
use std::io::{ BufRead, BufReader };
use std::num::ParseFloatError;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum SystemAxis {
    X,
    Y,
    Z,
}

use std::str::FromStr;
impl FromStr for SystemAxis {
    type Err = ();

    fn from_str(input: &str) -> Result<SystemAxis, Self::Err> {
        match input {
            "X" => Ok(SystemAxis::X),
            "Y" => Ok(SystemAxis::Y),
            "Z" => Ok(SystemAxis::Z),
            _ => Err(()),
        }
    }
}

#[derive(Clone)]
pub struct RegionParams {
    pub dimension: usize,
    pub dims: Vec<i32>,
    pub intervals: Vec<(f64, f64)>,
}

#[derive(Clone)]
pub struct RegionData {
    pub time_step: i32,
    pub intervals: Vec<(f64, f64)>,
    pub data: Array2<f64>,
}

fn get_region_params(
    values: &Vec<f64>,
    widths: &Vec<f64>
) -> Result<(i32, RegionParams), std::io::Error> {
    if values.len() < 1 {
        return Err(
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Not enough values to read region parameters"
            )
        );
    }

    let time_step: i32 = values[0] as i32;
    let dimension = values[1] as usize;
    let intervals = values[2..2 + dimension * 2]
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();
    let dims = widths
        .iter()
        .zip(&intervals)
        .map(|(width, &(a, b))| ((b - a) / width + 1f64) as i32)
        .collect();

    let rp = RegionParams {
        dimension,
        dims,
        intervals,
    };
    Ok((time_step, rp))
}

fn get_domain_params(
    values: Vec<f64>
) -> Result<(i32, RegionParams, RegionParams), std::io::Error> {
    if values.len() < 1 {
        return Err(
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Not enough values to parse region parameters"
            )
        );
    }

    let dimension = values[0] as usize;
    let rest_values = &values[1..];
    let dims = rest_values[0..dimension]
        .iter()
        .map(|&x| x as i32)
        .collect();
    let intervals = rest_values[dimension..dimension * 3]
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();

    let dp = RegionParams {
        dimension,
        dims,
        intervals,
    };

    let widths = dp.dims
        .iter()
        .zip(&dp.intervals)
        .map(|(&dim, &(a, b))| (b - a) / ((dim - 1) as f64))
        .collect();
    let (time_step, rp) = get_region_params(&rest_values[dimension * 3..].to_vec(), &widths)?;
    Ok((time_step, rp, dp))
}

fn parse_line_values(line: &str) -> Result<Vec<f64>, ParseFloatError> {
    line.split_whitespace()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
}

fn read_region_data(
    reader: &mut BufReader<File>,
    rp: &RegionParams
) -> Result<Array2<f64>, Box<dyn Error>> {
    let (rows, cols) = (rp.dims[1] as usize, rp.dims[0] as usize);
    let mut flattened_data = Vec::with_capacity(rows * cols);

    for _ in 0..rows {
        let mut buf = String::new();
        reader.read_line(&mut buf)?;
        let values = parse_line_values(&buf)?;
        if values.len() != cols {
            return Err(
                std::io::Error
                    ::new(std::io::ErrorKind::InvalidInput, "Invalid number of values in row")
                    .into()
            );
        }
        flattened_data.extend(values);
    }

    // Create a 2D array from the flattened data
    let array = Array2::from_shape_vec((rows, cols), flattened_data)?;
    Ok(array)
}

/// Reads the checkpoint-formatted data from a file. This assumes that the checkpoint file contains
/// only a single frame in time.
pub fn read_checkpoint(time_step: i32, file_path: &Path) -> Result<RegionData, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);

    let mut buf = String::new();
    while reader.read_line(&mut buf)? > 0 {
        let values = parse_line_values(&buf)?;
        let (time_step_0, rp, _) = get_domain_params(values)?;
        let data = read_region_data(&mut reader, &rp)?;
        if time_step_0 == time_step {
            return Ok(RegionData {
                time_step: time_step,
                intervals: rp.intervals,
                data: data,
            });
        }
        buf.clear();
    }
    Err(
        Box::new(
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Time step {} not found at {}", time_step, file_path.to_str().unwrap())
            )
        )
    )
}

/// Reads the checkpoint-formatted data from a file. This assumes that the checkpoint file contains
/// only a single frame in time.
pub fn read_full_checkpoint(
    file_path: &Path
) -> Result<(RegionParams, Vec<RegionData>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);

    let mut buf = String::new();
    reader.read_line(&mut buf)?;

    let values = parse_line_values(&buf)?;
    let (time_step, rp, dp) = get_domain_params(values)?;
    let data = read_region_data(&mut reader, &rp)?;
    let rd = RegionData {
        time_step: time_step,
        intervals: rp.intervals,
        data: data,
    };
    let mut checkpoint_data = vec![rd];

    let widths = dp.dims
        .iter()
        .zip(&dp.intervals)
        .map(|(&dim, &(a, b))| (b - a) / ((dim - 1) as f64))
        .collect();

    buf.clear();
    while reader.read_line(&mut buf)? > 0 {
        let values = parse_line_values(&buf)?;
        let (time_step, rp) = get_region_params(&values, &widths)?;
        let data = read_region_data(&mut reader, &rp)?;
        let rd = RegionData {
            time_step: time_step,
            intervals: rp.intervals,
            data: data,
        };
        checkpoint_data.push(rd);
        buf.clear();
    }
    Ok((dp, checkpoint_data))
}

/// Reads the phase-field data file. This is saved in a structured matrix output format.
pub fn read_data(file_path: &Path) -> Result<Vec<Array2<f32>>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // Read the entire file content into a string
    let content: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");

    // Split the content by two newlines to separate the matrices
    let matrices_str: Vec<&str> = content.split("\n\n").collect();

    let mut matrices = Vec::new();

    for matrix_str in matrices_str {
        let mut rows = Vec::new();
        for (i, line) in matrix_str.lines().enumerate() {
            if i == 0 {
                continue;
            }

            let values: Vec<f32> = line
                .split_whitespace()
                .skip(1)
                .map(|s| s.parse::<f32>())
                .collect::<Result<Vec<_>, _>>()?;
            rows.push(values);
        }
        // Convert the rows to a 2D array
        let rows_len = rows.len();
        let cols_len = rows[0].len();
        let flattened_data: Vec<f32> = rows.into_iter().flatten().collect();
        let array = Array2::from_shape_vec((rows_len, cols_len), flattened_data)?;
        matrices.push(array);
    }

    Ok(matrices)
}

pub fn read_data_dimensions(file_path: &Path) -> Result<(i32, i32, i32), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // we want to read frame by frame so that we end up only reading in one slice for each time frame.

    let mut max_row_count: i32 = 0;
    let mut row_count: i32 = 0;
    let mut x_axis: Vec<f32> = Vec::new();
    let mut current_z_layer: i32 = 0;

    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let tokens: Vec<f32> = line
                .split_whitespace()
                .map(|t| {
                    match String::from(t).parse::<f32>() {
                        Ok(v) => Ok(v),
                        Err(e) => Err(e),
                    }
                })
                .collect::<Result<Vec<f32>, _>>()?;
            if row_count == 0 {
                if let Some((&z_layer, x_header)) = tokens.split_first() {
                    let z_layer = z_layer as i32;
                    current_z_layer = z_layer;
                    x_axis = x_header.to_vec();
                }
            }
            row_count += 1;
        } else {
            max_row_count = max_row_count.max(row_count - 1);
            row_count = 0;
        }
    }

    Ok((x_axis.len() as i32, max_row_count, current_z_layer + 1))
}

/// Reads the phase-field data file. This is saved in a structured matrix output format.
pub fn read_data_slice_3d(
    file_path: &Path,
    dimension: &SystemAxis,
    index: usize
) -> Result<Vec<Array2<f32>>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // we want to read frame by frame so that we end up only reading in one slice for each time frame.

    let mut buffer: Vec<Vec<f32>> = Vec::new();
    let mut x_axis: Vec<f32> = Vec::new();
    let mut current_z_layer: usize = 0;
    let mut slice: Vec<Vec<f32>> = Vec::new();

    let mut all_slices: Vec<Array2<f32>> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            // now we can hanlde the buffer based on the dimension and index
            match dimension {
                SystemAxis::X => {
                    // get all the x values at index from the buffer:
                    let slice_column = buffer
                        .iter()
                        .map(|row| row[index])
                        .collect::<Vec<f32>>();
                    slice.push(slice_column);
                }
                SystemAxis::Y => {
                    slice.push(buffer[index].clone());
                }
                SystemAxis::Z => {
                    if current_z_layer == index {
                        slice = buffer.clone();
                    }
                }
            }
            buffer.clear();
            x_axis.clear();
        } else {
            let tokens: Vec<f32> = line
                .split_whitespace()
                .map(|t| {
                    match String::from(t).parse::<f32>() {
                        Ok(v) => Ok(v),
                        Err(e) => Err(e),
                    }
                })
                .collect::<Result<Vec<f32>, _>>()?;
            if x_axis.is_empty() {
                if let Some((&z_layer, x_header)) = tokens.split_first() {
                    let z_layer = z_layer as usize;
                    if z_layer < current_z_layer {
                        let rows_len = slice.len();
                        let cols_len = slice[0].len();
                        let flattened_data: Vec<f32> = slice.into_iter().flatten().collect();
                        let array = Array2::from_shape_vec((rows_len, cols_len), flattened_data)?;
                        all_slices.push(array);
                        slice = Vec::with_capacity(rows_len);
                    }
                    current_z_layer = z_layer;
                    x_axis = x_header.to_vec();
                }
            } else {
                if let Some((&y_coord, data)) = tokens.split_first() {
                    buffer.push(data.to_vec());
                }
            }
        }
    }
    let rows_len = slice.len();
    let cols_len = slice[0].len();
    let flattened_data: Vec<f32> = slice.into_iter().flatten().collect();
    let array = Array2::from_shape_vec((rows_len, cols_len), flattened_data)?;
    all_slices.push(array);

    Ok(all_slices)
}

pub fn read_com_file(
    file_path: &Path
) -> Result<Vec<(i32, u32, (f64, f64))>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut com_data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line
            .split(',')
            .map(|s| s.trim())
            .collect();
        if parts.len() == 4 {
            let time_step: i32 = parts[0].parse()?;
            let index: u32 = parts[1].parse()?;
            let x: f64 = parts[2].parse()?;
            let y: f64 = parts[3].parse()?;
            com_data.push((time_step, index, (x, y)));
        }
    }

    Ok(com_data)
}
