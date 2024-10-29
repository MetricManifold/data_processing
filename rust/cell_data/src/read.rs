use ndarray::Array2;
use std::fs::File;
use std::io::{ BufRead, BufReader };
use std::num::ParseFloatError;
use std::path::Path;

pub struct RegionParams {
    dimension: usize,
    dims: Vec<i32>,
    intervals: Vec<(f64, f64)>,
}

pub struct RegionData {
    pub time_step: i32,
    pub intervals: Vec<(f64, f64)>,
    pub data: Array2<f64>,
}

fn get_region_params(
    values: Vec<f64>,
    widths: Vec<f64>
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
        .map(|(&width, &(a, b))| ((b - a) / width + 1f64) as i32)
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
    let intervals = rest_values[dimension..dimension + dimension * 2]
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();

    let dp = RegionParams {
        dimension,
        dims,
        intervals,
    };
    let (time_step, rp) = get_region_params(
        rest_values[dimension * 3..].to_vec(),
        dp.dims
            .iter()
            .zip(&dp.intervals)
            .map(|(&dim, &(a, b))| (b - a) / (dim as f64))
            .collect()
    )?;
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
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
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

pub fn read_data(
    time_step: i32,
    file_path: &Path
) -> Result<RegionData, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);

    let mut buf = String::new();
    while reader.read_line(&mut buf)? > 0 {
        let values = parse_line_values(&buf)?;
        let (time_step_0, rp, dp) = get_domain_params(values)?;
        let data: ndarray::ArrayBase<
            ndarray::OwnedRepr<f64>,
            ndarray::Dim<[usize; 2]>
        > = read_region_data(&mut reader, &rp)?;
        if time_step_0 == time_step {
            return Ok(RegionData { time_step: time_step, intervals: rp.intervals, data: data });
        }
        buf.clear();
    }
    Err(Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "Time step not found")))
}
