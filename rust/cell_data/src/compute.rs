use ndarray::Array2;
use std::fs::File;
use std::io::{ BufRead, BufReader };

use crate::read::{ RegionParams };

pub fn compute_center_of_mass_one(data: &Array2<f64>) -> (f64, f64) {
    let (rows, cols) = data.dim();
    let mut total_mass = 0.0;
    let mut x_center = 0.0;
    let mut y_center = 0.0;

    for i in 0..rows {
        for j in 0..cols {
            let mass = data[[i, j]];
            total_mass += mass;
            x_center += (j as f64) * mass;
            y_center += (i as f64) * mass;
        }
    }

    x_center /= total_mass;
    y_center /= total_mass;

    (x_center, y_center)
}

pub fn compute_center_of_mass(data: &Vec<(Vec<(f64, f64)>, Array2<f64>)>) -> Vec<(f64, f64)> {
    data.iter()
        .map(|(intervals, data)| {
            let (x_center, y_center) = compute_center_of_mass_one(data);
            let (x_0, y_0) = (intervals[0].0, intervals[1].0);
            (x_0 + x_center, y_0 + y_center)
        })
        .collect()
}
