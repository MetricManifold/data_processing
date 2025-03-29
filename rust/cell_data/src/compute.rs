// use itertools::Itertools;
use ndarray::Array2;
// use std::fs::File;
// use std::io::{ BufRead, BufReader };
// use std::path::Display;
// use std::vec;

// use crate::read::{ RegionParams };

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
    data.into_iter()
        .map(|(intervals, data)| {
            let (x_center, y_center) = compute_center_of_mass_one(data);
            let (x_0, y_0) = (intervals[0].0, intervals[1].0);
            (x_0 + x_center, y_0 + y_center)
        })
        .collect()
}

use std::collections::HashMap;

pub fn calculate_time_dep_msd_particles(
    centers_of_mass: &[(i32, u32, (f64, f64))]
) -> Result<Vec<(i32, u32, f64)>, Box<dyn std::error::Error>> {
    let positions = centers_of_mass
        .into_iter()
        .fold(HashMap::new(), |mut acc, &(time_step, index, (x, y))| {
            acc.entry(time_step).or_insert_with(HashMap::new).insert(index, (x, y));
            acc
        });

    let mut time_steps: Vec<_> = positions.keys().cloned().collect::<Vec<_>>();
    time_steps.sort();

    let mut msd: Vec<(i32, u32, f64)> = Vec::new();
    let initial_time_step = &time_steps[0];
    let initial_positions = &positions[initial_time_step];

    for i in 1..time_steps.len() {
        let current_time_step = &time_steps[i];
        let current_positions = &positions[current_time_step];

        for (index, (x, y)) in initial_positions {
            let (x0, y0) = match current_positions.get(index) {
                Some(pos) => pos,
                None => Err("Missing previous position")?,
            };

            let dx = x - x0;
            let dx_p = dx.abs() - 1600.0;
            let dy = y - y0;
            let dy_p = dy.abs() - 1600.0;
            let displacement = (dx * dx).min(dx_p * dx_p) + (dy * dy).min(dy_p * dy_p);
            msd.push((*current_time_step, *index, displacement));
        }
    }

    // let mut msd: Vec<(i32, u32, f64)> = Vec::new();
    // for i in 1..time_steps.len() {
    //     let current_time_step = &time_steps[i];
    //     let current_positions = match positions.get(current_time_step) {
    //         Some(pos) => pos,
    //         None => Err("Missing current positions")?,
    //     };

    //     for (index, (x, y)) in current_positions {
    //         let displacements: Result<Vec<_>, Box<dyn std::error::Error>> = (0..i)
    //             .map(|j| {
    //                 let previous_time_step = &time_steps[j];
    //                 let previous_positions = match positions.get(previous_time_step) {
    //                     Some(pos) => pos,
    //                     None => Err("Missing previous positions")?,
    //                 };

    //                 let (x0, y0) = match previous_positions.get(index) {
    //                     Some(pos) => pos,
    //                     None => Err("Missing previous position")?,
    //                 };

    //                 let dx = x - x0;
    //                 let dx_p = dx.abs() - 1600.0;
    //                 let dy = y - y0;
    //                 let dy_p = dy.abs() - 1600.0;
    //                 let displacement = (dx * dx).min(dx_p * dx_p) + (dy * dy).min(dy_p * dy_p);
    //                 Ok(displacement)
    //             })
    //             .collect();
    //         let displacement = displacements?.into_iter().sum::<f64>() / (i as f64);
    //         msd.push((*current_time_step, *index, displacement));
    //     }
    // }

    // let time_deltas = time_steps
    //     .iter()
    //     .take(time_steps.len() - 1)
    //     .zip(time_steps.iter().skip(1));

    // let mut msd: Vec<(i32, u32, f64)> = Vec::new();
    // for (previous_time_step, current_time_step) in time_deltas {
    //     let previous_positions = match positions.get(previous_time_step) {
    //         Some(pos) => pos,
    //         None => Err("Missing previous positions")?,
    //     };
    //     let current_positions = match positions.get(current_time_step) {
    //         Some(pos) => pos,
    //         None => Err("Missing current positions")?,
    //     };

    //     for (index, (x0, y0)) in previous_positions {
    //         let (x, y) = match current_positions.get(index) {
    //             Some(pos) => pos,
    //             None => Err("Missing current position")?,
    //         };
    //         let dx = x - x0;
    //         let dx_p = dx.abs() - 1600.0;
    //         let dy = y - y0;
    //         let dy_p = dy.abs() - 1600.0;
    //         let displacement = (dx * dx).min(dx_p * dx_p) + (dy * dy).min(dy_p * dy_p);
    //         msd.push((*current_time_step, *index, displacement));
    //     }
    // }

    msd.sort_by(|(a0, a1, _), (b0, b1, _)|
        a0.partial_cmp(b0).unwrap_or(a1.partial_cmp(b1).unwrap_or(std::cmp::Ordering::Equal))
    );

    // let mut msd_running = Vec::new();
    // let mut sum = 0.0;
    // for i in 0..msd.len() {
    //     sum += msd[i].2;
    //     msd_running.push((msd[i].0, msd[i].1, sum / ((i + 1) as f64)));
    // }

    Ok(msd)
}

pub fn calculate_time_dep_msd(
    centers_of_mass: &[(i32, u32, (f64, f64))]
) -> Result<Vec<(i32, f64)>, Box<dyn std::error::Error>> {
    let msd_particles = calculate_time_dep_msd_particles(centers_of_mass)?;
    let msd = msd_particles
        .into_iter()
        .fold(HashMap::new(), |mut acc, (time_step, _, displacement)| {
            let msd_list = acc.entry(time_step).or_insert(Vec::new());
            msd_list.push(displacement);
            acc
        })
        .into_iter()
        .fold(Vec::new(), |mut acc, (time_step, msd_list)| {
            let msd_sum = msd_list.iter().fold(0.0, |acc, displacement| acc + displacement);
            let msd_count = msd_list.len() as u32;
            acc.push((time_step, msd_sum / (msd_count as f64)));
            acc
        });

    Ok(msd)
}
