use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::read::{ read_checkpoint, read_com_file };

use crate::compute::{
    calculate_time_dep_msd,
    calculate_time_dep_msd_particles,
    compute_center_of_mass_one,
};

use crate::utils::{ list_folders, list_simulation_dirs, list_simulations, list_motility_dirs };

use crate::plot::{ plot_center_of_mass, plot_msd };

pub fn compile_results() -> Result<(), Box<dyn std::error::Error>> {
    let source_data_path_str = "E:/results/cell-sim-1600000-8000000";
    let output_dir = Path::new("output");
    std::fs::create_dir_all(output_dir).expect("Unable to create output directory");

    // let mut image_paths = Vec::new();

    let directory = Path::new(source_data_path_str);
    for (motility_str, _) in list_motility_dirs(directory)? {
        let simulation_path_str = format!("{}/{}", source_data_path_str, motility_str);
        let directory = Path::new(&simulation_path_str);

        let folders = list_folders(directory)?;
        for (folder, soft, rho) in folders {
            // Adjust the range based on your data
            let data_path_str = format!("{}/{}", simulation_path_str, folder);
            let data_path = Path::new(&data_path_str);

            let simulation_dirs = list_simulation_dirs(data_path)?;
            // let filtered_simulation_dirs: Vec<_> = simulation_dirs
            //     .iter()
            //     .filter(|&&index| index == 0u32)
            //     .collect();
            for sim_index in simulation_dirs {
                println!("processing {} simulation {}", data_path_str, sim_index);
                let checkpoint_path_str = format!("{}/{}/checkpoint", data_path_str, sim_index);
                let checkpoint_path = Path::new(&checkpoint_path_str);

                let simulation_data = list_simulations(checkpoint_path)?;

                // let time_step_filter = 2_000_000;
                // let index_filter: i32 = 0;
                // let simulation_data: Vec<_> = simulation_data
                //     .iter()
                //     .filter(
                //         |(_, index, time_step)|
                //             *index == index_filter && *time_step <= time_step_filter
                //     )
                //     .collect();

                let data: Result<Vec<_>, _> = simulation_data
                    .into_iter()
                    .map(|(data_path_str, index, time_step)| {
                        let file_path_str = format!("{}/{}", checkpoint_path_str, data_path_str);
                        let file_path = Path::new(&file_path_str);
                        match read_checkpoint(time_step, &file_path) {
                            Ok(data) => Ok((time_step, index, data)),
                            Err(e) => Err(e),
                        }
                    })
                    .collect();

                let data = match data {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("Error reading data: {}", e);
                        continue;
                    }
                };
                let com = data
                    .into_iter()
                    .map(|(time_step, index, data)| {
                        let (x_center, y_center) = compute_center_of_mass_one(&data.data);
                        let (x_0, y_0) = (data.intervals[0].0, data.intervals[1].0);
                        (time_step, index, (x_0 + x_center, y_0 + y_center))
                    })
                    .collect::<Vec<_>>();

                let com_output_dir_str = format!(
                    "coms/{}/soft{}-rho{}/{}",
                    motility_str,
                    soft,
                    rho,
                    sim_index
                );
                let com_output_dir = Path::new(&com_output_dir_str);
                if !com_output_dir.exists() {
                    std::fs::create_dir_all(com_output_dir)?;
                }

                let mut file = File::create(format!("{}/com.txt", com_output_dir_str))?;
                for (time_step, index, (x, y)) in &com {
                    writeln!(file, "{}, {}, {}, {}", time_step, index, x, y)?;
                }

                // plot_center_of_mass(&com, "output")?;

                println!("finished processing checkpoint data at {}", checkpoint_path_str);
                // plot_contour(&data, &output_path, center_of_mass);
            }
            println!("finished processing everything at {}", data_path_str);
        }
    }

    Ok(())

    // create_movie(image_paths, "output/movie.gif");
}

pub fn plot_coms() -> Result<(), Box<dyn std::error::Error>> {
    let com_path_str = "coms";
    let directory = Path::new(com_path_str);
    for (motility_str, _) in list_motility_dirs(directory)? {
        let motility_com_path_str = format!("{}/{}", com_path_str, motility_str);
        let directory = Path::new(&motility_com_path_str);

        let folders = list_folders(directory)?;
        for (folder, soft, rho) in folders {
            let data_path_str = format!("{}/{}", com_path_str, folder);
            let data_path = Path::new(&data_path_str);

            let simulation_dirs = list_simulation_dirs(data_path)?;
            for sim_index in simulation_dirs {
                let com_output_dir_str = format!(
                    "coms/{}/soft{}-rho{}/{}",
                    motility_str,
                    soft,
                    rho,
                    sim_index
                );
                let com_output_file_str = format!("{}/com.txt", com_output_dir_str);
                let com_output_file = Path::new(&com_output_file_str);

                let com = read_com_file(com_output_file)?;
                plot_center_of_mass(&com, &com_output_dir_str)?;
            }
        }
    }
    Ok(())
}

pub fn compute_msd() -> Result<(), Box<dyn std::error::Error>> {
    let com_path_str = "coms";
    let directory = Path::new(com_path_str);
    let mut msd_values: HashMap<(String, u32, u32), HashMap<i32, (f64, u32)>> = HashMap::new();

    for (motility_str, _) in list_motility_dirs(directory)? {
        let motility_com_path_str = format!("{}/{}", com_path_str, motility_str);
        let directory = Path::new(&motility_com_path_str);

        let folders = list_folders(directory)?;
        for (folder, soft, rho) in folders {
            let data_path_str = format!("{}/{}", com_path_str, folder);
            let data_path = Path::new(&data_path_str);

            let simulation_dirs = list_simulation_dirs(data_path)?;
            let msd_entry = msd_values
                .entry((motility_str.clone(), soft, rho))
                .or_insert(HashMap::new());

            for sim_index in simulation_dirs {
                let com_output_dir_str = format!(
                    "coms/{}/soft{}-rho{}/{}",
                    motility_str,
                    soft,
                    rho,
                    sim_index
                );
                let com_output_file_str = format!("{}/com.txt", com_output_dir_str);
                let com_output_file = Path::new(&com_output_file_str);

                let com = read_com_file(com_output_file)?;
                let msd_particles = calculate_time_dep_msd_particles(&com)?;
                let msd = calculate_time_dep_msd(&com)?;

                let msd_output_file_str = format!("{}/msd.txt", com_output_dir_str);
                let mut file = File::create(msd_output_file_str)?;

                for (time_step, index, msd_value) in msd_particles {
                    writeln!(file, "{}, {}, {}", time_step, index, msd_value)?;
                }
                msd.iter().for_each(|(time_step, msd_value)| {
                    let time_entry = msd_entry.entry(*time_step).or_insert((0.0, 0));
                    let (msd_sum, msd_count) = time_entry;
                    *msd_sum += *msd_value;
                    *msd_count += 1;
                });
            }
        }
    }

    msd_values.into_iter().for_each(|((motility_str, soft, rho), msd_values)| {
        let average_msd: Vec<_> = msd_values
            .into_iter()
            .map(|(time_step, (msd_sum, msd_count))| (time_step, msd_sum / (msd_count as f64)))
            .collect();
        let output_plot_dir_str = format!("msd/{}/soft{}-rho{}", motility_str, soft, rho);
        std::fs::create_dir_all(&output_plot_dir_str).expect("Unable to create output directory");
        match plot_msd(&average_msd, &output_plot_dir_str) {
            Ok(_) => println!("Successfully plotted msd for soft{}-rho{}", soft, rho),
            Err(e) => eprintln!("Error plotting msd for soft{}-rho{}: {}", soft, rho, e),
        }

        let output_dir_str = format!("coms/{}/soft{}-rho{}", motility_str, soft, rho);
        let msd_output_file_str = format!("{}/msd.txt", output_dir_str);
        let mut file = File::create(msd_output_file_str).unwrap();
        for (time_step, msd_value) in &average_msd {
            writeln!(file, "{}, {}", time_step, msd_value).unwrap();
        }

        // ((*soft, *rho), average_msd)
    });

    Ok(())
}
