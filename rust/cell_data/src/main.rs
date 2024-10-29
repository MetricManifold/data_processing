use ndarray::Array2;
use std::fs::File;
use std::io::{ BufRead, BufReader };
use std::path::Path;
use itertools::Itertools;

mod read;
use read::{ read_data, RegionData };

mod compute;
use compute::{ compute_center_of_mass, compute_center_of_mass_one };

mod utils;
use utils::{ list_folders, list_simulation_dirs, list_simulations };

mod plot;
use plot::{ plot_center_of_mass };

fn compile_results() -> Result<(), Box<dyn std::error::Error>> {
    let directory = "E:/results/cell-sim-1600000-8000000";
    let output_dir = Path::new("output");
    std::fs::create_dir_all(output_dir).expect("Unable to create output directory");

    // let mut image_paths = Vec::new();

    for motility in [/*0.006, 0.008, 0.01, */ 0.012] {
        let simulation_path_str = format!("{}/{:.3}", directory, motility);
        let directory = Path::new(&simulation_path_str);
        let folders = list_folders(directory)?;

        for (folder, soft, rho) in folders {
            // Adjust the range based on your data
            let data_path_str = format!("{}/{}", simulation_path_str, folder);
            let data_path = Path::new(&data_path_str);

            let sim_dirs = list_simulation_dirs(data_path)?;
            for sim_index in sim_dirs.iter().filter(|index| **index == 0) {
                let checkpoint_path_str = format!("{}/{}/checkpoint", data_path_str, sim_index);
                let checkpoint_path = Path::new(&checkpoint_path_str);

                let simulation_data = list_simulations(checkpoint_path)?;

                // let mut msd_data = Vec::new();
                let data: Result<Vec<_>, _> = simulation_data
                    .iter()
                    .filter(|(_, index, time_step)| *index == 0 && *time_step <= 2_000_000)
                    .map(|(data_path_str, index, time_step)| {
                        let file_path_str = format!("{}/{}", checkpoint_path_str, data_path_str);
                        let file_path = Path::new(&file_path_str);
                        match read_data(*time_step, &file_path) {
                            Ok(data) => Ok((*time_step, *index, data)),
                            Err(e) => Err(e),
                        }
                    })
                    .collect();

                let data = data?;
                let com = data
                    .iter()
                    .map(|(time_step, index, data)| {
                        let (x_center, y_center) = compute_center_of_mass_one(&data.data);
                        let (x_0, y_0) = (data.intervals[0].0, data.intervals[1].0);
                        (*time_step, *index, (x_0 + x_center, y_0 + y_center))
                    })
                    .collect::<Vec<_>>();

                plot_center_of_mass(&com, "output")?;

                println!("finished processing checkpoint data at {}", checkpoint_path_str);
                // plot_contour(&data, &output_path, center_of_mass);
            }
            println!("finished processing everything at {}", data_path_str);
        }
    }

    Ok(())

    // create_movie(image_paths, "output/movie.gif");
}

fn main() {
    compile_results();
}
