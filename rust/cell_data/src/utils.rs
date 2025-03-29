use regex::Regex;
use std::fs;
use std::path::Path;

pub fn list_simulations(
    directory: &Path
) -> Result<Vec<(String, u32, i32)>, Box<dyn std::error::Error>> {
    let mut folders = Vec::new();
    let re = Regex::new(r"data(\d+)_(\d+)").unwrap();

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            continue;
        }

        let folder_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => name,
            None => {
                continue;
            }
        };

        if let Some(captures) = re.captures(folder_name) {
            let index_num: u32 = captures.get(1).ok_or("Missing index capture")?.as_str().parse()?;
            let time_num: i32 = captures.get(2).ok_or("Missing time capture")?.as_str().parse()?;
            folders.push((folder_name.to_string(), index_num, time_num));
        }
    }
    folders.sort_by(|a, b| { a.2.cmp(&b.2).then_with(|| a.1.cmp(&b.1)) });
    Ok(folders)
}

pub fn list_folders(
    directory: &Path
) -> Result<Vec<(String, u32, u32)>, Box<dyn std::error::Error>> {
    let mut folders = Vec::new();
    let re = Regex::new(r"soft(\d+)-rho(\d+)")?;

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let folder_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => name,
            None => {
                continue;
            }
        };

        if let Some(captures) = re.captures(folder_name) {
            let soft_num: u32 = captures.get(1).ok_or("Missing soft capture")?.as_str().parse()?;
            let rho_num: u32 = captures.get(2).ok_or("Missing rho capture")?.as_str().parse()?;
            folders.push((folder_name.to_string(), soft_num, rho_num));
        }
    }

    Ok(folders)
}

pub fn list_simulation_dirs(data_path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut sim_dirs = Vec::new();

    for entry in fs::read_dir(data_path)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let sim_dir_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => name,
            None => {
                continue;
            }
        };

        if let Ok(sim_index) = sim_dir_name.parse::<u32>() {
            sim_dirs.push(sim_index);
        }
    }

    Ok(sim_dirs)
}

pub fn list_motility_dirs(
    data_path: &Path
) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
    let mut motility_dirs = Vec::new();

    for entry in fs::read_dir(data_path)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let motility_dir_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => name,
            None => {
                continue;
            }
        };

        if let Ok(motility_value) = motility_dir_name.parse::<f32>() {
            motility_dirs.push((motility_dir_name.to_string(), motility_value));
        }
    }

    Ok(motility_dirs)
}
