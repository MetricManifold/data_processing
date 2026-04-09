//! Batch directory traversal and result aggregation.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A discovered run directory containing a trajectory.txt file.
#[derive(Debug, Clone)]
pub struct RunDir {
    /// Full path to the directory
    pub path: PathBuf,
    /// Relative path from the base (used as key in output)
    pub relative: String,
    /// Path to trajectory.txt
    pub trajectory: PathBuf,
    /// Group key extracted from pattern (e.g. "Jk_0.10" or "v0")
    pub group: String,
    /// Run key within group (e.g. "run_1" or "r1")
    pub run_id: String,
}

/// Discover all run directories matching a glob pattern under `base_dir`.
///
/// Pattern examples:
/// - `Jk_*/run_*` → adhesion study layout
/// - `v*/r*` → jamming study layout
/// - `*` → flat directory of runs
///
/// Each matched directory must contain `trajectory.txt`.
pub fn discover_runs(base_dir: &Path, pattern: &str) -> Result<Vec<RunDir>> {
    let parts: Vec<&str> = pattern.split('/').collect();
    let mut results = Vec::new();

    match parts.len() {
        1 => {
            // Single-level: pattern matches run directories directly
            discover_single_level(base_dir, parts[0], &mut results)?;
        }
        2 => {
            // Two-level: group/run pattern
            discover_two_level(base_dir, parts[0], parts[1], &mut results)?;
        }
        _ => {
            anyhow::bail!("Pattern must have 1 or 2 levels (e.g. 'v*/r*' or 'Jk_*/run_*')");
        }
    }

    results.sort_by(|a, b| a.relative.cmp(&b.relative));
    Ok(results)
}

fn discover_single_level(
    base_dir: &Path,
    run_pattern: &str,
    results: &mut Vec<RunDir>,
) -> Result<()> {
    for entry in std::fs::read_dir(base_dir).context("Reading base directory")? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !glob_match(&name, run_pattern) {
            continue;
        }
        let traj = entry.path().join("trajectory.txt");
        if traj.exists() {
            results.push(RunDir {
                path: entry.path(),
                relative: name.clone(),
                trajectory: traj,
                group: "default".to_string(),
                run_id: name,
            });
        }
    }
    Ok(())
}

fn discover_two_level(
    base_dir: &Path,
    group_pattern: &str,
    run_pattern: &str,
    results: &mut Vec<RunDir>,
) -> Result<()> {
    for group_entry in std::fs::read_dir(base_dir).context("Reading base directory")? {
        let group_entry = group_entry?;
        if !group_entry.file_type()?.is_dir() {
            continue;
        }
        let group_name = group_entry.file_name().to_string_lossy().to_string();
        if !glob_match(&group_name, group_pattern) {
            continue;
        }
        for run_entry in std::fs::read_dir(group_entry.path())? {
            let run_entry = run_entry?;
            if !run_entry.file_type()?.is_dir() {
                continue;
            }
            let run_name = run_entry.file_name().to_string_lossy().to_string();
            if !glob_match(&run_name, run_pattern) {
                continue;
            }
            let traj = run_entry.path().join("trajectory.txt");
            if traj.exists() {
                results.push(RunDir {
                    path: run_entry.path(),
                    relative: format!("{}/{}", group_name, run_name),
                    trajectory: traj,
                    group: group_name.clone(),
                    run_id: run_name,
                });
            }
        }
    }
    Ok(())
}

/// Simple glob matching supporting `*` (match anything) and `?` (match one char).
fn glob_match(text: &str, pattern: &str) -> bool {
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    glob_match_impl(&t, 0, &p, 0)
}

fn glob_match_impl(text: &[char], ti: usize, pattern: &[char], pi: usize) -> bool {
    if pi == pattern.len() {
        return ti == text.len();
    }
    if pattern[pi] == '*' {
        // * matches zero or more characters
        for i in ti..=text.len() {
            if glob_match_impl(text, i, pattern, pi + 1) {
                return true;
            }
        }
        return false;
    }
    if ti >= text.len() {
        return false;
    }
    if pattern[pi] == '?' || pattern[pi] == text[ti] {
        return glob_match_impl(text, ti + 1, pattern, pi + 1);
    }
    false
}

/// Group run results by their group key for batch aggregation.
pub fn group_by_key(runs: &[RunDir]) -> BTreeMap<String, Vec<&RunDir>> {
    let mut groups: BTreeMap<String, Vec<&RunDir>> = BTreeMap::new();
    for run in runs {
        groups.entry(run.group.clone()).or_default().push(run);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(glob_match("Jk_0.10", "Jk_*"));
        assert!(glob_match("run_1", "run_*"));
        assert!(glob_match("v0", "v*"));
        assert!(glob_match("r12", "r*"));
        assert!(!glob_match("other", "Jk_*"));
        assert!(glob_match("abc", "???"));
        assert!(!glob_match("ab", "???"));
    }
}
