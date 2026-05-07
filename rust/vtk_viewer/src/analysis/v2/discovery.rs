//! Discovery: filesystem layout → typed `RunSpec`s.
//!
//! A discovery rule is a path pattern with **typed** placeholders:
//!
//! ```text
//! "phase3a/d_{d:f64}R/run_{rep:int}"
//! "{study}/{N:int}c_rho{rho:int}_{cond}/run_{seed:int}"
//! ```
//!
//! Supported types: `int`, `f64`, `str` (default). Each match yields a
//! `RunSpec` whose `variables` map holds typed [`ScalarValue`]s. This is
//! the v2 improvement over the legacy stringly-typed discovery, where
//! `d` came out as `"2"` and you had to remember to parse it.

use anyhow::{anyhow, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// ScalarValue
// ---------------------------------------------------------------------------
/// Typed value parsed from a discovery placeholder. Serializable so it
/// round-trips through `RunAnalysis` JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScalarValue {
    Int(i64),
    Float(f64),
    Str(String),
}

impl ScalarValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(i) => Some(*i as f64),
            Self::Float(f) => Some(*f),
            Self::Str(s) => s.parse().ok(),
        }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            Self::Str(s) => s.parse().ok(),
        }
    }
    pub fn as_str(&self) -> String {
        match self {
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{}", f),
            Self::Str(s) => s.clone(),
        }
    }
}

impl std::fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(i) => write!(f, "{}", i),
            Self::Float(x) => write!(f, "{}", x),
            Self::Str(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// VarSpec (parsed from the placeholder)
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
enum VarKind {
    Int,
    Float,
    Str,
}

#[derive(Clone, Debug)]
struct VarSpec {
    name: String,
    kind: VarKind,
}

impl VarSpec {
    fn parse_capture(&self, raw: &str) -> Result<ScalarValue> {
        match self.kind {
            VarKind::Int => raw
                .parse::<i64>()
                .map(ScalarValue::Int)
                .with_context(|| format!("variable `{}` expects int, got `{}`", self.name, raw)),
            VarKind::Float => raw
                .parse::<f64>()
                .map(ScalarValue::Float)
                .with_context(|| format!("variable `{}` expects f64, got `{}`", self.name, raw)),
            VarKind::Str => Ok(ScalarValue::Str(raw.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// DiscoveryRule
// ---------------------------------------------------------------------------
/// A rule for finding runs on disk. Constructed from a pattern string.
///
/// Patterns use `{name}` for an untyped string capture (legacy
/// behaviour) or `{name:int}` / `{name:f64}` for typed captures.
///
/// The pattern is matched against the relative path from
/// `base_dir`. A run is accepted when (a) the path matches the regex
/// **and** (b) the trajectory file is present inside the matched
/// directory.
pub struct DiscoveryRule {
    pattern: String,
    regex: Regex,
    vars: Vec<VarSpec>,
    pub trajectory_name: String,
    pub checkpoint_name: Option<String>,
}

impl DiscoveryRule {
    pub fn new(
        pattern: &str,
        trajectory_name: &str,
        checkpoint_name: Option<&str>,
    ) -> Result<Self> {
        let (regex, vars) = compile_pattern(pattern)?;
        Ok(Self {
            pattern: pattern.to_string(),
            regex,
            vars,
            trajectory_name: trajectory_name.to_string(),
            checkpoint_name: checkpoint_name.map(|s| s.to_string()),
        })
    }

    /// Number of slashes (i.e. directory depth) implied by the pattern.
    /// Used to bound recursion in `discover()`.
    fn max_depth(&self) -> usize {
        self.pattern.matches('/').count() + 1
    }
}

/// Compile a placeholder pattern into a regex with named groups + the
/// typed variable specs. Pulled from the legacy `pattern_to_regex` and
/// extended for the `:type` suffix.
fn compile_pattern(pattern: &str) -> Result<(Regex, Vec<VarSpec>)> {
    let mut out = String::from("^");
    let mut vars = Vec::new();
    let mut chars = pattern.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                let mut spec = String::new();
                for inner in chars.by_ref() {
                    if inner == '}' {
                        break;
                    }
                    spec.push(inner);
                }
                if spec.is_empty() {
                    return Err(anyhow!("empty placeholder in pattern `{}`", pattern));
                }
                let (name, kind) = parse_placeholder(&spec)?;
                let regex_class = match kind {
                    VarKind::Int => r"-?\d+",
                    VarKind::Float => r"-?\d+(?:\.\d+)?",
                    VarKind::Str => r"[-\w.]+",
                };
                out.push_str(&format!("(?P<{}>{})", name, regex_class));
                vars.push(VarSpec { name, kind });
            }
            '/' => out.push_str(r"[/\\]"),
            c if r".+*?^$|()[]\\".contains(c) => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push('$');
    let re = Regex::new(&out)
        .with_context(|| format!("invalid pattern → regex: {}", out))?;
    Ok((re, vars))
}

fn parse_placeholder(spec: &str) -> Result<(String, VarKind)> {
    if let Some((name, ty)) = spec.split_once(':') {
        let kind = match ty {
            "int" | "i64" => VarKind::Int,
            "f64" | "float" => VarKind::Float,
            "str" | "string" => VarKind::Str,
            other => return Err(anyhow!("unknown placeholder type `{}`", other)),
        };
        Ok((name.to_string(), kind))
    } else {
        Ok((spec.to_string(), VarKind::Str))
    }
}

// ---------------------------------------------------------------------------
// RunSpec
// ---------------------------------------------------------------------------
/// One discovered run: directory path, trajectory path, optional
/// checkpoint path, and the typed variables extracted from the pattern.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub directory: PathBuf,
    pub trajectory: PathBuf,
    pub checkpoint: Option<PathBuf>,
    pub variables: BTreeMap<String, ScalarValue>,
}

impl RunSpec {
    pub fn var(&self, name: &str) -> Option<&ScalarValue> {
        self.variables.get(name)
    }
}

// ---------------------------------------------------------------------------
// discover()
// ---------------------------------------------------------------------------
/// Walk `base_dir` and return every run that matches `rule`.
pub fn discover(base_dir: &Path, rule: &DiscoveryRule) -> Result<Vec<RunSpec>> {
    let mut out = Vec::new();
    walk(base_dir, base_dir, rule, 0, &mut out)?;
    out.sort_by(|a, b| a.directory.cmp(&b.directory));
    Ok(out)
}

fn walk(
    base_dir: &Path,
    current: &Path,
    rule: &DiscoveryRule,
    depth: usize,
    out: &mut Vec<RunSpec>,
) -> Result<()> {
    if depth > rule.max_depth() {
        return Ok(());
    }
    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let rel = path
            .strip_prefix(base_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        if let Some(caps) = rule.regex.captures(&rel) {
            let traj = path.join(&rule.trajectory_name);
            if !traj.exists() {
                continue;
            }
            let mut variables = BTreeMap::new();
            for v in &rule.vars {
                let raw = caps
                    .name(&v.name)
                    .ok_or_else(|| anyhow!("missing capture `{}`", v.name))?
                    .as_str();
                variables.insert(v.name.clone(), v.parse_capture(raw)?);
            }
            let checkpoint = rule.checkpoint_name.as_ref().and_then(|n| {
                let p = path.join(n);
                if p.exists() {
                    Some(p)
                } else {
                    None
                }
            });
            out.push(RunSpec {
                directory: path.clone(),
                trajectory: traj,
                checkpoint,
                variables,
            });
            continue; // don't recurse into a matched dir
        }
        walk(base_dir, &path, rule, depth + 1, out)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn placeholder_parses_typed() {
        let (_re, vars) = compile_pattern("d_{d:f64}R/run_{rep:int}").unwrap();
        assert_eq!(vars[0].name, "d");
        assert_eq!(vars[0].kind, VarKind::Float);
        assert_eq!(vars[1].name, "rep");
        assert_eq!(vars[1].kind, VarKind::Int);
    }

    #[test]
    fn placeholder_default_str() {
        let (_re, vars) = compile_pattern("{study}/run_{seed}").unwrap();
        assert_eq!(vars[0].kind, VarKind::Str);
        assert_eq!(vars[1].kind, VarKind::Str);
    }

    #[test]
    fn compiled_regex_matches() {
        let (re, _) = compile_pattern("d_{d:f64}R/run_{rep:int}").unwrap();
        let caps = re.captures("d_2.0R/run_3").expect("should match");
        assert_eq!(caps.name("d").unwrap().as_str(), "2.0");
        assert_eq!(caps.name("rep").unwrap().as_str(), "3");
    }

    #[test]
    fn discovers_phase3a_layout_with_typed_vars() {
        // Build a phase3a-shaped tree under a tempdir.
        let root = std::env::temp_dir().join("v2_disc_test");
        let _ = fs::remove_dir_all(&root);
        for d in &[2.0_f64, 4.0, 8.0, 20.0] {
            for rep in 1..=3 {
                let dir = root.join("phase3a")
                    .join(format!("d_{}R", *d as i64))
                    .join(format!("run_{:02}", rep));
                fs::create_dir_all(&dir).unwrap();
                fs::write(dir.join("trajectory.txt"), b"# fake\n").unwrap();
            }
        }

        let rule = DiscoveryRule::new(
            "phase3a/d_{d:int}R/run_{rep:int}",
            "trajectory.txt",
            None,
        )
        .unwrap();
        let runs = discover(&root, &rule).unwrap();
        assert_eq!(runs.len(), 12, "expect 4 distances * 3 reps");
        let r0 = &runs[0];
        // Variables are typed. Lookup as int works.
        assert!(matches!(r0.variables["d"], ScalarValue::Int(_)));
        assert!(matches!(r0.variables["rep"], ScalarValue::Int(_)));
        assert!(r0.var("d").unwrap().as_f64().is_some());
        let _ = fs::remove_dir_all(&root);
    }
}
