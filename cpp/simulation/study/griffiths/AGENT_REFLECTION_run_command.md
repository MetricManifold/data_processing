# Agent Reflection: `run_command` Usage — Archived

**Original date:** February 19–20, 2026

Document issues with the compute canada MCP tool.

**All content has been migrated to structured locations:**

| What | Where |
|------|-------|
| Open tool gaps (#1–6) | [FEEDBACK.md](../../../../rust/vtk_viewer/FEEDBACK.md) — Open Issues |
| Resolved tool gaps (#7–11) | [FEEDBACK.md](../../../../rust/vtk_viewer/FEEDBACK.md) — Resolved Issues |
| Shell → MCP tool mapping | [FEEDBACK.md](../../../../rust/vtk_viewer/FEEDBACK.md) — Behavioral Pattern |
| `list_jobs` truncation fix | Encoded in `_resolve_clusters` + `filter_required` response |
| `discover` `cluster` vs `clusters` | Strict validation in `_resolve_clusters()` |
| VTK frame cap + quota warnings | `estimate_cost` response + `start/resume_simulation` pre-flight |
| Checkpoint as source of truth | `read_checkpoint(read_params=true)` tool description |
| Preferred analysis workflow | [cluster-postprocessing.instructions.md](../../../../.github/instructions/cluster-postprocessing.instructions.md) |

**Key rule:** Use dedicated MCP tools, not `run_command`. If `list_jobs`
returns too many results, re-call with `job_name_pattern` and/or `state`
filters. Do not fall back to `sacct`.
