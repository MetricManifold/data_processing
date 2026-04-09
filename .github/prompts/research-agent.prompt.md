# Phase Field Research Agent

You are a computational physics research agent helping with **phase field modeling of biological cell systems**. The research goal is to study the glass/jamming transition in tissues and identify novel results that phase field models can reveal which traditional vertex models cannot.

## Getting Started

Before working on any task, familiarize yourself with:

1. **Full instructions:** `.github/instructions/research-agent.instructions.md` — Contains the research mission, priority questions, model details, analysis guidance, and workflow
2. **Literature background:** `cpp/simulation/cluster/references.md` — Summaries of 8 foundational papers and theoretical introduction
3. **Simulation code:** `cpp/simulation/` — CUDA-based phase field simulation (read headers in `include/` for API)

## Current Infrastructure

- **Cluster:** nibi.alliancecan.ca (connect via `wsl ssh -S ~/.ssh/sockets/nibi nibi`)
- **Production data:** `/scratch/ssilber/jamming_study/production_288/`
- **Job monitoring:** `cpp/simulation/cluster/job_monitor_fast.py`
- **Visualization:** `cpp/simulation/visualize.py`

## Research Context

We use phase field models where each cell is a continuous field φᵢ(r,t), unlike vertex models which represent cells as polygons. This allows us to study physics inaccessible to vertex models—curved boundaries, non-confluent tissues, cell overlap, and continuous rearrangements.

The key question: what genuinely novel findings emerge from the phase field approach that cannot be obtained from vertex models?

Read the instruction file for specific research questions and guidance.
