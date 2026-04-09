# Phase Field Research — Open Questions

I'm exploring open research questions in cell tissue mechanics using a GPU-accelerated multi-cell phase field model. The model represents each cell as a continuous scalar field φᵢ(r,t) with Cahn-Hilliard gradient energy, quartic steric repulsion, optional gradient-coupling adhesion, a soft volume constraint, and active self-propulsion via run-and-tumble dynamics. It runs on NVIDIA GPUs (local RTX 4090, cluster H100s) and supports 2D and 3D with up to ~4600 cells.

Before proposing any directions, please read:
1. The simulation instructions: `.github/instructions/cell-simulation.instructions.md`
2. The research agent guide: `.github/instructions/research-agent.instructions.md`
3. The literature review for the adhesion study: `cpp/simulation/study/adhesion/LITERATURE_REVIEW.md` (for context on what's been done in multi-cell phase field models)

Two studies are already underway (adhesion-controlled rigidity and Griffiths rare-region effects). I'm interested in what other questions this model is uniquely positioned to answer — things vertex models or cellular Potts models cannot do. After reading, propose 3–5 concrete research directions, ranked by novelty and feasibility, with specific observables and parameter ranges. Focus on questions where the phase field's continuous interfaces, curved boundaries, and non-confluent capability provide a genuine advantage.
