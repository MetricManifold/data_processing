---
applyTo: "cpp/simulation/study/**/manuscript.tex,**/manuscript.tex"
---

# Manuscript Writing — Agent Instructions

> **When to consult this file:** You are writing, editing, or reviewing a LaTeX manuscript for submission to a physics journal (primarily Physical Review E). These rules encode the supervisor's preferences and lessons learned from prior editing sessions.

---

## Audience and tone

The reader is a physicist familiar with the field. Write at a sophisticated level. Do not explain things the reader already knows. Do not use language that sounds like it is selling or promoting the work.

---

## Absolute prohibitions

### Em dashes (`---`)
**Never use em dashes.** They are the single biggest marker of AI-generated text in academic writing. Use commas, parentheses, semicolons, or restructure the sentence.

- ❌ `the quench reveals the relaxation path---a measurement inaccessible to vertex models---while`
- ✅ `the quench reveals the relaxation path, a measurement inaccessible to vertex models. Meanwhile,`
- ✅ `the quench reveals the relaxation path (inaccessible in vertex models).`

### Emphasis for drama
Do not italicize words for rhetorical emphasis (`\emph{not}`, `\emph{both}`, `\emph{enhances}`). Italics are for defining terms on first use (e.g., `\emph{adhesion quench}`) or foreign phrases. The reader does not need typographical cues to understand emphasis.

### Self-congratulatory language
State results and let them speak for themselves. Do not write sentences whose purpose is to tell the reader that the result is good, important, or novel.

- ❌ "confirming that the gradient coupling encodes the same physics as..."
- ❌ "This establishes gradient-coupling adhesion as the phase field analog of..."
- ❌ "opening a path toward quantitative comparison"
- ✅ "The adhesion energy has the same form as the vertex model adhesion $-\gamma_\mathrm{vm}\sum\ell_{ij}$."

### Flowery / marketing vocabulary
The following words and phrases are banned. Replace with the suggested alternative or delete.

| Banned | Replacement |
|--------|-------------|
| transparent (about analysis) | simple, explicit |
| elegantly | (delete) |
| natural analog | counterpart, equivalent |
| opens a path toward | enables |
| central new ingredient | (describe what it is) |
| noteworthy | (delete, or "we note that") |
| squarely within | within |
| key features | (just describe them) |
| analytical transparency | analytical simplicity |
| a prerequisite for | required for |

### Rhetorical questions answered immediately
Do not set up dramatic tension with a question and then immediately answer it.

- ❌ "The central question is whether the transition coincides with $p_0^* \approx 3.81$."
- ✅ "We test whether the transition coincides with $p_0^* \approx 3.81$."

---

## Structure rules

### No separate Conclusion section
Use a single **Discussion** section at the end. Do not split into Discussion + Conclusion. Do not subsection the Discussion unless the paper is very long (>15 pages). A PRE article does not need a "roadmap" conclusion that restates the abstract.

### Do not restate the abstract in the conclusion/discussion
The discussion should say what was *learned*, not re-summarize what was done.

### Minimize signposting
- ❌ "In this work, we introduce..." → ✅ "We introduce..."
- ❌ "The remainder of this paper is organized as follows: Sec. II defines the model, Sec. III presents methods, Sec. IV–VII present results, and Sec. VIII discusses..." → ✅ Delete, or at most one sentence: "The model is defined in Sec.~II and methods in Sec.~III."

### Do not repeat yourself
If a concept is explained in the Model section, do not re-explain it in the Results or Discussion. A single clear explanation is sufficient. Common repetition traps:
- The stability bound $J < 2\gamma$ (state once in the derivation, reference elsewhere)
- The vertex model connection (state once, reference elsewhere)
- "This protocol has no vertex model analog" (say once in the introduction)

---

## Attributing prior work

### Credit the base model
When extending an existing model (e.g., Palmieri et al.), the Model section must open by stating this: "We build on the model of Palmieri et al. [ref]" or "We add [term] to the model of [ref]." Do not present the base model's equations as if they are original.

### Describe what others found, not what they didn't do
When reviewing literature, describe each paper's contribution positively.
- ❌ "The model was applied to cell sorting, not to a rigidity transition."
- ✅ "Using this model, Nonomura demonstrated differential-adhesion-driven cell sorting in binary mixtures."

### Use "the model" not "his/her model"
- ❌ "his model would require..."
- ✅ "the model would require..."

---

## Mathematics and terminology

### Use the original paper's terminology
If Palmieri calls the gradient + double-well terms "the free energy" (not "Cahn-Hilliard"), use the same language. Read the source paper before writing the Model section.

### Be mathematical, not vague
When describing mathematical choices, state them precisely.
- ❌ "we use bare gradients $\nabla\phi_i$"
- ✅ "we replace $h(\phi) = \phi^2(3-2\phi)$ with $\phi$ itself, so the coupling is $\nabla\phi_i \cdot \nabla\phi_j$ rather than $\nabla h(\phi_i) \cdot \nabla h(\phi_j)$"

### Justify choices with physics, not convenience
When making a simplification (e.g., dropping Nonomura's regularization), lead with the physical reason it works, not the practical benefit.
- ✅ "The stability bound $J < 2\gamma$ confines $J$ to a range where the interface remains stable without additional gradient stiffness. The trade-off is a narrower stable range in exchange for a single-parameter model."

### Do not fabricate thresholds
If there is no established threshold in the literature, do not invent one.
- ❌ "well below the $0.05R$ equilibration threshold"
- ✅ "negligible compared to the cell radius"

---

## Equations

### Move implementation details to Supplemental Material
The main text should contain the physically meaningful equations (free energy, equation of motion, stability bound, sharp-interface limit). The full variational derivative with all terms expanded, sum-field definitions, stencil descriptions, and numerical implementation details belong in the Supplemental Material.

### Break long equations across lines
If an equation overruns the column width, use `align` with `\nonumber\\` to split it. Check the compiled PDF.

---

## Figures

### Separate files for composability
Generate each panel as an individual PDF file. Compose them in LaTeX with `\includegraphics`. This makes it easy to update individual panels without regenerating the whole figure.

### Publication quality settings
- Use serif fonts (Computer Modern) matching LaTeX: `matplotlib.rcParams['font.family'] = 'serif'`, `'mathtext.fontset': 'cm'`
- Thin axis lines (0.6pt), small tick marks
- PRE single-column width: 3.375 inches
- Output as PDF for vector graphics; also save PNG at 600 dpi for quick preview
- Use LaTeX math in labels: `r'$\tilde{J}$'`

### Naming convention
```
figures/fig{N}_{description}.pdf
```

---

## References

### Use the same acknowledgments as the companion paper
Copy the acknowledgments block from the most recent submitted manuscript to maintain consistency across papers from the same group.

### Standard PRE bibliography style
Use `\begin{thebibliography}{99}` with `\bibitem` entries. Include DOI hyperlinks.

---

## Checklist before compiling

- [ ] No em dashes in text (comments with `% ---` are fine)
- [ ] No `\emph{}` used for rhetorical emphasis
- [ ] No fabricated numerical thresholds
- [ ] Base model properly attributed (Palmieri, Nonomura, etc.)
- [ ] No repetition between Introduction and Discussion
- [ ] No repetition between Model subsections
- [ ] Implementation details in Supplemental Material, not main text
- [ ] All `\ref{}` targets exist (no "undefined reference" warnings)
- [ ] Figures compile and are referenced in text
- [ ] Equations fit within column width

---

*Last updated: February 25, 2026*
