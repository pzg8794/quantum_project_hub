# Quantum Project Hub

This repository organizes research workstreams, experiment testbeds, and supporting documentation for quantum networking and learning-based routing projects.

## Purpose

- Provide a single navigation point across active quantum-routing experiments.
- Keep testbed implementations discoverable and reproducible.
- Separate production-style framework work from exploratory testbeds.

## Key areas

- `testbeds/`: focused experiment repositories (for example, EXPNeuralUCB and related RL/CMAB tracks).
- `docs/`: project-level notes, plans, and reference documentation.
- `experiments/`: run artifacts and experiment coordination material.
- `tools/`: helper scripts used for local workflows.

## Quick start

1. Review repository structure and active tracks:

```bash
ls -la
ls -la testbeds
```

2. Open the target testbed README and follow its run instructions:

- `testbeds/EXPNeuralUCB/README.md`
- `testbeds/Paper8-RL_Entanglement_Routing/README.md`
- `testbeds/CMAB-CoMM/README.md`

## Documentation standard

Each testbed README should include:

- Problem scope and objective.
- Environment/dependency requirements.
- Quick start commands.
- Expected outputs.
- Reproducibility notes.

## Reviewer note

If you are evaluating this work for applications or research readiness, start with the testbed READMEs under `testbeds/` and then cross-reference `docs/` for design and planning context.