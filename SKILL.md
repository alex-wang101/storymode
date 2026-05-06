---
name: reproducible-research-repo
description: Use this skill when creating, reviewing, or extending an open-source research/engineering repo where results must be reproducible through documented commands, configs, expected metrics, and generated reports.
---

# Reproducible Research Repo Skill

## Purpose

This skill helps build a repository and accompanying blog for an engineering or research finding that other people can reproduce from the command line.

The goal is not just to publish code. The goal is to make the finding auditable:

- A reader should understand the claim.
- A reviewer should be able to run a quick smoke test.
- A serious reproducer should be able to run the full experiment.
- Every reported number should map back to a config, command, run artifact, and expected metric.
- Any randomness, hardware dependency, external data dependency, or known source of nondeterminism should be documented.

Do not invent research results. Use placeholders where the real method, dataset, or metric is not yet implemented.

---

## Core Principle

Design the repo around this invariant:

> One config = one experimental condition = one reproducible claim.

Every meaningful experiment should have:

- a config file
- a command that runs it
- a run directory with artifacts
- a metrics file
- an expected metric entry
- a documentation row
- a report or table generated from the result

Avoid hardcoded experiments hidden inside scripts.

---

## When To Use This Skill

Use this skill when the user wants to:

- publish a research/engineering finding
- build a repo that others can reproduce via commands
- create a benchmark, ablation, evaluation, or experiment harness
- write a blog post connected to reproducible code
- structure an open-source research artifact
- make results easier for professors, reviewers, recruiters, or collaborators to verify

---

## Repository Expectations

The repository should support these user-facing workflows:

```bash
make setup
make smoke
make data
make reproduce
make eval
make report
make test
```

The repository should also expose script-level commands:

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py --config configs/smoke.yaml --dry-run
python scripts/run_experiment.py --config configs/reproduce_main.yaml
python scripts/compare_results.py --actual runs/latest/metrics.json --expected results/expected_metrics.json --tolerance 0.02
python scripts/generate_report.py --run-dir runs/latest
```

The quick path should be fast and low-friction. The full path can be slower, but must be clear.

Separate these two modes:

1. **Smoke reproduction**
   - Small deterministic test.
   - Runs in minutes.
   - Verifies that the pipeline works end to end.
   - Does not need to reproduce the full reported result.

2. **Full reproduction**
   - Runs the actual main experiment.
   - Produces the reported metrics, tables, and figures.
   - May require larger data, more compute, or longer runtime.

---

## Required Files

Create or maintain these files unless the repo already has an equivalent convention:

```text
README.md
BLOG.md
AGENTS.md
CITATION.cff
LICENSE
Makefile
pyproject.toml
.gitignore
.env.example
Dockerfile

src/<package_name>/
scripts/
configs/
data/
runs/
results/
reports/
tests/
docs/
```

Expected subfiles:

```text
src/<package_name>/__init__.py
src/<package_name>/pipeline.py
src/<package_name>/metrics.py
src/<package_name>/config.py
src/<package_name>/io.py
src/<package_name>/utils.py

scripts/download_data.py
scripts/prepare_data.py
scripts/run_experiment.py
scripts/evaluate.py
scripts/compare_results.py
scripts/generate_report.py

configs/smoke.yaml
configs/reproduce_main.yaml
configs/ablation_baseline.yaml
configs/ablation_our_method.yaml

results/expected_metrics.json
results/reproduction_log.md

docs/method.md
docs/reproducibility.md
docs/hardware.md
docs/troubleshooting.md
docs/experiment_registry.md

data/README.md
runs/.gitkeep
reports/figures/.gitkeep
```

If the project is not Python-based, adapt the implementation language while preserving the same reproducibility contract.

---

## README Requirements

The README should be optimized for a new reader deciding whether the result is credible.

It must include:

1. **One-sentence claim**

   Example:

   > We show that [method] improves [metric] by [amount] on [task/dataset] under [condition].

2. **Main result table**

   Include baseline, proposed method, key metrics, runtime, and notes.

3. **Quick reproduction**

   Include the shortest path:

   ```bash
   make setup
   make smoke
   make reproduce
   make eval
   ```

4. **Full reproduction**

   Include the expanded path:

   ```bash
   make data
   python scripts/run_experiment.py --config configs/reproduce_main.yaml
   python scripts/evaluate.py --run-dir runs/latest
   python scripts/generate_report.py --run-dir runs/latest
   ```

5. **Experiment registry**

   Every experiment should map to a config and command.

   Example columns:

   - condition
   - config
   - command
   - expected output
   - approximate runtime

6. **Hardware and runtime**

   Document:

   - OS
   - Python/Node version
   - CPU
   - GPU, if applicable
   - RAM
   - approximate runtime
   - approximate storage

7. **Reproducibility notes**

   Document:

   - random seeds
   - deterministic settings
   - known nondeterminism
   - acceptable tolerance
   - dependency versions
   - external services or APIs

8. **Troubleshooting**

   Include common failures and fixes.

9. **Citation**

   Include a citation section and connect it to `CITATION.cff`.

---

## BLOG.md Requirements

The blog should explain the finding, not just duplicate the README.

Use this narrative:

```markdown
# Title

## TL;DR

State the result in one paragraph.

## The Problem

Explain the engineering/research problem and why existing approaches are insufficient.

## The Hypothesis

Explain what you believed would work better.

## The Method

Explain the system, model, algorithm, or experiment design.

## The Result

Show the main table or figure.

## Why It Works

Give the intuition behind the result.

## Reproduce It

Link the result to repo commands:

```bash
make setup
make smoke
make reproduce
```

## Limitations

State where the method may fail, what assumptions it makes, and what was not tested.

## Future Work

List concrete next experiments.
```

The blog should link to the exact commit, config, expected metrics, and generated report once those exist.

---

## CLI Behavior

`scripts/run_experiment.py` must support:

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py --config configs/smoke.yaml --dry-run
python scripts/run_experiment.py --config configs/reproduce_main.yaml
```

Required behavior:

- `--list` prints available configs.
- `--dry-run` prints the resolved config and planned outputs without running the experiment.
- Running an experiment creates a timestamped directory under `runs/`.
- The run directory should contain:
  - `config_resolved.yaml`
  - `metrics.json`
  - `manifest.json`
  - `run_summary.txt` or `stdout.log`

`manifest.json` should include:

- experiment name
- config path
- timestamp
- git commit hash, if available
- Python/runtime version
- platform information
- seed
- command used
- output files
- dependency versions, if practical

---

## Config Requirements

Every experiment config should include:

```yaml
experiment:
  name: smoke
  description: "Tiny deterministic end-to-end check."

seed: 42

data:
  source: placeholder
  split: tiny

method:
  name: placeholder_method
  parameters: {}

evaluation:
  metrics:
    - accuracy
    - latency_ms

outputs:
  run_dir: runs
```

For real experiments, prefer explicit parameters over implicit code defaults.

Bad:

```yaml
method: our_method
```

Better:

```yaml
method:
  name: our_method
  retrieval_top_k: 50
  rerank_top_k: 10
  threshold: 0.72
  use_transcript_prior: true
```

---

## Metrics and Expected Results

`results/expected_metrics.json` should map experiment names to expected values and tolerances.

Example:

```json
{
  "smoke": {
    "accuracy": {
      "expected": 0.8,
      "tolerance": 0.001,
      "direction": "higher_is_better"
    },
    "latency_ms": {
      "expected": 100.0,
      "tolerance": 20.0,
      "direction": "lower_is_better"
    }
  }
}
```

`compare_results.py` should:

- load actual metrics
- load expected metrics
- compare each metric with tolerance
- print PASS/FAIL per metric
- exit nonzero on failure
- clearly report missing metrics

Do not silently pass when expected metrics are missing.

---

## Run Artifact Requirements

Each experiment run should be self-contained enough to debug later.

Each run directory should include:

```text
runs/<timestamp>_<experiment_name>/
├── config_resolved.yaml
├── metrics.json
├── manifest.json
├── run_summary.txt
└── artifacts/
```

Optional artifacts:

```text
predictions.jsonl
samples.jsonl
errors.jsonl
table.csv
figure.png
```

For large artifacts, store references rather than committing the files.

---

## Data Policy

The `data/` directory should not assume large raw datasets are committed to git.

`data/README.md` must explain:

- where the data comes from
- licensing or access restrictions
- expected directory layout
- download command
- preprocessing command
- checksums, if possible
- whether sample data is included
- whether full data requires credentials

If the project uses external APIs, include `.env.example` and document required environment variables.

Never commit secrets, private datasets, API keys, or credentials.

---

## Determinism and Reproducibility

Every nondeterministic step should expose a seed.

Document:

- random seed
- model version
- dataset version
- package versions
- hardware assumptions
- GPU nondeterminism
- acceptable tolerance

If exact reproducibility is impossible, say so directly and define an acceptable variance range.

Example:

```markdown
Due to GPU kernel nondeterminism, exact equality is not expected. We consider reproduction successful if the main metric is within ±0.02 of the reported value.
```

---

## Testing Requirements

Add basic tests for:

- config loading
- metric comparison
- smoke experiment
- output artifact creation
- deterministic behavior under a fixed seed

`make test` should run the test suite.

The smoke test should be fast enough for CI.

---

## Makefile Requirements

The Makefile should include:

```makefile
setup:
	python -m pip install -e ".[dev]"

smoke:
	python scripts/run_experiment.py --config configs/smoke.yaml

data:
	python scripts/download_data.py
	python scripts/prepare_data.py

reproduce:
	python scripts/run_experiment.py --config configs/reproduce_main.yaml

eval:
	python scripts/compare_results.py --actual runs/latest/metrics.json --expected results/expected_metrics.json

report:
	python scripts/generate_report.py --run-dir runs/latest

test:
	pytest

clean:
	rm -rf runs/* reports/generated_report.md
```

If `runs/latest` is a symlink, ensure it is created or updated after each run.

---

## Docker Requirements

If a Dockerfile is included, it should support:

```bash
docker build -t repro-project .
docker run --rm -it repro-project make smoke
```

Do not make Docker the only path unless local setup is unusually difficult.

---

## AGENTS.md Requirements

`AGENTS.md` should instruct future coding agents:

```markdown
# Agent Instructions

- Do not invent real research results.
- Keep all experiment behavior config-driven.
- Preserve CLI compatibility.
- Any new experiment must add:
  - a config file
  - an expected metric entry
  - a docs/experiment_registry.md row
  - a README row, if it supports the main claim
- Any nondeterministic step must expose a seed.
- Any external data dependency must be documented in data/README.md.
- Do not commit secrets or private data.
- Do not remove smoke tests.
- Do not change expected metrics without updating the reproduction log.
```

---

## Reproduction Log

`results/reproduction_log.md` should track meaningful result changes.

Each entry should include:

```markdown
## YYYY-MM-DD — Short description

- Commit:
- Config:
- Command:
- Dataset version:
- Hardware:
- Runtime:
- Metrics:
- Notes:
```

Use this log to explain why numbers changed.

---

## Implementation Standards

Use:

- `argparse` for scripts
- `pathlib` for paths
- `json` for machine-readable metrics
- `yaml` for configs
- clear error messages
- docstrings for public functions
- deterministic placeholder logic when real logic is missing

Avoid:

- hidden global state
- hardcoded local paths
- undocumented API calls
- silent downloads
- fake benchmark numbers
- metrics only printed to stdout
- results that cannot be traced to a config

---

## Final Validation

After implementation, run:

```bash
python -m compileall src scripts
make smoke
make eval
make report
make test
```

Then report:

- files created
- commands that pass
- commands that fail
- any placeholder logic used
- next steps needed to insert the real research implementation

---

## Output Standard For Coding Agents

When a coding agent finishes applying this skill, it should summarize:

```markdown
## Completed

- Created reproducible research repo scaffold.
- Added config-driven experiment runner.
- Added smoke reproduction path.
- Added expected metric comparison.
- Added generated report flow.

## Passing Commands

```bash
make smoke
make eval
make report
make test
```

## Placeholders

- Placeholder dataset generation.
- Placeholder method implementation.
- Placeholder expected metrics.

## Next Steps

1. Replace placeholder pipeline with real method.
2. Add real dataset download/preprocessing.
3. Update expected metrics from real runs.
4. Add final figures and blog links.
```

Do not claim real research reproduction is complete until the real method, data, and metrics are implemented.
