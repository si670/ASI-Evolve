## ASI-Evolve

ASI-Evolve is an agentic evolution framework for long-horizon AI research. It closes the loop between prior knowledge, proposal generation, experiment execution, and post-hoc analysis through a repeated `learn -> design -> experiment -> analyze` cycle.

This release focuses on the reusable framework plus the circle-packing benchmark used in our ablation studies. The full research program described in the accompanying paper covers broader tasks across architecture design, data curation, reinforcement learning algorithms, and biomedical modeling.

## What Is Included

- A general-purpose evolution pipeline with `Researcher`, `Engineer`, `Analyzer`, and optional `Manager` agents.
- A persistent experiment database that stores nodes containing motivation, code, results, analysis, and metadata.
- A cognition store for injecting human priors through embedding-based retrieval.
- Multiple sampling strategies for parent selection: `ucb1`, `random`, `greedy`, and an island-style sampler inspired by MAP-Elites.
- A runnable circle-packing demo in [`experiments/circle_packing_demo`](./experiments/circle_packing_demo).
- Saved high-scoring circle-packing programs from ablation runs in [`experiments/best/circle_packing`](./experiments/best/circle_packing).

## Framework Overview

Each evolution round follows the same structure:

1. Sample historical nodes from the database.
2. Retrieve relevant cognition items from the cognition store.
3. Ask the `Researcher` to propose the next candidate program.
4. Ask the `Engineer` to run the experiment and collect structured metrics.
5. Ask the `Analyzer` to turn outcomes into reusable lessons.
6. Store the new node and continue the loop.

The design is aligned with the method described in the paper: the cognition base improves cold-start exploration, while the analyzer helps transform raw results into reusable, task-specific insight.

## Repository Layout

```text
ASI-Evolve/
|-- main.py
|-- config.yaml
|-- cognition/
|-- database/
|-- pipeline/
|-- utils/
`-- experiments/
    |-- circle_packing_demo/
    `-- best/
        `-- circle_packing/
```

## Installation

```bash
pip install -r requirements.txt
```

The circle-packing demo expects:

- Python 3.10+.
- `bash` and `python3` available on your system path.
- Optional experiment tracking through Weights & Biases.

## Quick Start

Initialize the circle-packing cognition store:

```bash
python experiments/circle_packing_demo/init_cognition.py
```

Run the demo from the repository root:

```bash
python main.py \
  --experiment circle_packing_demo \
  --steps 10 \
  --sample-n 3 \
  --eval-script /absolute/path/to/experiments/circle_packing_demo/eval.sh
```

Use an absolute path for `--eval-script`. The engineer runs the evaluator from each step directory, so absolute paths are the most reliable option.

## Circle-Packing Demo

The demo optimizes a constructive program for packing 26 circles inside a unit square and maximizing the sum of radii. It includes:

- [`input.md`](./experiments/circle_packing_demo/input.md): task definition.
- [`config.yaml`](./experiments/circle_packing_demo/config.yaml): experiment-specific overrides.
- [`initial_program`](./experiments/circle_packing_demo/initial_program): a seed solution.
- [`init_cognition.py`](./experiments/circle_packing_demo/init_cognition.py): curated prior knowledge for the benchmark.
- [`evaluator.py`](./experiments/circle_packing_demo/evaluator.py) and [`eval.sh`](./experiments/circle_packing_demo/eval.sh): evaluation logic.
- [`prompts/`](./experiments/circle_packing_demo/prompts): task-specific prompts.

The `experiments/best/circle_packing` directory stores selected programs from our ablation runs, including strong UCB1 and island-sampler trajectories.

## Configuration

Configuration is merged in the following order:

1. Repository-level [`config.yaml`](./config.yaml)
2. Experiment-level `experiments/<name>/config.yaml`
3. An explicit file passed through `--config`

This makes it easy to keep stable defaults while overriding only what changes for a specific benchmark or run.

## Notes on Scope

This repository release contains the framework, the circle-packing demo, and artifacts used in the ablation section. The larger task suites discussed in the paper are intentionally heavier and are not bundled here as turnkey experiments.

## Citation

If you use ASI-Evolve in your work, please cite the accompanying paper:

```bibtex
@misc{asi_evolve_2026,
  title={ASI-Evolve},
  author={ASI-Evolve Team},
  year={2026},
  note={Preprint metadata should be updated once the paper title and author list are finalized}
}
```
