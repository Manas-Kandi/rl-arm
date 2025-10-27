# Panda Door RL

This repository contains a deterministic actor–critic (DDPG/TD3-style) pipeline for training a 7-DoF Franka Panda arm to open a door using the Robosuite PandaDoor task in MuJoCo. The codebase is modular, reproducible, and designed with future sim-to-real transfer at the GRASP Lab in mind.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/train_panda_door.yaml
```

To resume from a checkpoint supply `--resume path/to/checkpoint.pt`. Normalizer statistics are stored alongside checkpoints with the suffix `_normalizer.npz`.

## Project Layout

```
.
├── configs/
├── docker/
├── docs/
├── experiments/
│   └── checkpoints/
├── src/
│   ├── agents/
│   ├── envs/
│   ├── models/
│   ├── replay/
│   ├── utils/
│   ├── real/
│   ├── train.py
│   └── eval.py
└── README.md
```

- `configs/`: YAML experiment configurations.
- `src/envs/`: Simulation wrappers for Robosuite.
- `src/agents/`: Reinforcement learning agents (DDPG/TD3).
- `src/models/`: Policy and value networks.
- `src/replay/`: Replay buffer implementations.
- `src/utils/`: Utilities for normalization, logging, metrics, and safety.
- `src/train.py`: Training entry point.
- `src/eval.py`: Evaluation/evidence generation entry point.
- `experiments/`: Checkpoints, logs, and experiment metadata.

## Docker

A CUDA-enabled Dockerfile is provided under `docker/`. Build and run with:

```bash
cd docker
docker build -t panda-door-rl .
docker run --gpus all -it --rm \
    -v $(pwd)/..:/workspace/panda-door-rl \
    panda-door-rl
```

## Configuration

Training is configured entirely through YAML files. Key sections in `configs/train_panda_door.yaml`:

- `env`: robosuite task parameters, domain randomization, observation noise, and action delay knobs.
- `agent`: Actor/Critic model sizes, learning rates, TD3 options, gradient clipping, and target smoothing.
- `replay`: Buffer capacity, batch size, warmup steps, and update frequency.
- `exploration`: Gaussian noise schedule and optional action smoothing.
- `training`: Total steps, evaluation cadence, checkpoint cadence, reward scaling, and success thresholds.
- `logging`: Output directories plus TensorBoard & Weights & Biases configuration.
- `checkpoint`: Resume path and the number of latest checkpoints to retain.

Modify the YAML or author new config files under `configs/` to launch sweeps and ablations.

## Evaluation

Run deterministic evaluation on saved checkpoints:

```bash
python src/eval.py --config configs/train_panda_door.yaml \
                   --checkpoint experiments/<run>/checkpoints/checkpoint_500000.pt \
                   --episodes 20
```

Render the environment (requires proper display setup) by adding `--render`.

## Monitoring API

The repository ships with a lightweight FastAPI backend and React-ready JSONL metrics feed. To run the API locally:

```bash
source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn src.api.server:app --reload
```

Metrics streamed by the trainer are appended to `experiments/<run>/metrics.jsonl`. TensorBoard data is still written under `runs/`, so you can use either interface during development.

## React Dashboard

A Vite + React frontend lives under `frontend/` for real-time monitoring.

```bash
cd frontend
npm install
npm run dev  # opens http://localhost:3000 with proxy to the API

# production build
npm run build
```

After running `npm run build`, the FastAPI server automatically serves the static assets from `frontend/dist`, so deploying Uvicorn will host both the API and the dashboard.

## Sim-to-Real Checklist (Summary)

- Train robust policies in simulation with domain randomization.
- Validate across seeds and randomization ranges before hardware trials.
- Use the safety supervisor in `src/real/` to gate low-level commands.
- Follow the testing protocol outlined in `docs/` (see roadmap).

## Documentation

Additional guidelines, diagrams, and experiment reports can be stored under `docs/`.

## License

TBD. Update when distributing the project.
