# Tetris‑RL: AlphaZero‑Style Tetris Agent

A research-grade, extensible framework for training a high‑score and adversarial Tetris agent (in the spirit of AlphaZero) on blitz‑style gameplay (e.g., 2‑minute Tetr.io matches).

---

## Features

- **Custom Gym‑style environment** (`envs/tetris_env.py`):
  - 7‑bag randomizer
  - Super Rotation System (SRS) piece rotations with full wall‑kick support
  - Configurable hold, next‑piece queue, and step limits to simulate 2‑minute blitz mode
- **AlphaZero‑style components** (`agents/`):
  - Monte Carlo Tree Search engine (`mcts.py`)
  - Policy/value neural network scaffold (`neural_net.py`)
  - Self‑play trainer entrypoint (`alphazero_trainer.py`)
- **Experiment configuration** (`configs/`):
  - YAML‑based settings for reproducible runs (e.g., `blitz_2min.yaml`)
- **CLI scripts** (`scripts/`):
  - `train.py` — launch training with a single command
  - `benchmark.py` — evaluate against benchmark bots or random agents
  - `human_vs_bot.py` — play interactively against your trained model
- **Testing & documentation**:
  - Unit tests for environment dynamics and MCTS (`tests/`)
  - Notebook for analysis and plotting (`notebooks/score_distributions.ipynb`)
  - Markdown docs (`docs/`)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tetris-rl.git
   cd tetris-rl
   ```

2. **Create & activate a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional)** Install additional packages for notebooks
   ```bash
   pip install jupyter matplotlib numpy
   ```

---

## Usage

### 1. Train an AlphaZero agent

```bash
python scripts/train.py --config configs/blitz_2min.yaml
```  
This will:
- Initialize self‑play using your `TetrisEnv`
- Run MCTS + neural network updates
- Log metrics to `data/tensorboard/`
- Save model checkpoints in `data/checkpoints/`

### 2. Benchmark against baseline bots

```bash
python scripts/benchmark.py --model data/checkpoints/latest.pt
```  
Compare your agent’s high‑score distributions against random play or scripted bots.

### 3. Play interactively against your agent

```bash
python scripts/human_vs_bot.py --model data/checkpoints/latest.pt
```  
Controls via keyboard. See script header for key mappings.

---

## Project Structure

```
├── envs/                   # Custom Gym‑style environment
│   └── tetris_env.py       # SRS + wall kicks + bag randomizer
├── agents/                 # AlphaZero components
│   ├── mcts.py             # MCTS search logic
│   ├── neural_net.py       # Network architecture stub
│   └── alphazero_trainer.py# Training loop
├── configs/                # Experiment configurations (YAML)
├── scripts/                # CLI entrypoints (train, benchmark, play)
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests (pytest)
├── data/                   # Checkpoints, logs (git‑ignored)
└── docs/                   # Project documentation
```

---

## Contributing

1. Fork the repo and create your feature branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
2. Commit your changes and push:
   ```bash
   git commit -m "Add feature X"
   git push origin feature/YourFeatureName
   ```
3. Open a Pull Request describing your improvements.

Please ensure tests pass and follow PEP8 style.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---



