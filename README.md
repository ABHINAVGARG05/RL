# RL Cloud Scheduler: Resource Allocation with Dueling DQN

This project implements a Reinforcement Learning (RL) framework for dynamic resource allocation in a cloud computing environment. It uses a **Dueling Deep Q-Network (DQN)** agent to schedule incoming jobs across multiple servers, aiming to maximize resource utilization and prioritize high-importance tasks while minimizing SLA breaches.

## Project Structure

- `agents/`: Contains the RL agent logic.
  - `dqn_agent.py`: Implementation of the `DQNAgent` using a Dueling DQN architecture and Welford's online normalization for stable rewards.
- `env/`: Defines the simulation environment.
  - `resource_env.py`: A Gymnasium-based environment representing a cluster of machines with CPU and Memory capacities.
  - `server.py` & `task.py`: Low-level dataclasses for managing server state, task queues, and task generation.
- `baselines/`: Traditional scheduling heuristics for performance comparison.
  - `heuristics.py`: Includes First-Fit, Best-Fit, Round-Robin, and Greedy Priority strategies.
- `training/`: Scripts for model training and evaluation.
  - `train.py`: The main entry point for training the agent and plotting performance against baselines.
- `utils/`: Helper modules for data loading, logging, and configuration management.

## Features

- **Dueling DQN Architecture**: Separates the estimation of state value and action advantages for more stable learning.
- **Custom Gymnasium Environment**: Simulates resource consumption over time with multi-dimensional requirements (CPU, Memory, Priority, and Duration).
- **Welford Online Normalization**: Ensures consistent reward scaling across training episodes.
- **Borg Trace Integration**: Includes a `BorgDatasetLoader` to train the agent on realistic job traces.
- **Performance Benchmarking**: Automatically compares the trained RL agent against four standard heuristics.

## Requirements

The project requires **Python 3.9+** and the following libraries:

- `torch >= 2.0.0`
- `gymnasium >= 0.29.0`
- `numpy`, `pandas`, `matplotlib`, `seaborn`

Full dependencies are listed in `requirements.txt`.

## Getting Started

### 1. Generate Training Data

Before training, generate a dummy dataset (simulating a Borg trace subset) to feed into the environment:

```bash
python generate_dummy_trace.py
```

### 2. Train the Agent

Run the main training script to start the RL process. This will train the agent for 1000 episodes (by default) and save the best model to `checkpoints/dqn_resource.pt`.

```bash
python training/train.py
```

### 3. View Results

Training progress and final comparison charts (Reward curves, CPU/Mem utilization) are saved to:

```
training/plots/results.png
```

## Configuration

Hyperparameters such as learning rate, epsilon decay, and machine capacities can be adjusted in:

- `training/train.py` — via the `CONFIG` dictionary.
- `utils/config.py` — for default environment and network settings.