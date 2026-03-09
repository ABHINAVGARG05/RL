"""
Configuration Module for RL Cloud Scheduler

This module contains all hyperparameters and configuration settings
for the cloud environment simulation and DQN training.

The configuration is organized into logical groups:
- Environment settings (servers, tasks)
- DQN hyperparameters (learning rate, network architecture)
- Training settings (episodes, batch size)
- Reward shaping parameters
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class ServerConfig:
    """Configuration for server parameters in the cloud environment."""
    
    # Number of servers in the cloud cluster
    num_servers: int = 5
    
    # CPU capacity range for each server (min, max)
    # Servers will have random capacity within this range
    cpu_capacity_range: Tuple[float, float] = (100.0, 100.0)
    
    # Maximum queue length before server is considered overloaded
    max_queue_length: int = 20


@dataclass
class TaskConfig:
    """Configuration for task generation parameters."""
    
    # CPU requirement range for tasks (min, max)
    cpu_requirement_range: Tuple[float, float] = (10.0, 50.0)
    
    # Processing time range in time steps (min, max)
    processing_time_range: Tuple[int, int] = (1, 10)
    
    # Probability of a new task arriving at each time step
    arrival_probability: float = 0.7


@dataclass
class DQNConfig:
    """Configuration for Deep Q-Network hyperparameters."""
    
    # Learning rate for Adam optimizer
    learning_rate: float = 1e-3
    
    # Discount factor (gamma) for future rewards
    # Higher values prioritize long-term rewards
    gamma: float = 0.99
    
    # Epsilon-greedy exploration parameters
    epsilon_start: float = 1.0      # Initial exploration rate
    epsilon_end: float = 0.01       # Minimum exploration rate
    epsilon_decay: float = 0.995    # Decay rate per episode
    
    # Replay buffer capacity
    buffer_size: int = 100000
    
    # Mini-batch size for training
    batch_size: int = 64
    
    # Target network update frequency (in steps)
    target_update_freq: int = 100
    
    # Neural network hidden layer sizes
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    
    # Whether to use double DQN
    use_double_dqn: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Total number of training episodes
    num_episodes: int = 500
    
    # Maximum steps per episode
    max_steps_per_episode: int = 200
    
    # Frequency of logging (every N episodes)
    log_frequency: int = 10
    
    # Frequency of model checkpoints (every N episodes)
    checkpoint_frequency: int = 50
    
    # Path to save trained models
    model_save_path: str = "checkpoints/"
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class RewardConfig:
    """Configuration for reward function parameters."""
    
    # Weight for latency in reward calculation
    latency_weight: float = 1.0
    
    # Penalty weight for server overload (lambda in the formula)
    overload_penalty: float = 10.0
    
    # Threshold for considering a server overloaded (as fraction of capacity)
    overload_threshold: float = 0.9
    
    # Bonus for balanced load distribution
    balance_bonus: float = 0.1


@dataclass
class Config:
    """
    Master configuration class that combines all config groups.
    
    This provides a single access point for all configuration parameters
    used throughout the project.
    
    Example usage:
        config = Config()
        print(config.server.num_servers)
        print(config.dqn.learning_rate)
    """
    
    server: ServerConfig = field(default_factory=ServerConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    # Device configuration (auto-detect GPU)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Post-initialization setup and validation."""
        # Validate configuration values
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.server.num_servers > 0, "Must have at least one server"
        assert 0 <= self.dqn.gamma <= 1, "Gamma must be in [0, 1]"
        assert self.dqn.epsilon_start >= self.dqn.epsilon_end, \
            "Epsilon start must be >= epsilon end"
        assert self.dqn.batch_size > 0, "Batch size must be positive"
        assert self.training.num_episodes > 0, "Must have at least one episode"


# Create a default configuration instance for easy import
default_config = Config()


def get_config(**kwargs) -> Config:
    """
    Factory function to create a configuration with custom parameters.
    
    Args:
        **kwargs: Override parameters for any configuration group.
                  Use nested dict format, e.g., server={'num_servers': 10}
    
    Returns:
        Config: Configuration instance with custom parameters
    
    Example:
        config = get_config(
            server={'num_servers': 10},
            dqn={'learning_rate': 0.001}
        )
    """
    config = Config()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key) and isinstance(value, dict):
            sub_config = getattr(config, key)
            for sub_key, sub_value in value.items():
                if hasattr(sub_config, sub_key):
                    setattr(sub_config, sub_key, sub_value)
    
    return config
