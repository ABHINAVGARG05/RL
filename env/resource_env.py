import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

class ResourceAllocationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_machines: int = 4,
        cpu_capacity: float = 16.0,  
        mem_capacity: float = 64.0,   
        max_jobs_per_ep: int = 200,
        rejection_penalty: float = -2.0,
        sla_breach_penalty: float = -5.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_machines = n_machines
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.max_jobs = max_jobs_per_ep
        self.rejection_penalty = rejection_penalty
        self.sla_breach_penalty = sla_breach_penalty
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(n_machines + 1)

        obs_dim = 2 * n_machines + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cpu_free = np.full(self.n_machines, self.cpu_capacity, dtype=np.float32)
        self.mem_free = np.full(self.n_machines, self.mem_capacity, dtype=np.float32)
        self.jobs_processed = 0
        self.total_reward = 0.0
        self.current_job = self._generate_job()
        return self._obs(), {}

    def step(self, action: int):
        job_cpu, job_mem, job_priority = self.current_job
        reward = 0.0
        info = {}

        if action == self.n_machines:
            reward = self.rejection_penalty * job_priority
            info["event"] = "rejected"
        else:
            m = action
            if self.cpu_free[m] >= job_cpu and self.mem_free[m] >= job_mem:
                self.cpu_free[m] -= job_cpu
                self.mem_free[m] -= job_mem
                cpu_util = 1 - (self.cpu_free[m] / self.cpu_capacity)
                mem_util = 1 - (self.mem_free[m] / self.mem_capacity)
                efficiency = (cpu_util + mem_util) / 2
                # ← Normalise: keep priority influence but cap reward magnitude
                reward = job_priority * (0.5 + 0.5 * efficiency)  # range: [0.5, 1.5] × priority
                info["event"] = "allocated"
                info["machine"] = m
            else:
                reward = self.sla_breach_penalty * job_priority
                info["event"] = "sla_breach"

        self.jobs_processed += 1
        self.total_reward += reward
        done = self.jobs_processed >= self.max_jobs
        self.current_job = self._generate_job()

        return self._obs(), reward, done, False, info

    def _generate_job(self):
        """Randomly sample a job's resource requirements and priority."""
        cpu_req = self.rng.uniform(0.5, self.cpu_capacity * 0.4)
        mem_req = self.rng.uniform(1.0, self.mem_capacity * 0.4)
        priority = self.rng.choice([1.0, 2.0, 3.0], p=[0.5, 0.35, 0.15])
        return np.array([cpu_req, mem_req, priority], dtype=np.float32)

    def _obs(self):
        cpu_norm = self.cpu_free / self.cpu_capacity
        mem_norm = self.mem_free / self.mem_capacity
        job_cpu_norm = np.array([self.current_job[0] / self.cpu_capacity])
        job_mem_norm = np.array([self.current_job[1] / self.mem_capacity])
        job_pri_norm = np.array([self.current_job[2] / 3.0])
        return np.concatenate([cpu_norm, mem_norm, job_cpu_norm, job_mem_norm, job_pri_norm]).astype(np.float32)

    def utilization(self):
        cpu_util = 1 - self.cpu_free.mean() / self.cpu_capacity
        mem_util = 1 - self.mem_free.mean() / self.mem_capacity
        return {"cpu": cpu_util, "mem": mem_util}

    def render(self):
        u = self.utilization()
        job = self.current_job
        print(
            f"[Jobs {self.jobs_processed:4d}] "
            f"CPU util={u['cpu']:.1%}  MEM util={u['mem']:.1%} | "
            f"Next job: CPU={job[0]:.1f} MEM={job[1]:.1f} Pri={int(job[2])}"
        )