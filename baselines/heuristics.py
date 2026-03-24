import numpy as np

class FirstFitBaseline:
    name = "FirstFit"
    def select_action(self, obs: np.ndarray, env) -> int:
        job_cpu = env.current_job[0]
        job_mem = env.current_job[1]
        for m in range(env.n_machines):
            if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem:
                return m
        return env.n_machines 


class BestFitBaseline:
    name = "BestFit"
    def select_action(self, obs: np.ndarray, env) -> int:
        job_cpu = env.current_job[0]
        job_mem = env.current_job[1]
        best_m = None
        best_score = float("inf")
        for m in range(env.n_machines):
            if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem:
                leftover = (env.cpu_free[m] - job_cpu) + (env.mem_free[m] - job_mem)
                if leftover < best_score:
                    best_score = leftover
                    best_m = m
        return best_m if best_m is not None else env.n_machines


class RandomBaseline:
    name = "Random"
    def select_action(self, obs: np.ndarray, env) -> int:
        return np.random.randint(env.n_machines + 1)


class GreedyPriorityBaseline:
    name = "GreedyPriority"
    def select_action(self, obs: np.ndarray, env) -> int:
        job_cpu = env.current_job[0]
        job_mem = env.current_job[1]
        job_priority = env.current_job[2]

        candidates = [
            m for m in range(env.n_machines)
            if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem
        ]
        if not candidates:
            return env.n_machines

        if job_priority >= 2.0:
            return min(candidates, key=lambda m: (env.cpu_free[m] - job_cpu) + (env.mem_free[m] - job_mem))
        else:
            return candidates[0]