import pandas as pd
import numpy as np


class BorgDatasetLoader:
    def __init__(
        self, file_path: str, scale_cpu: float = 16.0, scale_mem: float = 64.0
    ):
        print(f"Loading Borg dataset from {file_path}")
        self.data = pd.read_csv(file_path)

        # scaling the data:
        self.data["cpu_req"] = self.data["cpu_req"] * scale_cpu
        self.data["mem_req"] = self.data["mem_req"] * scale_mem

        self.curr_indx = 0
        self.max_indx = len(self.data)

    def next_job(self) -> tuple: 
        if self.curr_indx >= self.max_indx:
            self.curr_indx = 0

        row = self.data.iloc[self.curr_indx]
        self.curr_indx += 1

        job_features = np.array(
            [row["cpu_req"], row["mem_req"], row["priority"]], dtype=np.float32
        )

        duration = int(row["duration"])
        return job_features, duration

    def reset(self) -> None:
        self.curr_indx = 0
