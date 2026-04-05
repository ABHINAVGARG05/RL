import pandas as pd
import numpy as np
import os


def create_dummy_trace(filename="data/borg_trace_subset.csv", num_rows=1000):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = {
        "cpu_req": np.random.uniform(0.01, 0.5, num_rows),
        "mem_req": np.random.uniform(0.01, 0.5, num_rows),
        "priority": np.random.choice([0, 1, 2, 9, 10, 11], num_rows),
        "duration": np.random.randint(1, 20, num_rows),
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Successfully generated {num_rows} rows of dummy Borg data at {filename}")


if __name__ == "__main__":
    create_dummy_trace()
