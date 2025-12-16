import pandas as pd
import numpy as np
import os

from src.utils.constants import DATA_DIR


DATA_PATH = os.path.join(DATA_DIR, "data.csv")


def load_data(config, slave_id=None):
    n = int(config.get("global", "n"))

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    A = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    if slave_id is None:
        # Master gets the full data
        return A, y
    else:
        # Slave gets its horizontal partition of A
        A_split = np.array_split(A, n, axis=0)
        A_i = A_split[slave_id]

        # And its vertical partition of A.T
        AT = A.T
        AT_split = np.array_split(AT, n, axis=1)
        AT_i = AT_split[slave_id]
        
        return A_i, AT_i    