import numpy as np
import pandas as pd

AGE_CORRECTIONS = {
    18: +0.8, 19: +0.6, 20: +0.5, 21: +0.4, 22: +0.3,
    23: +0.2, 24: +0.1, 25:  0.0, 26:  0.0, 27: -0.1,
    28: -0.2, 29: -0.3, 30: -0.5, 31: -0.7, 32: -1.0,
    33: -1.3, 34: -1.6, 35: -2.0,
}

def _correction(age: float) -> float:
    age_int = int(np.clip(round(age), 18, 35))
    return AGE_CORRECTIONS.get(age_int, 0.0)

def predict(train, test: pd.DataFrame) -> np.ndarray:
    corrections = test["age_at_date"].apply(_correction).values
    return (test["pie"] + corrections).values