import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LinearRegression()
    model.fit(train[["pie"]], train["target_pie"])
    return model.predict(test[["pie"]])