import pandas as pd
import numpy as np


class LinearRegression:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
