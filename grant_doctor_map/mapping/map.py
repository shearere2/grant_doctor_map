import pandas as pd
import numpy as np
import entity_resolution_model
def map(df:pd.DataFrame)->np.ndarray:
    series = entity_resolution_model.predict(df)
    return series

if __name__ == "__main__":
    block = None
    block['prediction'] = map(block)