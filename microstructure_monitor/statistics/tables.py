"""Read persisted stock identifiers without CSV index type inference."""
import pandas as pd


def read_features(path):
    frame = pd.read_csv(path, converters={0: str})
    return frame.set_index(frame.columns[0]).copy()
