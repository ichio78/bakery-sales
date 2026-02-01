import pandas as pd

def build_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df[feature_cols].copy()
