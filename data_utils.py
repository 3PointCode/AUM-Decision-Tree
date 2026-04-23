import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    df["target"] = (df["quality"] >= 6).astype(int)

    # input features
    X = df.drop(columns=["quality", "target"]).values
    y = df["target"].values

    return X, y, df