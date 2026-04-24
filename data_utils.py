import pandas as pd

def quality_to_binary_target(quality):
    return (quality >= 6).astype(int)

def load_data(path):
    df = pd.read_csv(path)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    df["target"] = quality_to_binary_target(df["quality"])

    # input features
    X = df.drop(columns=["quality", "target"]).values
    y = df["target"].values

    return X, y, df