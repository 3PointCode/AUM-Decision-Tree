import numpy as np
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

def main():
    X, y, df = load_data("data/WineQT.csv")

    print("Probe number:", X.shape[0])
    print("Features number:", X.shape[1])
    print("Class distribution:", np.bincount(y))
    print(df.head())

if __name__ == "__main__":
    main()
