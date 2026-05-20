import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_heart_data(filepath="../data/heart_disease.csv"):
    # 1. Load data
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    df = pd.read_csv(filepath, names=columns, na_values="?")

    # 2. Handle missing values
    df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

    # 3. Binarize target: 0 = Healthy, 1 = Sick (original values > 0)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # 4. Separate features (X) and target (y)
    X = df.drop(columns=["target"])
    y = df["target"]

    # 5. Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled numpy array back to DataFrame for column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X, X_scaled, y


def load_wine_data(filepath="../data/wine_quality-red.csv"):
    # 1. Load data
    df = pd.read_csv(filepath, sep=";")

    # 2. Handle missing values
    df = df.dropna()

    # 3. Binarize target: 1 = Good (quality >= 6), 0 = Bad (quality < 6)
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)

    # 4. Separate features (X) and target (y)
    X = df.drop(columns=["quality"])
    y = df["quality"]

    # 5. Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X, X_scaled, y
