# # src/data_utils.py

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# # Path is relative to src/ (we run python main.py from inside src/)
# DATA_PATH = "../data/diabetes.csv"


# def load_and_split_data(test_size: float = 0.2, random_state: int = 42):
#     """
#     Loads the provided diabetes.csv file, drops the Id column,
#     standardizes features, and returns train/test splits.

#     Assumes columns:
#     Id, Pregnancies, Glucose, BloodPressure, SkinThickness,
#     Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
#     """

#     df = pd.read_csv(DATA_PATH)

#     # Target is Outcome
#     target_col = "Outcome"
#     if target_col not in df.columns:
#         # Fallback: last column as target if name ever changes
#         target_col = df.columns[-1]

#     # Drop Id (not a real feature) and the target column from X
#     drop_cols = [target_col]
#     if "Id" in df.columns:
#         drop_cols.append("Id")

#     X = df.drop(columns=drop_cols).values.astype(float)
#     y = df[target_col].values.astype(int)

#     # Standardize features: mean=0, std=1
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled,
#         y,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=y,
#     )

#     # Feature names without Id and Outcome
#     feature_names = df.drop(columns=drop_cols).columns

#     return X_train, X_test, y_train, y_test, feature_names, scaler
