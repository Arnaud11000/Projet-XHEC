from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def prepare_data(df):
    X = df.drop(columns=["Id","Cover_Type"])
    y = df["Cover_Type"]
    return X, y

def build_preprocessor(X):

    binary_cols = [
        col for col in X.columns if 
        col.startswith(("Wilderness_Area","Soil_Type"))
        ]
    numerical_cols = [
        col for col in X.columns if 
        col not in binary_cols
        ]

    preprocessor = ColumnTransformer([
        ("scaler", StandardScaler(), numerical_cols),
        ("binary", "passthrough", binary_cols)
        ])

    return preprocessor