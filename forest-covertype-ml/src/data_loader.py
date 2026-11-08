import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"

def load_data(test_size=0.2, random_state=42):

    df = pd.read_csv(DATA_PATH)

    df_train, df_test = train_test_split(
        df, 
        test_size=test_size, 
        stratify = df["Cover_Type"], 
        random_state=random_state
        )

    return df_train, df_test