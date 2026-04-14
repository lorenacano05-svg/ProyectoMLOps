import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def preprocess_data(input_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    # 🔹 1. Cargar datos
    df_stroke = pd.read_csv(input_path)

    # 🔹 2. Limpieza
    df_stroke = df_stroke.drop(columns=['id'])

    # 🔹 3. Separar variables
    X = df_stroke.drop('stroke', axis=1)
    y = df_stroke['stroke']

    # 🔹 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 🔹 5. Guardar datos (SIN transformar)
    with open(os.path.join(output_path, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)

    with open(os.path.join(output_path, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)

    with open(os.path.join(output_path, "y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)

    with open(os.path.join(output_path, "y_test.pkl"), "wb") as f:
        pickle.dump(y_test, f)


if __name__ == "__main__":
    preprocess_data("healthcare-dataset-stroke-data.csv", "data/processed")