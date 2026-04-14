import os
import pickle
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(data_path):

    # 🔹 cargar datos
    X_train = load_pickle(os.path.join(data_path, "X_train.pkl"))
    X_test = load_pickle(os.path.join(data_path, "X_test.pkl"))
    y_train = load_pickle(os.path.join(data_path, "y_train.pkl"))
    y_test = load_pickle(os.path.join(data_path, "y_test.pkl"))

    # 🔹 columnas (de tu notebook)
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    cat_cols = [
        'gender', 'ever_married', 'work_type',
        'Residence_type', 'smoking_status'
    ]
    bin_cols = ['hypertension', 'heart_disease']

    # 🔹 pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    bin_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('bin', bin_pipeline, bin_cols)
    ])

    # 🔹 MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("stroke-prediction3")

    # 🔹 MODELO 1: Logistic Regression
    with mlflow.start_run(run_name="Logistic Regression"):

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("recall_class_1", report["1"]["recall"])
        mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    train("data/processed")