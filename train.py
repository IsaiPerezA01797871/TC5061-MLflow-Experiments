import matplotlib.pyplot as plt
import tempfile
from sklearn.metrics import ConfusionMatrixDisplay
import argparse
from pathlib import Path

import pandas as pd
import mlflow 
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
 
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_features(df):

    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[feature_columns].copy()
    y = df["Machine failure"].copy()

    return X, y



def split_data(X, y, seed=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor():

    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    categorical_features = ["Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             categorical_features)
        ]
    )

    return preprocessor


def build_model(preprocessor, args):

    class_weight = (
        None if args.class_weight == "none"
        else "balanced"
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                C=args.C,
                max_iter=args.max_iter,
                class_weight=class_weight,
                random_state=args.seed
            ))
        ]
    )

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        )
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return metrics, cm

def log_confusion_matrix(cm):

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Failure", "Failure"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title("AI4I - Confusion Matrix")

    # Crear archivo temporal y registrarlo en MLflow
    with tempfile.TemporaryDirectory() as temp_dir:

        image_path = Path(temp_dir) / "confusion_matrix.png"

        fig.savefig(
            image_path,
            dpi=150,
            bbox_inches="tight"
        )

        mlflow.log_artifact(
            str(image_path),
            artifact_path="evaluation"
        )

    plt.close(fig)

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Entrenamiento parametrizado AI4I - MLflow"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la regularización"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones"
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        choices=["none", "balanced"],
        default="none",
        help="Ponderación de clases"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria"
    )

    return parser.parse_args()


def main():

    
    args = parse_arguments()

    # Configurar MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("AI4I_Parameterized_Experiments")

    print("\n===== CONFIGURACIÓN DEL EXPERIMENTO =====")
    print(f"C: {args.C}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Class weight: {args.class_weight}")
    print(f"Seed: {args.seed}")

    # Cargar dataset
    df = load_data(DATA_PATH)

    print("\nDataset cargado correctamente.")
    print(f"Registros: {df.shape[0]:,}")
    print(f"Variables: {df.shape[1]}")

    # Preparar variables
    X, y = prepare_features(df)

    print("\nVariables preparadas correctamente.")
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

    print("\nDistribución de la variable objetivo:")
    print(y.value_counts())

    # Dividir entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(
        X, y, seed=args.seed
    )

    print("\nDivisión de datos completada.")
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")

    print(
        f"\nPorcentaje de fallas en Train: "
        f"{y_train.mean():.2%}"
    )
    print(
        f"Porcentaje de fallas en Test: "
        f"{y_test.mean():.2%}"
    )

    # Construir preprocesamiento
    preprocessor = build_preprocessor()

    print("\nPreprocesamiento configurado correctamente.")
    print(preprocessor)

    # Construir modelo
    model = build_model(preprocessor, args)

    # Entrenar modelo
    model.fit(X_train, y_train)

    print("\nModelo entrenado correctamente.")
    print("Algoritmo: Logistic Regression")
    print("Pipeline: Preprocesamiento + Clasificación")

    # Evaluar modelo
    metrics, cm = evaluate_model(
        model, X_test, y_test
    )

    print("\n===== RESULTADOS DEL MODELO =====")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nMatriz de confusión:")
    print(cm)


    with mlflow.start_run(
        run_name="LogisticRegression_Experiment"
    ):

        # Registrar hiperparámetros
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", 0.20)

        # Registrar métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(
                metric_name,
                metric_value
            )
        # Registrar matriz de confusión
        log_confusion_matrix(cm)

        print("Matriz de confusión guardada en MLflow.")

                # Registrar el modelo entrenado
        mlflow.sklearn.log_model(
            sk_model=model,
            name="ai4i_logistic_regression"
        )

        print("Modelo entrenado guardado correctamente en MLflow.")

        print(
            "\nExperimento registrado correctamente en MLflow."
        )


# 12. PUNTO DE ENTRADA

if __name__ == "__main__":
    main()
