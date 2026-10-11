# AI4I — Entrenamiento parametrizado y seguimiento con MLflow

## 1. Objetivo

Desarrollar un flujo reproducible de entrenamiento y evaluación de un modelo de Machine Learning para predecir fallas en maquinaria industrial utilizando el dataset AI4I 2020 Predictive Maintenance.

Se utiliza regresión logística, un pipeline de preprocesamiento y MLflow para registrar hiperparámetros, métricas, matrices de confusión y modelos entrenados.

## 2. Dataset

**Fuente:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset.

- Registros: 10,000
- Variable objetivo: `Machine failure`
- Sin falla: 9,661 registros (96.61%)
- Con falla: 339 registros (3.39%)

Variables predictoras:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Se excluyen identificadores y variables que podrían introducir fuga de información.

## 3. Tecnologías

- Python 3.13.5
- NumPy 2.5.3
- Pandas 3.0.6
- Scikit-learn 1.9.1
- Matplotlib 3.11.2
- MLflow 3.16.1

## 4. Estructura del proyecto

```text
Actividad_mlflow_experimentos/
├── data/
│   └── ai4i2020.csv
├── evidence/
├── notebooks/
│   └── TC5061_Notebook_Baseline_PerezIsai.ipynb
├── train.py
├── requirements.txt
└── README.md
```

El notebook contiene el análisis exploratorio y el baseline inicial. El archivo `train.py` concentra el flujo ejecutable de entrenamiento, evaluación y registro de experimentos.

## 5. Preparación del entorno

Desde PowerShell, dentro del directorio del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En el entorno corporativo utilizado durante la actividad, el entorno virtual se creó fuera del repositorio debido a restricciones de escritura.

## 6. Ejecución de MLflow

Iniciar MLflow en una terminal independiente:

```powershell
python -m mlflow ui
```

Abrir la interfaz:

http://127.0.0.1:5000

El script utiliza esa dirección para conectarse al servidor de seguimiento.

Experimento utilizado:

`AI4I_Parameterized_Experiments`

## 7. Entrenamiento

Ejecutar con parámetros predeterminados:

```powershell
python train.py
```

Argumentos disponibles:

| Argumento | Valor predeterminado | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la fuerza de regularización |
| `--max-iter` | 1000 | Número máximo de iteraciones |
| `--class-weight` | none | Ponderación de clases |
| `--seed` | 42 | Semilla aleatoria |

## 8. Experimentos ejecutados

```powershell
python train.py --C 1.0 --max-iter 1000 --class-weight none --seed 42
python train.py --C 0.1 --max-iter 1000 --class-weight none --seed 42
python train.py --C 10.0 --max-iter 1000 --class-weight none --seed 42
python train.py --C 1.0 --max-iter 1000 --class-weight balanced --seed 42
python train.py --C 0.1 --max-iter 1000 --class-weight balanced --seed 42
```

Todos los experimentos utilizan la misma división estratificada de datos: 80% entrenamiento y 20% prueba.

## 9. Resultados comparativos

| Experimento | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| E1 | 0.9675 | 0.6364 | 0.1029 | 0.1772 |
| E2 | 0.9685 | 0.8571 | 0.0882 | 0.1600 |
| E3 | 0.9675 | 0.6154 | 0.1176 | 0.1975 |
| E4 | 0.8245 | 0.1418 | 0.8235 | 0.2419 |
| E5 | 0.8230 | 0.1389 | 0.8088 | 0.2371 |

El experimento E4 obtuvo el mayor recall y F1-score, detectando 56 de las 68 fallas reales del conjunto de prueba.

Sin embargo, también generó 339 falsas alarmas, lo que representa una limitación importante para su posible aplicación industrial.

## 10. Reproducibilidad

Se repitió E4 utilizando los mismos hiperparámetros y la semilla `42`.

Las métricas y la matriz de confusión coincidieron con la corrida original dentro del mismo entorno.

Las dependencias principales se encuentran fijadas en `requirements.txt`.

## 11. Evidencias

La carpeta `evidence/` contiene las capturas de MLflow utilizadas para documentar el seguimiento y la comparación de experimentos.

## 12. Limitaciones

- Dataset con fuerte desbalance de clases.
- Evaluación realizada sobre una única división entrenamiento/prueba.
- El incremento de recall implica un aumento considerable de falsas alarmas.
- No se realizó validación cruzada ni evaluación sobre datos industriales nuevos.
- Los resultados no representan una validación para producción.

## 13. Uso de inteligencia artificial

Se utilizó ChatGPT como herramienta de apoyo para comprender conceptos de Machine Learning y MLOps, estructurar el código, resolver errores de ejecución y apoyar la documentación. La ejecución de los experimentos y la revisión de los resultados se realizaron en el entorno local de trabajo.