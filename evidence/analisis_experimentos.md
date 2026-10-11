# Análisis comparativo de experimentos — AI4I 2020

**Modelo:** Regresión logística  
**Herramienta:** MLflow  
**Experimentos:** Cinco configuraciones de hiperparámetros

## Objetivo

Evaluar el efecto de la regularización y la ponderación de clases sobre el desempeño de un modelo de clasificación para predecir fallas de maquinaria industrial, utilizando el dataset AI4I 2020 Predictive Maintenance.

## Resultados

| Experimento | C | Class weight | Accuracy | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|
| E1 | 1.0 | none | 0.9675 | 0.6364 | 0.1029 | 0.1772 |
| E2 | 0.1 | none | 0.9685 | 0.8571 | 0.0882 | 0.1600 |
| E3 | 10.0 | none | 0.9675 | 0.6154 | 0.1176 | 0.1975 |
| E4 | 1.0 | balanced | 0.8245 | 0.1418 | 0.8235 | 0.2419 |
| E5 | 0.1 | balanced | 0.8230 | 0.1389 | 0.8088 | 0.2371 |

## Configuración seleccionada

El experimento **E4 (`C=1.0`, `class_weight=balanced`, `max_iter=1000`, `seed=42`)** obtuvo el mejor recall (82.35%) y F1-score (24.19%) de las configuraciones evaluadas.

Este modelo detectó 56 de las 68 fallas reales del conjunto de prueba, frente a solamente 7 detectadas por el baseline E1. Sin embargo, produjo 339 falsas alarmas, por lo que su aplicación industrial requeriría evaluar el costo de inspecciones innecesarias frente al costo de fallas no detectadas.

## Hiperparámetro de mayor influencia

El hiperparámetro con mayor impacto observado fue **`class_weight`**. Al utilizar `balanced`, el recall aumentó considerablemente, mientras que la precisión disminuyó.

Las modificaciones de `C` generaron diferencias menores. No se evaluó el efecto de `max_iter`, ya que permaneció fijo en 1000.

## Reproducibilidad

Se repitió el entrenamiento E4 con la misma semilla (`42`), división estratificada 80/20, hiperparámetros y entorno de ejecución. Las métricas y la matriz de confusión coincidieron con la corrida original.

Las versiones principales de las dependencias se documentaron en `requirements.txt`.

## Limitaciones y conclusión

El dataset presenta un fuerte desbalance de clases (3.39% de fallas). Los experimentos se evaluaron sobre una única partición de prueba, sin validación cruzada ni datos externos.

Se concluye que E4 es el mejor candidato de los cinco bajo el criterio de priorizar la detección de fallas, pero **no está validado para producción**. Como siguiente paso se recomienda evaluar el umbral de clasificación, los costos de falsas alarmas y fallas omitidas, y la estabilidad del desempeño sobre datos nuevos.

**Evidencia:** Captura de la tabla comparativa de las cinco corridas en MLflow.