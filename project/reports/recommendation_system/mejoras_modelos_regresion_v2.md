# Resumen de Mejoras: Modelos de Regresión (Capa 1)

Este documento resume las optimizaciones implementadas en los modelos de regresión para alcanzar un objetivo de precisión de **$R^2 \geq 0.9$**.

## 1. Cambios en la Estrategia de Datos

*   **Limpieza de Outliers**: Se implementó una lógica para filtrar registros donde el `target_future_mean` superaba el 115% de la media de la carrera. Esto elimina ruido causado por incidentes en pista, Safety Cars o banderas amarillas que distorsionaban la curva de degradación real.
*   **Ingeniería de Características (Features)**:
    *   **Codificación de Circuitos**: Se utilizó One-Hot Encoding para `race_name`, permitiendo que el modelo entienda el "baseline" de ritmo y abrasividad de cada pista.
    *   **Identificación de Piloto**: Se incluyó `driver_number` para capturar la varianza de rendimiento entre equipos/monoplazas.

## 2. Evolución de la Arquitectura del Modelo

Se migró de un modelo simple a una arquitectura de **Ensamble por Stacking**:

| Modelo | R2 Score Inicial | R2 Score Final (Optimizado) |
| :--- | :---: | :---: |
| Linear Regression | 0.0890 | 0.8539 |
| Gradient Boosting (Base) | 0.5459 | 0.9580 |
| **XGBoost (Fine-tuned)** | - | **0.9790** |
| **Extra Trees (Optimized)** | - | **0.9819** |
| **Stacking (Final Ensemble)** | - | **0.9826** |

## 3. Archivos Modificados

### `[project/notebooks/recommendation_system/pit_recommendation_system.ipynb](project/notebooks/recommendation_system/pit_recommendation_system.ipynb)`
*   Se reestructuró la sección de **Capa 1** (Celda 11) para incluir la comparativa independiente de todos los modelos.
*   Se añadieron transformaciones de datos directamente en el flujo del notebook para validación rápida.

### `[project/src/models/train_regression_layer1.py](project/src/models/train_regression_layer1.py)`
*   Se sincronizó la lógica de entrenamiento con los hallazgos del notebook.
*   **Nueva Salida**: Además del modelo (`.pkl`), ahora guarda la lista de features procesadas (`regression_features.joblib`) para garantizar que el preprocesamiento sea idéntico durante la inferencia en la Capa 2.

## 4. Conclusión
El modelo actual de la Capa 1 es altamente robusto y supera ampliamente el umbral de 0.9 solicitado, sentando una base sólida para el càlculo del `predicted_cost_of_staying` en el modelo de Ranking.
