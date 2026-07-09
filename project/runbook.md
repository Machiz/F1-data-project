# Guía de Ejecución y Reproducibilidad (Runbook) — F1 Strategic Decision Engine

Este documento proporciona las instrucciones paso a paso para configurar el entorno de desarrollo, descargar los datos a través de la API OpenF1, ejecutar el pipeline de procesamiento de datos y reproducir los experimentos de entrenamiento de modelos y análisis de grafos con resultados consistentes y deterministas.

---

## 🛠️ 1. Configuración del Entorno de Trabajo

Para garantizar la reproducibilidad científica y evitar conflictos entre librerías, todas las ejecuciones deben realizarse bajo el mismo entorno virtual de Python.

### Requisitos Previos:
*   **Python:** Versión `3.10` o superior (se recomienda 3.10).
*   **Directorio de Trabajo:** Todos los comandos detallados en este Runbook deben ejecutarse desde la carpeta `project/` del repositorio:
    ```bash
    cd project
    ```

### Pasos de Instalación:

1.  **Crear el entorno virtual (venv):**
    ```bash
    python -m venv venv
    ```

2.  **Activar el entorno virtual:**
    *   **En Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **En macOS/Linux (Bash/zsh):**
        ```bash
        source venv/bin/activate
        ```

3.  **Instalar dependencias requeridas:**
    El archivo `requirements.txt` incluye las versiones estables de las librerías principales (`pandas`, `polars`, `xgboost`, `scikit-learn`, `networkx`, `requests`, `pyarrow`, `joblib`, `jupyter` y `streamlit` para el demo interactivo). Instálalas ejecutando:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## ♻️ Nota importante: reconstrucción tras clonar el repositorio

El repositorio **no versiona los modelos entrenados (`models/*.pkl`, `*.joblib`) ni los datos procesados pesados (`data/processed/recommendation/*.parquet`)**, ya que superan el límite de tamaño de GitHub y, además, se regeneran de forma determinista a partir de los datos crudos.

En consecuencia, tras un `git clone` la carpeta `models/` y los parquets de recomendación **no existen todavía**. Antes de ejecutar la auditoría o el demo, es obligatorio reconstruirlos corriendo la cadena completa de los Pasos 1 a 5 (secciones 2 a 6). Un clon recién descargado no tiene modelos entrenados hasta que se ejecuta esa cadena.

Si solo se desea reconstruir el subsistema de recomendación (asumiendo que las features de telemetría ya existen), basta con ejecutar la sección 6 en su orden estricto.

---

## 🏎️ 2. Paso 1: Ingesta de Datos Crudos (E-L)

El primer paso descarga la telemetría, tiempos de vuelta, neumáticos e intervalos desde la API oficial de OpenF1. El script cuenta con un sistema de reintentos exponenciales y estrangulamiento de peticiones para evitar bloqueos del servidor (Error 429).

Ejecuta el script de extracción indicando las 4 carreras de la temporada 2026:
```bash
python src/data_extraction/extract_f1_data.py --year 2026 --races Australia China Japan "United States"
```

*   **Entrada:** Llamadas REST a la API `https://api.openf1.org/v1`.
*   **Salida:** Archivos CSV unificados en `data/raw/` estructurados por carrera:
    *   `data/raw/australia_2026/laps.csv`, `pit.csv`, `stints.csv`, `car_data.csv`, `weather.csv`, `drivers.csv`
    *   `data/raw/china_2026/...`
    *   `data/raw/japan_2026/...`
    *   `data/raw/united_states_2026/...`

---

## 🧹 3. Paso 2: Preprocesamiento y Extracción de Eventos

Este pipeline unifica los CSVs crudos de telemetría y sensores, corrige inconsistencias de posición mediante el tiempo acumulado de carrera y genera un archivo consolidado de interacciones tácticas (Adelantamientos y Pit Stops).

Ejecuta el pipeline de eventos:
```bash
python src/features/f1_events_pipeline.py
```

*   **Entradas:** Archivos CSV de `data/raw/`.
*   **Salidas:**
    *   **Master Parquets:** `data/processed/master/{carrera}_master.parquet` (1 fila = 1 vuelta de 1 piloto).
    *   **Events Parquets:** `data/processed/events/{carrera}_events.parquet` (1 fila = 1 interacción/evento táctico).

---

## 📈 4. Paso 3: Feature Engineering y Reducción Dimensional

Dado que las siguientes etapas se realizan mediante cuadernos de Jupyter para permitir análisis visuales, abre Jupyter Notebook e inicia el servidor:
```bash
jupyter notebook
```

Ejecuta secuencialmente los siguientes notebooks:

### A. Ingeniería de Características
Abre y ejecuta todas las celdas de `notebooks/feature engineering/Feature_engineering_v5.ipynb`.

*   **Objetivo:** Divide el espacio de datos en Capa A (Telemetría) y Capa B (Táctica).
*   **Salida:** `data/processed/features/telemetry_features_v4.parquet` y `tactical_features_v4.parquet`.

### B. PCA (Reducción Dimensional Lineal)
Abre y ejecuta todas las celdas de `notebooks/dimensionality reduction/PCA_v4.ipynb`.

*   **Objetivo:** Reduce las 24 variables numéricas de telemetría a 6 componentes principales ortogonales.
*   **Salida:** `data/processed/features/telemetry_pca_v4.parquet`.

### C. t-SNE (Embeddings de Manifold Learning)
Abre y ejecuta todas las celdas de `notebooks/dimensionality reduction/tSNE_Embeddings_Manifold_Learning.ipynb`.

*   **Objetivo:** Proyecta eventos tácticos de alta dimensionalidad en espacios de 2D y 3D.
*   **Salida:** `data/processed/features/tactical_embeddings.parquet`.

---

## 🔬 5. Paso 4: Análisis de Clustering

Para replicar y validar la segmentación no supervisada de estados de rendimiento físico del monoplaza, ejecuta los tres notebooks comparativos en la carpeta `notebooks/clustering models/`:

1.  **K-Means V2:** Ejecuta `K_Means_Clustering_V2_Telemetry_PCA.ipynb`.
2.  **Hierarchical Clustering:** Ejecuta `Hierarchical_Clustering_Telemetry_PCA.ipynb`.
3.  **DBSCAN V3:** Ejecuta `DBSCAN_V3_Telemetry_PCA.ipynb`.

*   **Entrada:** `data/processed/features/telemetry_pca_v4.parquet`.
*   **Salida:** Evaluaciones de cohesión y separación de clústeres.

---

## 🎯 6. Paso 5: Pipeline del Recomendador y Sistema de Ranking

El núcleo del motor de decisión está compuesto por una arquitectura híbrida de dos capas desacopladas (predicción física en la Capa 1 y ranking point-wise en la Capa 2).

### Formulación del target: siete acciones con NO_PIT

Cada grupo de decisión (carrera, piloto, vuelta) genera **siete candidatos**:

*   `wait_laps = 0 … 5`: parar en boxes tras esperar *w* vueltas.
*   `wait_laps = 6` → **NO_PIT / STAY_OUT**: no parar en la ventana de las próximas 5 vueltas.

La acción `NO_PIT` es explícita y recibe la etiqueta ganadora (`0.0`) en las vueltas donde no hubo una parada real dentro de la ventana; los offsets `0 … 5` reciben `-2.0` salvo el que coincide con una parada real, que recibe su `success_score`. Esta formulación corrige un sesgo previo en el que `wait_laps = 0` obtenía la mejor etiqueta por defecto, forzando al ranker a aprender la regla trivial "parar de inmediato". Con `NO_PIT` como clase propia, "quedarse fuera" se aprende como una decisión deliberada y no como un artefacto del etiquetado.

### Orden estricto de ejecución

Restricciones de dependencia: `update_candidates_cost.py` requiere que la Capa 1 ya esté entrenada; la Capa 2 requiere los candidatos con el costo ya inyectado. La generación de candidatos no depende de la Capa 1 (inicializa el costo en `0.0`), por lo que puede ejecutarse antes o después de entrenarla, siempre antes del puente de costo.

#### 6.1. Generar Candidatos del Recomendador (Capa C)
Expande la telemetría agregando tráfico temporal en ventana móvil, genera los siete candidatos por grupo y asigna el target de éxito (incluida la acción `NO_PIT`):
```bash
python src/features/f1_recommender_pipeline.py
```
*   **Salida:** `data/processed/recommendation/pit_decision_candidates_v1.parquet` (7 filas por grupo de decisión).

#### 6.2. Entrenar el Modelo Físico de Degradación (Capa 1)
Entrena el ensamble por Stacking (XGBoost + Extra Trees → Ridge Regression) con validación cruzada GroupKFold por carrera:
```bash
python src/models/train_regression_layer1.py
```
*   **Salida:** `models/regression_layer1_model.pkl` y la metadata de alineación `models/regression_features.joblib`.

#### 6.3. Calcular el Puente de Costo Estratégico
Predice los tiempos futuros de permanencia y genera el coste acumulado en segundos por cada ventana de espera. El candidato `NO_PIT` (`wait_laps = 6`) se acota con `clip(upper=5)` para representar la permanencia durante toda la ventana de 5 vueltas, no una espera literal de 6:
```bash
python src/models/update_candidates_cost.py
```
*   **Salida:** Actualiza `pit_decision_candidates_v1.parquet` inyectando la columna `predicted_cost_of_staying`.

#### 6.4. Entrenar el Ranker de Decisiones (Capa 2)
Entrena el Point-wise Ranker (Random Forest Regressor) para priorizar las **siete acciones** (6 offsets de parada + `NO_PIT`) y guardarlo en producción:
```bash
python src/models/train_ranking_layer2.py
```
*   **Salida:** `models/ranking_layer2_model.pkl`. La evaluación de utilidad y sesgo se realiza en el Paso 5.5 (auditoría), que es la métrica de referencia del sistema.

#### 6.5. Auditar el Sesgo del Ranker (verificación, no entrenamiento)
Evalúa el desempeño por clase, el baseline trivial y la accuracy en los grupos donde una parada era óptima:
```bash
python src/models/audit_ranking_bias.py
```
*   **Salida:** `reports/ranking_system/ranking_bias_audit.md`. Ver la sección 9 de este runbook para la interpretación de las cifras.

#### 6.6. Lanzar el Demo Interactivo (opcional)
Con los modelos ya entrenados, ejecuta el asistente táctico en tiempo real simulado (muro de boxes):
```bash
streamlit run demo/realtime_demo/app_streamlit.py
```
*   El banner distingue las tres acciones: **BOX** (parar ahora), **STAY** (ventana óptima en *k* vueltas) y **STAY OUT / NO_PIT** (no parar en la ventana).
*   Requiere `streamlit` instalado (incluido en `requirements.txt`).

---

## 🕸️ 7. Paso 6: Construcción y Análisis de Grafos

Genera las centralidades de PageRank, Betweenness y Componentes Conexas para ambos grafos de combate y DRS:

```bash
# Grafo de Adelantamientos Rueda a Rueda
python src/graphs/graph_construction.py

# Grafo de Proximidad Física e Intervalos DRS
python src/graphs/drs_graph_construction.py
```

*   **Entrada:** Archivos Parquet de `data/processed/events/` y `data/raw/`.
*   **Salida:** Métricas de dominancia y agrupamientos impresas en consola e integradas en los reportes de grafos.

---

## 🔒 8. Verificación de Reproducibilidad y Consistencia

Para asegurar que los resultados no varíen entre ejecuciones independientes o distintas máquinas, se implementaron los siguientes controles en el código:

1.  **Semilla Aleatoria Fijada (`random_state=42`):**
    Todos los estimadores no deterministas o basados en particiones estocásticas están configurados con una semilla fija. Esto incluye:
    *   `train_regression_layer1.py` (XGBRegressor y ExtraTreesRegressor con `random_state=42`).
    *   `train_ranking_layer2.py` (RandomForestRegressor con `random_state=42`).
    *   Notebooks de Clustering (K-Means y DBSCAN inicializados con semillas constantes).
2.  **Cero Fuga de Información (*No Lookahead Bias*):**
    La validación de la Capa 1 y la Capa 2 se realiza sobre conjuntos de prueba agrupados por circuito (`GroupKFold`), asegurando que las métricas de rendimiento en producción simulen correctamente la llegada a una pista completamente nueva y desconocida. El demo replica esta condición filtrando únicamente la telemetría hasta la vuelta consultada (sin datos futuros).
3.  **Integridad de Datos en Inferencia:**
    El puente de datos `update_candidates_cost.py` y el pipeline en tiempo real (`realtime_pipeline.py`) utilizan `regression_features.joblib` para forzar la misma alineación de columnas numéricas y dummies de la Capa 1. La generación de candidatos de entrenamiento e inferencia está espejada (siete candidatos en ambos casos, con el mismo `clip(upper=5)` en el costo), garantizando que el ranker vea en producción la misma escala de `predicted_cost_of_staying` que aprendió en entrenamiento.

---

## 📊 9. Limitaciones Conocidas y Resultados de Auditoría

Esta sección documenta con transparencia el desempeño real del recomendador, medido por `audit_ranking_bias.py` sobre 3331 grupos de decisión.

### Resultados de la auditoría

| Métrica | Valor |
|---|---|
| Accuracy global (acción exacta) | 0.8775 |
| Baseline "siempre NO_PIT" | 0.8898 |
| **Accuracy de decisión binaria (parar vs no parar)** | **0.8937** |
| Grupos con parada óptima real (óptimo ≠ NO_PIT) | 367 |
| Accuracy binaria en esos grupos (detecta que hay que parar) | 0.3678 |
| Accuracy exacta en esos grupos (offset correcto) | 0.2207 |

### Interpretación honesta

La accuracy global (0.8775) queda ligeramente **por debajo** del baseline trivial "siempre NO_PIT" (0.8898). Esto **no es una regresión ni un error**: es el costo esperado de que el modelo *intente* proponer paradas. Un modelo que nunca para jamás se equivoca en los grupos de quedarse fuera, pero tampoco acierta nunca una parada. Por ello, la accuracy global dejó de ser el indicador de referencia.

El indicador correcto es la **decisión binaria parar / no parar (0.8937), que sí supera al baseline (0.8898)**. El modelo tiene señal neta positiva, aunque el margen sea reducido. Descomponiéndola: cuando el modelo recomienda quedarse fuera acierta cerca del 96% de las veces (alta especificidad, rara vez inventa una parada), pero de las 367 ventanas de parada realmente óptimas solo detecta el 36.78% (baja sensibilidad), y de esas acierta el offset exacto en el 22.07%.

En resumen: el sesgo estructural previo —el 97% de las predicciones colapsando en `wait_laps = 0`— **quedó eliminado**. El modelo pasó de no superar al baseline trivial a superarlo en la decisión relevante. La limitación pendiente es la baja sensibilidad, esperable dada la escasez y el ruido de las ventanas de parada reales y la naturaleza proxy del `success_score`.

### Hoja de ruta de mejora (de menor a mayor esfuerzo)

1.  **Ponderación de muestras (`sample_weight`) en la Capa 2:** dar más peso a los grupos con parada real para elevar la sensibilidad a costa de la accuracy global (que ya no es prioritaria). Experimento reversible de bajo costo.
2.  **Margen en la inferencia:** recomendar `NO_PIT` solo si su score supera al mejor candidato de parada por un umbral; mueve el punto de operación hacia más llamadas de pit sin reentrenar.
3.  **Modelo en dos etapas:** un clasificador binario "¿parar en la ventana?" seguido del ranker de offset únicamente sobre los grupos con predicción de parada. Ataca el desbalance de raíz.
4.  **Mejores etiquetas / PPO:** el `success_score` proxy es el techo del sistema supervisado. La línea PPO modela nativamente la decisión secuencial de parada (incluida la de quedarse fuera) evaluando la recompensa de la carrera simulada, resolviendo el contrafactual sin depender de observaciones históricas. Es la vía de mayor rigor y mayor esfuerzo; actualmente existe solo la búsqueda de hiperparámetros (`ppo_best_hyperparameters.joblib`), sin agente entrenado.