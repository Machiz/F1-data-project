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
    El archivo [requirements.txt](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/requirements.txt) incluye las versiones estables de las librerías principales (`pandas`, `polars`, `xgboost`, `scikit-learn`, `networkx`, `requests`, `pyarrow`, `joblib` y `jupyter`). Instálalas ejecutando:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

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
Abre y ejecuta todas las celdas de:  
[Feature_engineering_v5.ipynb](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/notebooks/feature%20engineering/Feature_engineering_v5.ipynb)

*   **Objetivo:** Divide el espacio de datos en Capa A (Telemetría) y Capa B (Táctica).
*   **Salida:** `data/processed/features/telemetry_features_v4.parquet` y `tactical_features_v4.parquet`.

### B. PCA (Reducción Dimensional Lineal)
Abre y ejecuta todas las celdas de:  
[PCA_v4.ipynb](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/notebooks/dimensionality%20reduction/PCA_v4.ipynb)

*   **Objetivo:** Reduce las 24 variables numéricas de telemetría a 6 componentes principales ortogonales.
*   **Salida:** `data/processed/features/telemetry_pca_v4.parquet`.

### C. t-SNE (Embeddings de Manifold Learning)
Abre y ejecuta todas las celdas de:  
[tSNE_Embeddings_Manifold_Learning.ipynb](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/notebooks/dimensionality%20reduction/tSNE_Embeddings_Manifold_Learning.ipynb)

*   **Objetivo:** Proyecta eventos tácticos de alta dimensionalidad en espacios de 2D y 3D.
*   **Salida:** `data/processed/features/tactical_embeddings.parquet`.

---

## 🔬 5. Paso 4: Análisis de Clustering

Para replicar y validar la segmentación no supervisada de estados de rendimiento físico del monoplaza, ejecuta los tres notebooks comparativos en la carpeta `notebooks/clustering models/`:

1.  **K-Means V2:** Ejecuta [K_Means_Clustering_V2_Telemetry_PCA.ipynb](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/notebooks/clustering%20models/K_Means_Clustering_V2_Telemetry_PCA.ipynb).
2.  **Hierarchical Clustering:** Ejecuta [Hierarchical_Clustering_Telemetry_PCA.ipynb](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/notebooks/clustering%20models/Hierarchical_Clustering_Telemetry_PCA.ipynb).
3.  **DBSCAN V3:** Ejecuta [DBSCAN_V3_Telemetry_PCA.ipynb](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/notebooks/clustering%20models/DBSCAN_V3_Telemetry_PCA.ipynb).

*   **Entrada:** `data/processed/features/telemetry_pca_v4.parquet`.
*   **Salida:** Evaluaciones de cohesión y separación de clústeres.

---

## 🎯 6. Paso 5: Pipeline del Recomendador y Sistema de Ranking

El núcleo del motor de decisión está compuesto por una arquitectura híbrida de dos capas desacopladas (Predicción física en la Capa 1 y Clasificación Point-wise en la Capa 2).

Sigue el orden estricto de comandos para reproducir el entrenamiento y las métricas de recomendación:

### 1. Generar Candidatos del Recomendador (Capa C)
Expande la telemetría agregando tráfico temporal en ventana móvil y el target de éxito:
```bash
python src/features/f1_recommender_pipeline.py
```
*   **Salida:** `data/processed/recommendation/pit_decision_candidates_v1.parquet`.

### 2. Entrenar el Modelo Físico de Degradación (Capa 1)
Entrena el ensamble por Stacking (XGBoost + Extra Trees -> Ridge Regression) con validación cruzada GroupKFold por carrera:
```bash
python src/models/train_regression_layer1.py
```
*   **Salida:** `models/regression_layer1_model.pkl` y la metadata de alineación `models/regression_features.joblib`.

### 3. Calcular el Puente de Costo Estratégico
Predice los tiempos futuros de permanencia y genera el coste acumulado en segundos por cada ventana de espera de pits:
```bash
python src/models/update_candidates_cost.py
```
*   **Salida:** Actualiza `pit_decision_candidates_v1.parquet` inyectando la columna `predicted_cost_of_staying`.

### 4. Entrenar el Ranker de Decisiones (Capa 2)
Entrena el Point-wise Ranker (Random Forest Regressor) para priorizar las 6 opciones de parada en boxes y guardarlo en producción:
```bash
python src/models/train_ranking_layer2.py
```
*   **Salida:** `models/ranking_layer2_model.pkl` (Evaluación final offline NDCG@1 = 89.74%).

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
    La validación de la Capa 1 y la Capa 2 se realiza sobre conjuntos de prueba agrupados por circuito (`GroupKFold`), asegurando que las métricas de rendimiento en producción simulen correctamente la llegada a una pista completamente nueva y desconocida.
3.  **Integridad de Datos en Inferencia:**
    El puente de datos `update_candidates_cost.py` utiliza `regression_features.joblib` para forzar la misma alineación de columnas numéricas y dummies de la Capa 1. Si hay diferencias en los circuitos cargados, el script los alinea y re-indexa dinámicamente con ceros, garantizando la consistencia del esquema.
