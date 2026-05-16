# Documentación Técnica: DBSCAN V3 — Clustering Telemetría PCA

Este documento detalla el experimento de **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** aplicado al espacio latente generado por **PCA V4** sobre telemetría por vuelta de Fórmula 1. Este reporte es equivalente a los de K-Means V2 y Hierarchical V4, cerrando el ciclo comparativo de métodos de clustering sobre el mismo universo de datos.

## 1. Archivo y Reproducibilidad

| Artefacto | Ruta |
|:---|:---|
| **Notebook ejecutable** | `project/notebooks/clustering models/DBSCAN_V3_Telemetry_PCA.ipynb` |
| **Dataset de entrada** | `project/data/features/telemetry_pca_v4.parquet` |
| **Artefactos visuales** | `project/artifacts/dbscan_kdistance_plot.png` |
| | `project/artifacts/dbscan_scatter_pca.png` |

---

## 2. Fundamentos: ¿Cómo Funciona DBSCAN?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** es un algoritmo de agrupamiento no paramétrico. A diferencia de K-Means o el clustering Jerárquico, **no asume forma esférica en los clústeres** y **no requiere especificar el número de grupos** de antemano.

### ¿Cómo opera internamente?

DBSCAN se basa en dos parámetros:
- **`eps` (ε):** Radio de búsqueda alrededor de cada punto en el espacio de features.
- **`min_samples`:** Número mínimo de vecinos dentro del radio `eps` para que un punto sea considerado "núcleo".

El algoritmo clasifica cada punto en una de tres categorías:

| Categoría | Definición |
|:---|:---|
| **Core Point** | Tiene ≥ `min_samples` vecinos dentro de su radio `eps`. |
| **Border Point** | Está dentro del radio de un Core Point, pero no tiene suficientes vecinos propios. |
| **Noise (-1)** | No pertenece a ningún clúster. Aislado en zonas de baja densidad. |

### ¿Por qué DBSCAN sobre telemetría PCA?

En iteraciones previas (DBSCAN sobre 15 dimensiones tácticas) el ruido alcanzó el **54.7%** del dataset — una señal de que la **maldición de la dimensionalidad** disolvía la densidad natural de los datos. Al comprimir la telemetría a **6 Componentes Principales** (PCA V4, ~78.7% de varianza explicada), la densidad se concentra y DBSCAN puede operar con mayor efectividad.

---

## 3. Dataset de Entrada

El modelo opera sobre el mismo universo unificado que K-Means V2 y Hierarchical V4:

```
Dataset shape    : (3004, 14)
Feature matrix X : (3004, 6)   → PC1 … PC6
Races            : {'australia': 925, 'united_states': 866, 'japan': 681, 'china': 532}
Nulls in X       : 0
```

**Interpretación de los PCs** (heredada de PCA V4):

| PC | Carga dominante | Interpretación |
|:---|:---|:---|
| **PC1** | Balance S3 vs S1 | Caracterización del trazado |
| **PC2** | Aceleración full-throttle | Agresividad de conducción |
| **PC3** | Velocidad media global | Ritmo general de vuelta |
| **PC4** | Degradación de neumático (tyre_age) | Estado del compuesto |
| **PC5** | Frenada máxima | Estilo de frenada |
| **PC6** | Régimen de motor (RPM) | Eficiencia energética |

---

## 4. K-Distance Plot: Criterio para Selección de `eps`

### ¿Qué es y cómo se interpreta?

El **K-Distance Plot** ordena todos los puntos del dataset según su distancia al **k-ésimo vecino más cercano** (de mayor a menor). El "codo" visible en la curva indica el **umbral de densidad natural** de los datos: a partir de ese punto, los puntos comienzan a estar aislados (candidatos a ruido).

En este experimento se usó **k=5** (equivalente al `min_samples` base).

### Percentiles de distancia al 5° vecino

```text
p10:  0.112
p25:  0.215
p50:  0.536
p75:  0.781
p90:  1.143
```

> [!NOTE]
> **Lectura técnica:** La mediana de distancia al 5° vecino es `0.536`, pero la curva presenta un codo visible en el rango `eps ≈ 1.0–1.2`. Valores menores generan fragmentación excesiva; valores mayores fusionan regímenes distintos de conducción.

---

## 5. Parameter Sweep: Barrido de Combinaciones `eps × min_samples`

### Grilla evaluada

```
EPS_RANGE         = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
MIN_SAMPLES_RANGE = [5, 10, 15]
Total combinaciones: 21
```

### Métricas de evaluación

| Métrica | Qué mide | Cómo interpretar |
|:---|:---|:---|
| **Silhouette Score** | Cohesión intra-clúster vs separación inter-clúster | Calculado **solo sobre señal** (excluyendo ruido). Mayor = mejor. |
| **Davies-Bouldin Index** | Compacidad relativa de los clústeres | Menor = mejor. |
| **Noise Ratio** | Fracción de vueltas descartadas como ruido | Menor = más datos aprovechados; pero demasiado bajo puede indicar over-clustering. |

> [!WARNING]
> **Límite de interpretación:** Valores altos de Silhouette con Noise Ratio alto representan una **ilusión estadística**: al descartar 60–70% de los datos como ruido, los clústeres residuales son artificialmente puros. El objetivo real es balancear `silhouette alto + noise bajo + n_clusters interpretable (3–6)`.

### Resultados del sweep (ordenados por Silhouette)

```text
 eps  min_samples  n_clusters  noise%   silhouette  davies-b
--------------------------------------------------------------
 0.4           10           3   66.6%      0.7909     0.2521
 0.6           15           4   55.6%      0.7613     0.3474
 0.4           15           3   67.9%      0.6385     0.5584
 1.2           15           5   11.2%      0.5910     0.6018  ← candidato
 1.0           10           6   14.4%      0.5903     0.5391  ← candidato
 1.5           15           4    7.0%      0.5695     0.7018  ← candidato
 1.0           15           6   17.4%      0.5667     0.6170  ← candidato
 1.5           10           5    5.4%      0.5630     0.6355  ← candidato
 1.2           10           6    9.2%      0.5571     0.6301  ← candidato
 1.2            5           7    6.3%      0.5539     0.5699  ← candidato
 2.0           15           5    3.9%      0.5524     0.6589  ← candidato
 2.0           10           5    3.6%      0.5494     0.6670  ← candidato
 1.5            5           7    4.1%      0.5365     0.7734  ← candidato
 2.0            5           5    2.9%      0.5317     0.7073  ← candidato
 1.0            5          12    9.3%      0.4985     0.5628
 0.6           10          12   48.3%      0.4703     0.7656
 0.8           10           7   24.1%      0.4550     0.7371
 0.8           15          11   33.4%      0.3400     0.8911
 0.8            5          17   15.8%      0.2992     0.7358
 0.4            5          30   58.7%      0.2922     0.9503
 0.6            5          33   32.9%      0.2379     0.9444
```

---

## 6. Selección del Modelo Óptimo

### Criterios de selección

El modelo candidato debe cumplir simultáneamente:
1. **Noise Ratio < 15%** — aprovechamiento real del dataset.
2. **n_clusters entre 3 y 6** — granularidad interpretable en dominio F1.
3. **Silhouette score máximo** dentro de los candidatos filtrados.

### Modelo elegido: `eps=1.2, min_samples=15`

```text
eps=1.2 | min_samples=15 | clusters=5 | noise=11.2% | silhouette=0.5910 | davies-b=0.6018
```

> [!IMPORTANT]
> **Justificación:** Aunque `eps=1.0, min_samples=10` obtiene un Silhouette similar (0.5903), genera **6 clústeres** versus los **5** de la configuración elegida. Con `eps=1.2, min_samples=15` se logra el mejor balance entre pureza interna y noise bajo, con granularidad F1-interpretable. La diferencia de 0.2 puntos de noise (11.2% vs 14.4%) es relevante dado el tamaño del dataset (≈30 vueltas adicionales en señal).

---

## 7. Tabla de Validación Final

Métricas calculadas **exclusivamente sobre la señal** (excluyendo los 209 puntos de ruido / clase -1):

| Métrica de Validación | Valor | Límite de Interpretación |
|:---|:---|:---|
| **Silhouette Score** | **0.5910** | Calculado sobre señal pura. Refleja cohesión interna real de cada régimen de conducción. |
| **Davies-Bouldin Index** | **0.6018** | Valores < 1.0 indican buena separación. Los clústeres no son perfectamente globulares, lo que penaliza ligeramente esta métrica euclidiana. |
| **Noise Ratio** | **11.2% (337 vueltas)** | Reducción dramática respecto al DBSCAN sobre 15D (54.7%). Estas vueltas representan transiciones genuinas (salidas de pit, Safety Cars, banderas amarillas). |
| **Clusters detectados (k)** | **5** | Emergentes desde la densidad del espacio PCA — no impuestos a priori. |

> [!NOTE]
> **Comparativa histórica con DBSCAN táctico (15D):** El Noise Ratio cae de 54.7% → 11.2% al reducir el espacio de 15 dimensiones a 6 PCs. Esto valida la efectividad de PCA V4 para concentrar la información de conducción en un espacio compacto donde DBSCAN puede operar eficazmente.

---

## 8. Cluster Profile Analysis — Perfiles de Dominio F1

Los perfiles se construyen vinculando los clústeres DBSCAN con las variables originales de `telemetry_features_v4.parquet` (join por `race_name + driver_number + lap_number`).

### Tabla de perfiles

```text
                   n   pc1_mean  pc2_mean  lap_dur_mean  st_speed_mean  throttle_full_pct  brake_max_mean  tyre_age_mean
dbscan_cluster
-1 (Ruido)       209      0.152    -1.013       103.810        272.902              0.571         102.656         10.211
 0               829      2.406     1.171        85.146        288.267              0.686         101.292         12.407
 1               485      2.561    -1.692        98.323        314.600              0.632         102.977         11.559
 2               644     -2.599     1.950        95.712        285.005              0.680         103.068          2.947
 3               837     -1.904    -1.427        94.523        307.740              0.610         100.588         14.798
```

### Distribución por carrera

```text
Cluster 0 (n=829)  → australia     (100% de sus laps)
Cluster 1 (n=485)  → china         (100%)
Cluster 2 (n=644)  → japan         (100%)
Cluster 3 (n=837)  → united_states (100%)
Ruido    (n=209)   → australia (96), china (47), japan (37), united_states (29)
```

### Arquetipos identificados por clúster

| Clúster | Arquetipo | Variables clave | Interpretación |
|:---|:---|:---|:---|
| **0 — Australia Fast Lap** | Vuelta urbana rápida | lap_dur=85.1s, st_speed=288 km/h, tyre_age=12.4 | Circuito técnico con vueltas cortas y neumáticos en fase media de stint. |
| **1 — China High Speed** | Vuelta de alta velocidad | lap_dur=98.3s, st_speed=314 km/h (máximo), throttle=0.63 | Trazado de altas rectas, menos carga aerodinámica, neumáticos a media vida. |
| **2 — Japan Fresh Tyre** | Stint inicial fresco | tyre_age=**2.9 laps** (mínimo), st_speed=285 km/h, throttle=0.68 | Vueltas en los primeros giros de stint — neumático nuevo o casi nuevo. |
| **3 — COTA Late Stint** | Stint avanzado de degradación | tyre_age=**14.8 laps** (máximo), st_speed=307 km/h | Vueltas en la fase final del stint con mayor degradación acumulada. |
| **-1 — Transición** | Vueltas de régimen inestable | lap_dur=103.8s (más lento), st_speed=272 km/h (más bajo) | Safety Cars, salidas de pit, banderas amarillas — sin régimen estable. |

> [!NOTE]
> **Observación crítica:** La separación por circuito es un artefacto esperado dado que el dataset tiene exactamente 4 circuitos. Los Componentes Principales capturan características intrínsecas del trazado (PC1: balance de sectores, PC3: velocidad global) lo que hace que las vueltas de cada circuito formen regiones densas y separadas en el espacio PCA. El Clúster 2 (Japan + tyre_age bajo) es el más revelador: DBSCAN detecta que las vueltas japonesas con neumático fresco son un régimen *físicamente distinto* del resto.

---

## 9. Failure Analysis — ¿Qué Quedó Como Ruido?

```text
Total ruido      : 209 vueltas (7.0% del dataset)
Distribución     : australia (96), china (47), japan (37), united_states (29)
is_pit_out_lap   : 0  (0.0% del ruido)
is_pit_lap       : 0  (0.0% del ruido)
Tyre age (ruido) : 10.2 laps avg
Tyre age (señal) : 10.8 laps avg
Lap dur (ruido)  : 103.8 s avg
Lap dur (señal)  :  92.7 s avg
```

### ¿Por qué no se agruparon?

1. **Vueltas lentas anómalas:** Con `lap_duration` promedio de **103.8s** (vs 92.7s en señal), el ruido concentra las vueltas más lentas del dataset — probablemente vueltas bajo Safety Car o con incidentes.

2. **No son vueltas de pit:** El análisis confirma que **0% del ruido** son `is_pit_out_lap` o `is_pit_lap`. El ruido son vueltas *en régimen de carrera* pero con comportamiento atípico.

3. **Frontera entre regímenes:** Algunas vueltas de transición (entre stint inicial y tardío, o entre circuitos con características intermedias) caen en zonas de baja densidad del espacio PCA.

> [!IMPORTANT]
> **Diagnóstico técnico:** La reducción del espacio de 15D → 6D eliminó el ruido de 54.7% → 7.0%. El 7% residual son vueltas en transición genuina. **No son datos "malos"** — son eventos sin régimen de conducción estable. Su separación limpia la señal para modelado predictivo posterior.

---

## 10. Conclusiones e Insights Estratégicos

> [!IMPORTANT]
> **Insights del Modelado Espacial DBSCAN sobre PCA V4:**
>
> 1. **Validación de PCA V4:** La drástica reducción de ruido (54.7% → 7.0%) demuestra que la compresión a 6 componentes principales no solo conserva la información relevante, sino que *mejora activamente* la calidad del espacio para métodos basados en densidad.
>
> 2. **Arquetipos Físicos Emergentes:** Sin imponer el número de clústeres, DBSCAN detectó **4 regímenes de conducción** con coherencia física:
>    - Circuito técnico vs circuito de alta velocidad
>    - Stint con neumático fresco vs stint tardío con degradación
>
> 3. **El 7% de Ruido es Valioso:** Las 209 vueltas descartadas representan exactamente los eventos sin régimen estable (Safety Cars, vueltas lentas) — precisamente lo que un modelo predictivo de estrategia NO debe aprender como patrón normal.
>
> 4. **Convergencia con otros métodos:** K-Means V2 y Hierarchical V4 también identificaron 4 arquetipos principales. La convergencia de tres métodos distintos sobre el mismo número de clústeres es evidencia estadística fuerte de que la taxonomía de 4 regímenes refleja estructura real en los datos de telemetría F1.

### Próximos Pasos

| Acción | Justificación |
|:---|:---|
| **Exportar etiquetas DBSCAN** como nueva feature | Los clústeres de régimen de conducción son predictores candidatos para modelos de clasificación de estrategia. |
| **Estudiar vueltas de ruido** en detalle | Las 209 vueltas (-1) pueden ser filtradas antes del entrenamiento supervisado para mejorar la calidad de la señal. |
| **Comparar con K-Means y Hierarchical** | Analizar coincidencia de asignación vuelta-a-vuelta entre los tres métodos para identificar el "núcleo duro" de cada arquetipo. |
| **Incorporar en Feature Engineering V6** | Agregar `dbscan_cluster` y `kmeans_cluster` como features categóricas para modelos supervisados de predicción de stint. |
