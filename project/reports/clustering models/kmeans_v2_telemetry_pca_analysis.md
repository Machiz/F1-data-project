# Documentación Técnica: K-Means V2 — Clustering Telemetría PCA

Este documento detalla el análisis exhaustivo de **K-Means Clustering** aplicado sobre los componentes principales (PCA V4) de la telemetría por vuelta de Fórmula 1. Esta versión incorpora métricas de validación avanzadas, visualización de centroides y un análisis de impacto de variables físicas sobre los clústeres detectados.

## 1. Archivo y Reproducibilidad

| Artefacto | Ruta |
|:---|:---|
| **Notebook ejecutable** | `project/notebooks/clustering models/K_Means_Clustering_V2_Telemetry_PCA.ipynb` |
| **Dataset de entrada** | `project/data/features/telemetry_pca_v4.parquet` |

---

## 2. Fundamentos: ¿Cómo Funciona K-Means y Para Qué Se Usa?

**K-Means Clustering** es un algoritmo de aprendizaje no supervisado que agrupa datos en `k` clústeres basándose en la proximidad al centroide más cercano. A diferencia de DBSCAN, K-Means **requiere especificar el número de clústeres de antemano** y **asume formas aproximadamente esféricas** en los grupos.

### ¿Cómo opera internamente?

1. Se inicializan `k` centroides aleatorios en el espacio de features.
2. Cada punto se asigna al centroide más cercano (minimizando distancia euclidiana).
3. Los centroides se recalculan como el promedio de los puntos asignados.
4. El proceso iterativo converge cuando los centroides dejan de moverse significativamente.

### ¿Por qué K-Means sobre telemetría PCA?

Al comprimir la telemetría a **6 Componentes Principales** (PCA V4, ~78.7% de varianza explicada), los datos están en un espacio compacto y normalizado que maximiza la efectividad de las distancias euclidianas que usa K-Means. Sin PCA, el alto número de variables y su correlación reducirían la calidad de los clústeres.

---

## 3. Dataset de Entrada

```
Dimensiones del Espacio Latente: (3004, 6)   → PC1 … PC6
Races                           : {'australia': 925, 'united_states': 866, 'japan': 681, 'china': 532}
Nulls in X                      : 0
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

## 4. Selección de K: El Balance entre Precisión y Explicabilidad

Para seleccionar `k`, el análisis utiliza **dos métricas complementarias** evaluadas sobre el rango `k ∈ [2, 9]`:

| Métrica | Qué mide | Cómo interpretar |
|:---|:---|:---|
| **Inercia (Codo)** | Suma de distancias cuadradas al centroide más cercano | Disminuye monótonamente con k. El "codo" marca la ganancia marginal decreciente. |
| **Silhouette Score** | Cohesión intra-clúster vs separación inter-clúster | Mayor = mejor separación. Pico = k óptimo. |

### Resultados del Barrido de K

```text
k  |  Inercia     |  Silhouette
-----------------------------
2  |  9842.1      |  0.3721
3  |  7234.5      |  0.4105
4  |  5891.2      |  0.4409  ← óptimo
5  |  5102.4      |  0.4201
6  |  4683.9      |  0.4057
7  |  4401.2      |  0.3884
8  |  4198.7      |  0.3672
9  |  4033.5      |  0.3451
```

### ¿Por qué k=4?

1. **Matemáticamente:** El gráfico de Inercia muestra un "codo" claro en `k=4`. A partir de ahí, la ganancia en cohesión interna disminuye su ritmo de mejora.
2. **Silhouette Score:** El pico se encuentra en `k=4` (~0.44), lo que indica que esta estructura maximiza la distancia inter-clúster mientras mantiene la cohesión intra-clúster.
3. **Justificación de Dominio:** Los 4 clústeres se alinean con los estados reales de un monoplaza en competición (ver perfiles a continuación).

---

## 5. Tabla de Validación Final

Métricas del modelo seleccionado con `k=4`:

| Métrica de Validación | Valor | Interpretación |
|:---|:---|:---|
| **Silhouette Score** | **0.4409** | Separación moderada-alta. Los clústeres presentan cohesión interna sólida en el espacio PCA. |
| **Inercia** | **5891.2** | Distancia total al centroide. El "codo" valida k=4 como punto de equilibrio. |
| **n_clusters** | **4** | Número óptimo por consenso multimétrico. |
| **n_init (iteraciones)** | **20** | Número de inicializaciones aleatorias para garantizar estabilidad del resultado. |

> [!NOTE]
> **Sobre la inicialización:** Con `n_init=20` y `random_state=42`, el modelo ejecuta 20 inicializaciones independientes y selecciona la que minimiza la inercia final, garantizando reproducibilidad y robustez.

---

## 6. Estructura de los Clústeres en el Espacio PCA

Los centroides (punto promedio de cada estado táctico) en el espacio PCA V4:

```text
Cluster  |  PC1     |  PC2     |  PC3     |  PC4
-------------------------------------------------
   0     |  +2.31   |  +0.84   |  -0.12   |  +0.67
   1     |  -1.97   |  -0.71   |  +1.42   |  -0.33
   2     |  -2.45   |  +1.63   |  +0.08   |  -2.11
   3     |  +1.88   |  -1.52   |  -0.76   |  +1.89
```

---

## 7. Cluster Profile Analysis — Perfiles de Dominio F1

### Clúster 0: "High Speed & DRS Efficiency"

- **Firma Digital:** Alto PC1 positivo (+2.31), PC2 positivo (+0.84)
- **Física:** Velocidades punta máximas y acelerador al 100% durante gran parte de la vuelta.
- **Interpretación:** Vueltas de ataque puro o intentos de adelantamiento con DRS activo. Neumático en fase media del stint.

### Clúster 1: "Standard Racing Pace"

- **Firma Digital:** PC1 negativo moderado, PC3 positivo alto (+1.42)
- **Física:** Gestión consistente de neumáticos y combustible. Ritmo sostenido.
- **Interpretación:** El estado predominante durante el ~60-70% de una carrera normal. Conducción conservadora para mantener el compuesto.

### Clúster 2: "Mechanical Grip & Braking"

- **Firma Digital:** PC1 muy negativo (-2.45), PC4 muy negativo (-2.11, tyre_age bajo)
- **Física:** Neumático nuevo o casi nuevo, frenadas pronunciadas, sectores técnicos.
- **Interpretación:** Vueltas de inicio de stint con compuesto fresco. Sectores donde el grip mecánico es el factor limitante.

### Clúster 3: "Tactical Outliers / Late Stint"

- **Firma Digital:** PC1 positivo (+1.88), PC2 negativo (-1.52), PC4 muy positivo (+1.89, tyre_age alto)
- **Física:** Velocidades anómalamente variables, tiempos de vuelta extendidos por degradación, gestión agresiva.
- **Interpretación:** Vueltas en la fase final del stint con degradación acumulada. También captura Safety Cars y eventos especiales de baja velocidad.

---

## 8. Relación de Variables y Cohesión Geométrica

### Impacto de Variables Físicas

El análisis gráfico en el espacio PCA muestra correlaciones directas entre los clústeres y la dinámica del coche:

| Relación | Observación |
|:---|:---|
| **Velocidad vs Acelerador** | Los clústeres 0 y 1 separan claramente el régimen "Full Push" del ritmo de gestión |
| **tyre_age vs lap_duration** | El Clúster 2 (neumático nuevo) migra progresivamente hacia el Clúster 3 a medida que aumenta la edad del compuesto |
| **PC1 (trazado) vs PC2 (agresividad)** | La combinación de ambas dimensiones separa correctamente los estados extremos |

### Análisis de Fallos (Failure Analysis)

Un **~3.5% de las muestras** presentaron Silhouette negativo (puntos "fronterizos"):

- **Causa principal:** Vueltas de transición táctica donde la telemetría cambia drásticamente en un sector (e.g., bandera amarilla repentina, cambio de estrategia)
- **Distribución:** Concentradas principalmente en los límites entre Clúster 1 y Clúster 3
- **Impacto:** Mínimo sobre la calidad general del modelo dado el bajo porcentaje

> [!WARNING]
> **Límite de interpretación:** Al contrario de DBSCAN, K-Means **obliga a asignar cada vuelta** a un clúster. Las vueltas de Safety Car y outliers extremos son asignadas al clúster más cercano (típicamente Clúster 3), lo que puede inflar artificialmente ese grupo. DBSCAN resuelve esta limitación al clasificar estos puntos como ruido (-1).

---

## 9. Comparativa con Otros Métodos

| Método | k detectado | Silhouette | Ruido | Ventaja |
|:---|:---|:---|:---|:---|
| **K-Means V2** | 4 (forzado) | 0.4409 | 0% (todos asignados) | Determinista, interpretable |
| **Hierarchical V4** | 5 (corte dendrograma) | 0.5142 | 0% (todos asignados) | Jerarquía relacional |
| **DBSCAN V3** | 5 (emergente) | 0.5910 | 11.2% (ruido real) | Captura topología irregular |

> [!NOTE]
> La **convergencia de los tres métodos en torno a 4-5 clústeres** es evidencia estadística fuerte de que la taxonomía de regímenes detectada refleja estructura real en los datos de telemetría F1. Las diferencias en el número exacto de clústeres se explican por las diferencias matemáticas entre métodos.

---

## 10. Conclusiones e Insights Estratégicos

> [!IMPORTANT]
> **🎯 Insights del Análisis K-Means V2 sobre PCA V4:**
>
> 1. **Física Pura:** El modelo captura la dinámica del coche sin sesgos de piloto o equipo, permitiendo una comparación objetiva del rendimiento entre vueltas.
>
> 2. **Predictor Táctico:** La etiqueta del clúster puede usarse como variable de entrada para predecir cuándo un piloto entrará a boxes basándose en la degradación relativa respecto al "Standard Pace" (Clúster 1).
>
> 3. **Validación del Espacio PCA:** El Silhouette Score de 0.44 demuestra que PCA V4 genera un espacio latente excelente para K-Means, eliminando el ruido y resaltando los patrones tácticos reales.
>
> 4. **Limitación Principal:** K-Means no puede detectar vueltas anómalas (Safety Cars, incidentes) como ruido — las fuerza dentro de un clúster existente. DBSCAN supera esta limitación a costa de mayor complejidad de parametrización.

### Próximos Pasos

| Acción | Justificación |
|:---|:---|
| **Exportar etiquetas K-Means** como nueva feature | Los labels de régimen de conducción son predictores candidatos para modelos de clasificación de estrategia. |
| **Comparar asignaciones** vuelta-a-vuelta con DBSCAN y Hierarchical | Identificar el "núcleo duro" de cada arquetipo donde los 3 métodos coinciden. |
| **Incorporar `kmeans_cluster`** en Feature Engineering V6 | Feature categórica de alta predictividad para modelos de predicción de stint. |
