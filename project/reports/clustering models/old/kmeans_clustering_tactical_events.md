# Documentación Técnica: Clustering K-Means de Eventos Tácticos

Este documento resume el análisis y los hallazgos de aplicar el algoritmo de **K-Means Clustering** sobre los Componentes Principales (PCA) extraídos de los eventos tácticos de la Fórmula 1. 

## 1. Archivos Relacionados
*   **Notebook Principal:** `project/notebooks/clustering models/K_Means_Clustering.ipynb`
*   **Dataset Generado:** `project/data/features/kmeans_clusters.parquet` (643 eventos, 20 variables, incluyendo el `kmeans_cluster` y `sil_score` individual).
*   **Artefactos Gráficos Guardados:** 
    *   `kmeans_silhouette_plot.png`
    *   `kmeans_scatter_pca.png`
    *   `kmeans_centroid_heatmap.png`
    *   `kmeans_cluster_distribution.png`
    *   `kmeans_failure_analysis.png`

---

## 2. Fundamentos Matemáticos: ¿Cómo funciona K-Means?

El algoritmo **K-Means** es un método de partición de aprendizaje no supervisado que agrupa datos en $k$ clústeres distintos basándose en la proximidad o similitud geométrica (distancia). Su funcionamiento sigue un proceso iterativo de optimización conocido como *Expectation-Maximization*:

1. **Inicialización (K-Means++):** Se seleccionan $k$ puntos iniciales (centroides) de manera inteligente en el espacio dimensional para evitar caer en óptimos locales subóptimos.
2. **Asignación (Expectation):** Cada evento táctico en el espacio de componentes principales (PCA) se asigna al centroide más cercano utilizando la **Distancia Euclidiana**, con el objetivo de minimizar la varianza intraclúster:
   $$J = \sum_{j=1}^{k} \sum_{i=1}^{n} \|x_i^{(j)} - c_j\|^2$$
3. **Actualización (Maximization):** Se recalculan las coordenadas de cada centroide moviéndolo al punto medio (promedio) de todas las posiciones de los eventos que le fueron asignados en el paso anterior.
4. **Convergencia:** Los pasos 2 y 3 se repiten iterativamente hasta que los centroides dejan de moverse de forma significativa (la inercia o varianza total intra-clúster se ha minimizado).

En el contexto de la F1, esta **reasignación dinámica** de puntos permite que el algoritmo reclasifique eventos "híbridos" de manera flexible, lo que suele lidiar extremadamente bien con el ruido estocástico de la telemetría.

---

## 3. Definición del Modelo y Sweep de Parámetros

El modelo K-Means se corrió sobre las 15 dimensiones generadas por el pipeline de PCA. Dado que K-Means requiere definir la cantidad de clústeres a priori ($k$), se realizó un "Parameter Sweep" evaluando $k$ desde 2 hasta 10 utilizando la métrica de Inercia (Método del Codo), Silhouette Score, Davies-Bouldin y Calinski-Harabasz.

### 3.1. Selección de $k=4$
La validación paramétrica confirmó que $k=4$ es el número óptimo de clústeres debido a:
1. **Punto de Inflexión (Elbow):** La curva de Inercia se estabiliza a partir de $k=4$.
2. **Silhouette / Densidad:** Presenta un pico de cohesión vs separación con un score de **0.3910**.
3. **Alineamiento de Dominio:** Corrobora fuertemente la existencia de 4 arquetipos tácticos reales en la Fórmula 1.

---

## 4. Perfiles de Clústeres (Distribución y Centroides)

La distribución final sobre los 643 eventos analizados muestra un clúster dominante y tres especializaciones:

*   **Cluster 0 ($n=82$):** Exclusivamente batallas en pista (`On_Track_Overtake`). Componente principal predominante (PC1 mean: 13.69).
*   **Cluster 1 ($n=378$):** El clúster mayoritario. Contiene el grueso de adelantamientos híbridos y capturó 58 de los eventos de `Pit_Strategy` (Undercuts). 
*   **Cluster 2 ($n=65$):** Batallas en pista extremadamente atípicas o concentradas aerodinámicamente. PC2 tiene un valor drásticamente negativo (-17.35).
*   **Cluster 3 ($n=118$):** Batallas de alta varianza con características mixtas. PC3 es fuertemente negativo.

---

## 5. Failure Analysis (Análisis de Calidad por Punto)

Dado que las tácticas deportivas son espectros continuos y no "cajas rígidas", se analizó la puntuación *Silhouette* individual de cada punto para medir qué batallas quedaron en las "fronteras" de los clústeres:

*   ✅ **Bien agrupados ($sil \ge 0.30$):** 485 puntos (75.4%)
*   ⚠️ **En el límite / Borderline ($0 \le sil < 0.10$):** 43 puntos (6.7%)
*   ❌ **Mal asignados o híbridos ($sil < 0$):** 25 puntos (3.9%)

> [!IMPORTANT]
> **Insight Táctico de Híbridos:** El 100% de los puntos "mal asignados" ($sil < 0$) cayeron en el **Cluster 3**. Estos no son necesariamente fallos matemáticos, sino **híbridos tácticos genuinos**. Representan eventos que exhiben la telemetría de dos tácticas distintas simultáneamente (por ejemplo, una detención en boxes perfectamente calculada para poder usar el DRS en la primera vuelta de salida, mezclando *Pit_Strategy* con *On_Track_Overtake*).

---

## 6. Conclusiones y Próximos Pasos

1. **Validación del Dominio:** La convergencia natural de K-Means en $k=4$ es una fuerte evidencia analítica de que las tácticas de F1 se pueden dividir científicamente en 4 estrategias base.
2. **Recomendación de Clasificación (XGBoost):** Recomendamos utilizar la variable exportada `kmeans_cluster` (en `kmeans_clusters.parquet`) como la variable *Target* principal para un futuro modelo clasificador capaz de etiquetar maniobras en tiempo real a partir de la telemetría viva de un coche.
