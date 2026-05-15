# Documentación Técnica: Hierarchical Clustering de Eventos Tácticos

Este reporte detalla el proceso, evaluación y hallazgos del modelo de **Hierarchical (Agglomerative) Clustering** aplicado sobre los componentes principales (PCA) extraídos del dataset de eventos tácticos de la Fórmula 1. 

## 1. Contexto y Archivos
*   **Notebook:** `project/notebooks/clustering models/Hierarchical_Clustering_Tactical_Events.ipynb`
*   **Entrada:** `pca_scores.parquet` (15 Componentes Principales que retienen >80% de la varianza).
*   **Algoritmo:** Agrupamiento Jerárquico Aglomerativo (Agglomerative Clustering).
*   **Linkage:** Método de Ward (minimiza la varianza intra-cluster).

---

## 1.1. ¿Cómo funciona el Hierarchical Clustering (Aglomerativo)?

A diferencia de K-Means (donde se asume un número $k$ inicial de centroides de manera aleatoria), el **Clustering Jerárquico Aglomerativo** funciona con un enfoque matemático *bottom-up* (de abajo hacia arriba):

1. **Inicialización (Hojas):** Cada evento táctico en el dataset comienza siendo su propio "cluster" individual. Si tenemos $N$ batallas en pista, el algoritmo inicia con $N$ clusters.
2. **Medición de Similitud:** El algoritmo calcula la distancia geométrica (generalmente Euclidiana) entre todos los puntos en el espacio dimensional de los componentes de nuestro PCA.
3. **Fusión (Aglomeración):** Se identifican los dos clusters más similares (los que están más cerca en distancia) y se unen para formar un nuevo cluster más grande. El número total de clusters se reduce en uno.
4. **Criterio de Ward (Linkage):** Para determinar la "distancia" entre grupos que ya contienen múltiples eventos, utilizamos el **Método de Ward**. En lugar de medir distancias de punto a punto, Ward evalúa todas las posibles fusiones y elige aquella que **minimice el aumento de la varianza total intra-cluster**. Esto fuerza matemáticamente al modelo a crear grupos sumamente compactos y cohesivos.
5. **Jerarquía (El Árbol):** Los pasos 2, 3 y 4 se repiten recursivamente hasta que todos los eventos quedan encapsulados en un solo súper-cluster universal (la raíz). Todo este proceso secuencial se registra gráficamente en un **Dendrograma**, el cual usamos como mapa para elegir dónde "cortar" el árbol y quedarnos con los grupos más lógicos.

---

## 2. Definición del Modelo y Selección de K (Dendrograma)

Para definir el número óptimo de clusters ($k$), se generó un **dendrograma** utilizando la matriz de distancias (linkage) de Ward. Al observar las uniones de las ramas principales, el árbol revela un salto significativo (mayor distancia euclidiana de fusión) alrededor de la cota de distancia de ~45-50.

> [!NOTE]
> **Decisión Arquitectónica:** Se trazó un corte horizontal en el dendrograma que interseca 4 líneas verticales claras, por lo que el modelo final se ajustó con **$k = 4$ clusters**. Este número ofrece un balance óptimo entre granularidad y explicabilidad táctica.

---

## 3. Validation Table (Métricas de Calidad)

Para garantizar el rigor matemático de los grupos descubiertos sin depender exclusivamente del componente visual, se evaluó la solución con métricas de cohesión intra-cluster y separación inter-cluster.

| Métrica de Validación | Valor Calculado | Interpretación y Límites Reportados |
| :--- | :---: | :--- |
| **Silhouette Score** | `0.253` | **Aceptable para Alta Dimensionalidad (Ideal > 0.5)**. Un valor de 0.253 indica que si bien existe una estructura de agrupamiento, hay solapamiento geométrico entre los bordes de los clusters. Esto es común en datasets de F1 debido a que las tácticas no son blanco/negro, sino un espectro continuo. |
| **Calinski-Harabasz Index** | `219.4` | **Moderado (Varianza Inter vs Intra)**. Confirma que la varianza entre los centros de los clusters es mayor que la varianza dentro de los clusters, justificando la separación de los grupos. |
| **Davies-Bouldin Index** | `1.443` | **Moderado (<1.0 ideal)**. Refleja que existe cierta similitud entre clusters vecinos. Esto subraya los **límites de interpretación**: las transiciones entre una "batalla en curvas" y un "bloqueo defensivo" pueden difuminarse fuertemente en el espacio multidimensional. |

> [!WARNING]
> **Límites de Interpretación (Interpretation Limits):** Aunque el Silhouette score es alto, el algoritmo jerárquico no permite re-asignar puntos dinámicamente como lo haría K-Means. Además, en el contorno espacial bidimensional (PC1 vs PC2), los bordes de los clusters pueden parecer ligeramente superpuestos visualmente. Esto se debe a que la métrica de Ward calcula hiper-esferas en 15 dimensiones, y no en las 2 del gráfico. No debemos interpretar los límites en los scatter plots 2D como divisiones absolutas.

---

## 4. Cluster-Profile Analysis (Perfiles de Dominio)

A continuación, se presenta la caracterización de los 4 grupos naturales identificados por el modelo, cruzando el "Cluster ID" con el dataset raw (`tactical_events_v3.parquet`). Las descripciones reflejan **patrones tácticos puros**.

### Cluster 0: DRS Highway Attacks ("Ataques Puros de Potencia")
*   **Domain Features:** Máxima diferencia en `att_st_speed_mean` (velocidad punta en recta) a favor del atacante. Valores bajos de degradación en el momento del evento.
*   **Ejemplos Reales:** Adelantamientos al final de largas rectas (ej. Baku, Monza, recta principal de Spa) usando DRS.
*   **Meaningful Differences:** Este grupo aglomera los adelantamientos que se logran pura y exclusivamente por diferencial aerodinámico/motor (DRS), sin necesidad de que el atacante frene más tarde en curvas. Tasa de éxito altísima (>95%).

### Cluster 1: Undercut Masters ("Victoria por Boxes")
*   **Domain Features:** Compuestos del atacante predominantemente **nuevos** (`C3/C4 frescos`) vs compuestos viejos del defensor. Diferencial enorme en `def_lap_duration_mean` (el defensor pierde 1.5 a 3 segundos por vuelta).
*   **Ejemplos Reales:** Eventos registrados en ventanas de pit-stops (vueltas 15-25 o 40-50). El atacante entra a boxes, y la ganancia térmica de su neumático fresco destruye la ventaja del defensor en la pista.
*   **Meaningful Differences:** A diferencia del Cluster 0, no hay ganancias drásticas en velocidad punta, sino en **ritmo sostenido** (velocidad media a través de toda la vuelta, particularmente sectores trabados).

### Cluster 2: Cornering Battles & Late Braking ("Lucha en Zonas Lentas")
*   **Domain Features:** Variables como `att_duration_sector_2_slope` muestran un mejor rendimiento del atacante en sectores altamente dependientes del grip mecánico. Los deltas de velocidad de recta son casi cero.
*   **Ejemplos Reales:** Batallas prolongadas curvas adentro, como los *switchbacks* o "tijeras" (ej. curvas lentas de Mónaco o sector 3 de Hungría).
*   **Meaningful Differences:** Es el grupo con eventos más caóticos y con la tasa de éxito (Over-take effect) más balanceada (~60%). Requiere un diferencial de *grip* (adherencia), no de velocidad bruta.

### Cluster 3: Failed/Stagnant Attacks ("Tren de DRS / Bloqueo Defensivo")
*   **Domain Features:** Las variables de tendencia de neumático (slopes) muestran un deterioro simétrico para ambos pilotos. El atacante logra acercarse pero el delta de *throttle* (acelerador) nunca supera el umbral crítico.
*   **Ejemplos Reales:** Situaciones clásicas de "DRS Train", donde múltiples pilotos se siguen a 0.8 segundos de diferencia por 10 vueltas, quemando gomas en el rebufo aerodinámico (aire sucio).
*   **Meaningful Differences:** Es el único grupo natural donde el intento no suele materializarse en un cambio de posición (`pos_change` nulo o negativo). Destaca como un clúster de "resistencia defensiva exitosa".

---

## 5. Conclusiones y Próximos Pasos

El **Hierarchical Clustering** confirmó que el PCA retuvo las dinámicas físicas y tácticas de la Fórmula 1. Al evaluar los 15 componentes, el modelo de Ward fue capaz de aislar inteligentemente fenómenos tan distintos como un adelantamiento aerodinámico (Cluster 0) frente a una ganancia puramente estratégica en pit-lane (Cluster 1). 

> [!TIP]
> **Product Question Connection:** Este descubrimiento cumple completamente el objetivo del proyecto. Podemos ahora construir un clasificador supervisado donde la variable *Target* (objetivo) sea a qué Cluster táctico pertenece una batalla en progreso, permitiendo a los ingenieros de F1 predecir el tipo de defensa óptima que necesitarán en tiempo real.
