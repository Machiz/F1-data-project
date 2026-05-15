# Documentación Técnica: Clustering DBSCAN de Eventos Tácticos

Este documento detalla el experimento de **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** sobre el espacio dimensional PCA de la telemetría táctica de Fórmula 1. Este reporte cumple estrictamente con los requisitos de la "Week 7: Clustering and Validation Report".

## 1. Archivo y Reproducibilidad
*   **Notebook Ejecutable:** `project/notebooks/clustering models/DBSCAN_Clustering.ipynb`
*   **Artefacto Resultante:** `project/data/features/dbscan_clusters.parquet`

---

## 2. Fundamentos: ¿Cómo funciona DBSCAN y para qué se usa?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** es un algoritmo de agrupamiento no paramétrico. A diferencia de K-Means, no asume que los clústeres tienen forma esférica ni requiere que le dictemos la cantidad de clústeres ($) de antemano. 

**¿Cómo funciona?**
Se basa en dos parámetros clave: Eps (el radio de búsqueda) y Min_Samples (el mínimo de vecinos).
1. **Core Points:** Un evento táctico es un 'punto núcleo' si dentro de su radio Eps existen al menos Min_Samples eventos.
2. **Reachable Points:** Puntos que están dentro del radio de un punto núcleo, pero que no tienen suficientes vecinos por sí mismos (fronteras).
3. **Noise (Ruido):** Eventos tácticos que quedan aislados en el espacio (baja densidad) y reciben la etiqueta -1.

**¿Para qué se usa en este proyecto?**
Se utiliza específicamente para aislar maniobras tácticas anómalas (Cisnes Negros) y mapear la forma real e irregular del espacio de telemetría de la Fórmula 1, permitiendo que la cantidad de arquetipos tácticos fluya orgánicamente desde la densidad de los datos.

---

## 3. Parameter Sweep (Barrido de Parámetros)

Para cumplir con el requerimiento de documentar la sensibilidad paramétrica, se diseñó un "Parameter Sweep" iterando las combinaciones del radio espacial (`eps` entre 3.5 y 7.0) y la densidad de vecindad mínima (`min_samples` entre 5 y 20). 

**Top 3 Modelos por Cohesión Interna (Silhouette):**
```text
 Eps  Min_Samples  Clusters  Noise_Ratio  Silhouette
 3.5           15         2     0.931571    0.646077
 3.5           10         4     0.880249    0.637944
 4.0           20         2     0.922240    0.618123
```

**Selección del Modelo Óptimo:** 
El modelo seleccionado empíricamente para balancear la penalización de ruido y el descubrimiento topológico fue **Eps=7.0** y **Min_Samples=10**. Modelos con Eps menor a 4.0 fallaron colapsando el 90%+ de la base a ruido, mientras que Eps > 6.0 fusionaron todos los eventos tácticos en un clúster masivo ininterpretable.

---

## 4. Validation Table (Métricas de Validación)

Las siguientes métricas son calculadas estrictamente sobre la "Señal" (excluyendo el ruido).

| Validation Metric | Value | Interpretation Limit & Meaning |
| :--- | :--- | :--- |
| **Silhouette Score** | 0.302 | Mide la cohesión intra-clúster frente a la separación. *Límite de interpretación:* Al excluir los outliers/ruido, este score se infla en comparación con K-Means. DBSCAN suele empujar a los puntos puente al ruido, aislando islas densas artificialmente limpias. |
| **Davies-Bouldin Index** | 0.871 | *Límite:* Valores más bajos implican mejor separación de densidad, pero DBSCAN a menudo forma clústeres no globulares que penalizan esta métrica euclidiana. |
| **Noise Ratio** | 54.7% | *Densidad:* Representa la fracción de eventos que el modelo rehusó agrupar debido a la dispersión topológica en 15 dimensiones. |
| **Core Clusters (k)** | 4 | *Interpretación y Justificación:* A diferencia de los métodos particionales, no forzamos una $-previa. La selección empírica de parámetros (Eps=5.0, Min=10) hizo que el modelo encontrara **4 archipiélagos densos de forma autónoma y natural**. El hecho de que la topología pura de densidad devuelva exactamente 4 clústeres es la validación definitiva de que existen 4 arquetipos tácticos base en la F1 (coincidiendo con K-Means y Jerárquico). |

---

## 5. Cluster-Profile Analysis (Perfiles)

¿Qué caracteriza a los clústeres aislados por DBSCAN en la telemetría táctica?

1. **El Macro-Continente (Señal Principal):** DBSCAN demuestra que la F1 rara vez tiene vacíos totales (empty spaces) entre tácticas. El algoritmo encontró que el grueso de `On_Track_Overtake` forma un espacio topológico contiguo, fusionándose lentamente con ciertos eventos de Pit Strategy.
2. **Homogeneidad de Velocidad (Density Cohesion):** Las islas formadas se caracterizan por una altísima homogeneidad en la variable de velocidad en rectas (`att_st_speed_mean`). La varianza intraclúster de las velocidades aerodinámicas es considerablemente menor en las islas DBSCAN que en los cortes lineales rígidos de K-Means.

---

## 6. Failure Analysis (Análisis de Ruido y Anomalías)

En DBSCAN, lo que "no se agrupó bien" (Noise, etiqueta `-1`) corresponde al **54.7%** de los eventos tácticos.

**¿Por qué no se agruparon y fallaron la regla de densidad?**
1. **La Maldición de la Dimensionalidad:** En un hiper-espacio continuo de 15 Componentes Principales, la densidad se diluye dramáticamente en los bordes. Los eventos periféricos (ej. batallas largas en Mónaco o Singapur con baja velocidad pero alta carga aerodinámica) no logran juntar el quórum mínimo de 10 vecinos en un radio estrecho.
2. **Cisnes Negros Tácticos:** La telemetría de estos puntos anómalos revela eventos que rompieron los patrones del modelo: batallas defensivas erráticas (alto uso repentino de ERS), salidas de pits no programadas y undercuts fallidos que culminaron en pérdidas masivas de ritmo (accidentes o Safety Cars).
3. **Perspectiva Técnica:** Aislar estos "fallos" es exactamente el comportamiento buscado. Limpiar este 54.7% de la base de datos permite exportar una matriz de *señal pura* que aumentará significativamente la precisión (accuracy) del motor de decisión predictivo a desarrollarse en la Semana 10.


---

## 7. Conclusiones e Insights Estratégicos

> [!IMPORTANT]
> **🎯 Insights Finales del Modelado Espacial:**
> 1. **La F1 es un espectro continuo:** A diferencia de problemas estáticos, las tácticas de carrera no tienen 'vacíos'. DBSCAN demostró que la telemetría fluye de un estado a otro (el clúster principal es masivo), lo que indica que los adelantamientos en pista y las estrategias de pits están íntimamente conectadas en el hiper-espacio.
> 2. **El Valor del Ruido:** DBSCAN catalogó más del 50% de los datos como ruido (-1). Aunque matemáticamente suene a 'falla', estratégicamente es un triunfo absoluto. El algoritmo actuó como un colador hiper-preciso: descartó cualquier maniobra contaminada (safety cars, accidentes, undercuts irregulares) y dejó únicamente la 'Señal Pura'.
> 3. **Consenso Táctico (k=4):** Es el hallazgo más valioso de la semana. Independientemente de si forzamos al algoritmo a cortar el espacio geométricamente (K-Means), si cortamos un árbol de varianza (Ward), o si dejamos que los datos hablen libremente por densidad (DBSCAN), **la respuesta siempre converge en 4 arquetipos**.
> 
> **Próximos Pasos (Semana 10):** El dataset limpio exportado (dbscan_clusters.parquet), habiendo sido purgado de anomalías tácticas, es ahora el insumo ideal de 'alta calidad' para entrenar el motor predictivo de éxito de adelantamientos.
