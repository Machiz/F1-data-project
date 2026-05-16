# Reporte Técnico Avanzado: K-Means V2 - Telemetría PCA (V4)

Este documento detalla la implementación y los hallazgos de la versión optimizada de **K-Means Clustering** sobre la telemetría por vuelta. Se han incorporado métricas de validación avanzadas, visualización de centroides y un análisis de impacto de variables físicas.

## 1. Validación Técnica y Selección de K

Se realizó un barrido exhaustivo de $k$ (número de clústeres) evaluando la **Inercia** y el **Silhouette Score** para garantizar una selección fundamentada y no arbitraria.

### Metodología de Selección
*   **Inercia (Codo):** Se observó una estabilización clara en $k=4$, donde la ganancia en cohesión interna disminuye su ritmo de mejora.
*   **Silhouette Score:** El pico de separación se encuentra en $k=4$ (~0.44), lo que indica que esta estructura maximiza la distancia inter-cluster mientras mantiene la cohesión intra-cluster.
*   **Justificación de Dominio:** Los 4 clusters se alinean con los estados reales de un monoplaza: *Qualy Mode*, *Racing Pace*, *Tyre Management* y *Anomalías (Incidentes)*.

---

## 2. Perfilado de Clústeres (Cluster-Profile Analysis)

A través del análisis de centroides en el espacio PCA y su correlación con la telemetría original, identificamos:

### Cluster 0: "High Speed & DRS Efficiency"
*   **Firma Digital:** Alto PC1 positivo.
*   **Física:** Velocidades punta máximas y acelerador al 100% durante gran parte de la vuelta.
*   **Uso:** Vueltas de ataque puro o intentos de adelantamiento con DRS.

### Cluster 1: "Mechanical Grip & Braking"
*   **Firma Digital:** PC2 negativo pronunciado.
*   **Física:** Sectores con frenadas bruscas y paso por curvas lentas donde el grip mecánico es el factor limitante.
*   **Uso:** Sectores técnicos o circuitos urbanos.

### Cluster 2: "Standard Racing Pace"
*   **Firma Digital:** Perfil balanceado en el origen del espacio latente.
*   **Física:** Gestión consistente de neumáticos y combustible.
*   **Uso:** El estado predominante durante el 70% de una carrera normal.

### Cluster 3: "Tactical Outliers / Safety Car"
*   **Firma Digital:** Desviación significativa en PC3/PC4.
*   **Física:** Velocidades anómalamente bajas y tiempos de vuelta extendidos.
*   **Uso:** Identificación automática de vueltas bajo Safety Car, errores de pilotaje o paradas en boxes.

---

## 3. Relación de Variables y Cohesión

### Impacto de Variables Físicas
El análisis gráfico demostró una correlación directa entre los clusters y la dinámica del coche:
*   **Velocidad vs Acelerador:** Los clusters separan claramente el régimen de "Full Push" (Cluster 0) de los regímenes de gestión (Cluster 2).
*   **Degradación:** Se observa cómo el Cluster 2 se desplaza hacia el Cluster 3 a medida que `tyre_age` aumenta, capturando la pérdida de rendimiento por desgaste.

### Análisis de Fallos (Failure Analysis)
Solo un **~3.5%** de las muestras presentaron una silueta negativa. Estas "Vueltas Fronterizas" corresponden principalmente a periodos de transición donde la telemetría cambia drásticamente en un solo sector (v.g. una bandera amarilla repentina).

---

## 4. Conclusiones e Insights Estratégicos
1. **Física Pura:** El modelo captura la dinámica del coche sin sesgos de piloto o equipo, permitiendo una comparación objetiva del rendimiento.
2. **Predictor Táctico:** La etiqueta del cluster puede usarse como variable de entrada para predecir cuándo un piloto entrará a boxes basándose en su degradación relativa al "Standard Pace".
3. **Optimización de Telemetría:** El PCA V4 ha demostrado ser un espacio latente excelente para el clustering, eliminando el ruido y resaltando los patrones tácticos reales.

---
*Análisis basado en K_Means_Clustering_V2_Telemetry_PCA.ipynb*
