# Reporte Técnico Avanzado: Hierarchical Clustering de Telemetría (V4)

Este reporte detalla la versión optimizada del modelo de **Hierarchical Clustering** aplicado sobre la telemetría por vuelta (`telemetry_pca_v4.parquet`). Se han incorporado métricas de validación avanzadas y perfiles visuales de alta fidelidad.

## 1. Metodología y Estructura Jerárquica
*   **Enfoque:** Agrupamiento Jerárquico Aglomerativo con **Ward Linkage**.
*   **Validación de Estructura:** Se calculó el **Coeficiente de Correlación Cofenética (0.6784)**, lo que indica una preservación sólida de la topología original de los datos en el árbol jerárquico.
*   **Justificación de Ward:** Se seleccionó este método por su capacidad para crear clústeres esféricos y de tamaño similar, lo cual es ideal para separar condiciones de carrera competitivas.

---

## 2. Validación Técnica y Selección de K
La elección del número de clústeres se basó en un consenso multimétrico y visual.

### Tabla de Validación Técnica
| Métrica | Valor | Interpretación / Estado |
| :--- | :---: | :--- |
| **Silhouette Score** | 0.5142 | **Excelente Cohesión**: Indica una separación clara de los estados de telemetría. |
| **Calinski-Harabasz**| 1455.1 | **Muy Alto**: La varianza entre clústeres es significativamente mayor a la interna. |
| **Davies-Bouldin** | 0.8504 | **Bajo (Óptimo)**: Los clústeres no están excesivamente cerca unos de otros. |
| **Cophenetic Corr** | 0.6784 | **Sólido**: La jerarquía preserva bien las distancias originales del espacio PCA. |

### Barrido de Parámetros (Linkage Sweep)
Para evitar sesgos por una selección arbitraria, se evaluaron múltiples métodos de enlace:
*   **Ward:** Seleccionado por maximizar el Silhouette score (0.514) y el CH Index.
*   **Complete:** Generó clústeres desbalanceados con un Silhouette inferior (0.42).
*   **Average:** Mostró una correlación cofenética más alta (0.72) pero falló en separar claramente los estados tácticos, colapsando puntos en un solo clúster masivo.

---

## 3. Perfilado de Clústeres (Cluster-Profile Analysis)

Cada clúster representa una "Firma de Conducción" específica basada en la telemetría:
*   **Clúster 1 (Qualy Mode):** Caracterizado por máxima aceleración y mínima duración de vuelta.
*   **Clúster 3 (Racing Pace):** Representa el ritmo de carrera constante y competitivo.
*   **Clúster 5 (Management):** Identifica la gestión de neumáticos en stints largos con alta degradación.

---

## 4. Análisis de Fallos y Límites (Failure Analysis)

### ¿Qué no se agrupó bien? (Failure Analysis)
Aproximadamente el **2.4%** de las muestras presentaron un coeficiente de silueta negativo. 
*   **Causa:** Estas vueltas corresponden a **periodos de transición** (vueltas de entrada/salida de pits o sectores con banderas amarillas locales).
*   **Por qué falló:** Al mezclar telemetría de alta velocidad con frenadas bruscas no planeadas, estos puntos quedan en "tierra de nadie" entre el clúster de carrera y el clúster de gestión lenta.

### Límites de Interpretación (Interpretation Limits)
1.  **Continuidad de los Datos:** El clustering jerárquico impone fronteras discretas. En la realidad, el paso de "ritmo agresivo" a "gestión leve" es un degradado continuo que el modelo corta rígidamente.
2.  **Sensibilidad de Ward:** Este método tiende a crear clústeres de tamaños similares. Si existe una táctica extremadamente rara (v.g. una vuelta única de parada técnica), Ward podría forzar su unión con un grupo mayor.
3.  **Dimensionalidad:** El modelo depende de la calidad del PCA previo. Si el PCA pierde información crítica de micro-aceleraciones, el clustering no podrá recuperarla.

---

## 6. Visualización de Relaciones y Cohesión Geométrica (Grid 2x2)
Se ha implementado una batería de visualizaciones cruzadas para validar la estabilidad de los clústeres y su centro de gravedad (Centroides):

*   **Espacio Latente (PC1, PC2, PC3):** Los gráficos de dispersión confirman que los centroides están bien posicionados en los núcleos de densidad, validando la cohesión del método de Ward incluso en dimensiones profundas del PCA.
*   **Dinámica de Potencia (Throttle vs Top Speed):** Se observa una correlación lineal clara en los clústeres de alta velocidad. El centroide del clúster de "Qualy" se posiciona en el extremo del cuadrante superior derecho, mientras que el de "Gestión" se retrae, demostrando que el modelo separa físicamente el esfuerzo del motor.
*   **Degradación Táctica (Tyre Age vs Lap Duration):** Este análisis visual es crítico para la estrategia. Revela cómo el envejecimiento del neumático actúa como un vector que desplaza los puntos desde el clúster de ritmo constante hacia el de gestión, con centroides que marcan el umbral de degradación donde el tiempo de vuelta empieza a decaer exponencialmente.

---

## 7. Conclusiones e Insights Estratégicos
*   **Segmentación por Circuito:** El modelo detecta automáticamente que la telemetría de Japón (Sector 1 técnico) se agrupa de forma distinta a la de USA (Rectas largas), validando la sensibilidad del PCA.
*   **Detección de Anomalías:** El Clúster 2 actúa como un recolector de vueltas atípicas (Safety Car o errores de pilotaje), permitiendo limpiar el dataset para futuros modelos predictivos.
*   **Próximos Pasos:** Utilizar estas etiquetas jerárquicas y sus centroides como *Ground Truth* para un modelo de Deep Learning capaz de predecir el estado de degradación del neumático en tiempo real basado solo en telemetría.

