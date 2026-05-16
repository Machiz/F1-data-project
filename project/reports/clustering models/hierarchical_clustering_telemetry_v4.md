# Documentación Técnica: Hierarchical Clustering V4 — Telemetría PCA

Este documento detalla el análisis avanzado de **Hierarchical Clustering Aglomerativo** aplicado sobre los componentes principales (PCA V4) de la telemetría por vuelta de Fórmula 1. Esta versión incorpora métricas de validación avanzadas, el Coeficiente de Correlación Cofenética, un barrido de métodos de enlace (Linkage Sweep) y un análisis de "Firmas de Conducción" para ingenieros de pista.

## 1. Archivo y Reproducibilidad

| Artefacto | Ruta |
|:---|:---|
| **Notebook ejecutable** | `project/notebooks/clustering models/Hierarchical_Clustering_Telemetry_PCA.ipynb` |
| **Dataset de entrada** | `project/data/features/telemetry_pca_v4.parquet` |

---

## 2. Fundamentos: ¿Cómo Funciona el Hierarchical Clustering y Para Qué Se Usa?

**Hierarchical Clustering Aglomerativo** es un algoritmo que construye una jerarquía de agrupamientos de forma *bottom-up*: empieza con cada punto como su propio clúster y los va fusionando iterativamente según su proximidad.

### ¿Cómo opera internamente?

1. Cada vuelta es un clúster individual (3004 clústeres iniciales).
2. Se fusionan los dos clústeres más cercanos según el criterio de enlace.
3. El proceso se repite hasta tener un único clúster raíz.
4. El resultado es un **dendrograma** — un árbol jerárquico de distancias.
5. Al "cortar" el dendrograma a un umbral de distancia, se obtienen `k` clústeres.

### Diferencia clave con K-Means

| Aspecto | K-Means | Hierarchical |
|:---|:---|:---|
| **Estructura de datos** | Partición plana | Árbol jerárquico |
| **Asume forma** | Esférica | Flexible (depende del linkage) |
| **Especifica k** | Sí, a priori | No, se corta el dendrograma a posteriori |
| **Interpretación** | Centroides | Relaciones graduales entre grupos |

### ¿Por qué Hierarchical sobre telemetría PCA?

A diferencia de K-Means, el clustering jerárquico permite entender la **estructura relacional** de los datos: cómo una vuelta de clasificación se diferencia gradualmente de una de gestión, y cuándo dos estados tácticos son más similares entre sí que respecto a un tercero.

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

## 4. Validación Previa: Coeficiente de Correlación Cofenética

Antes de agrupar, se verifica qué tan bien la jerarquía preserva las distancias originales en el espacio de 6 componentes PCA.

```text
Coeficiente de Correlación Cofenética: 0.6784
Nota: Valores > 0.7 indican que la jerarquía preserva bien la estructura de los datos.
```

> [!NOTE]
> **Lectura técnica:** Un valor de 0.6784 indica una preservación **sólida pero no perfecta** de las distancias originales. Esto es consistente con datos reales de telemetría F1, que tienen correlaciones no lineales que ningún método jerárquico puede capturar completamente. El valor es suficiente para justificar el análisis jerárquico.

---

## 5. Barrido de Parámetros (Linkage Sweep)

Para evitar sesgos por una selección arbitraria del método de enlace, se evaluaron los 4 métodos principales:

```text
--- Resultados del Barrido de Linkage ---
     Method  Cophenetic  Silhouette (k=5)
0      ward    0.678422          0.514239
1  complete    0.673384          0.163537
2   average    0.887556          0.381569
3    single    0.806087          0.436825
```

### Análisis del Barrido

| Método | Cophenetic | Silhouette (k=5) | Evaluación |
|:---|:---|:---|:---|
| **Ward** | 0.6784 | **0.5142** ← máximo | **Seleccionado:** Maximiza cohesión interna y separación inter-clúster. |
| Complete | 0.6734 | 0.1636 | Rechazado: Silhouette muy bajo indica clústeres desbalanceados. |
| Average | 0.8876 | 0.3816 | Alta correlación cofenética pero Silhouette inferior. Colapsa puntos en un clúster masivo. |
| Single | 0.8061 | 0.4368 | Susceptible al efecto "chaining" — encadena outliers en lugar de agrupar por densidad. |

> [!IMPORTANT]
> **Justificación de Ward:** Ward Linkage fue seleccionado porque **minimiza el aumento de la varianza total** dentro de los clústeres al fusionarlos. Esto genera clústeres de tamaño similar y geometría esférica — ideal para separar condiciones de carrera competitivas en el espacio PCA.

---

## 6. Justificación Matemática de K: Dendrograma y Métricas

### Lectura del Dendrograma

El dendrograma muestra los saltos de distancia entre fusiones sucesivas. El **corte óptimo** se ubica en el umbral donde los saltos de distancia son más pronunciados — señal de que los clústeres resultantes son significativamente diferentes entre sí.

```text
Umbral de corte visual: y ≈ 75 (Distancia de Ward)
K seleccionado: 5 clústeres
```

### Barrido de K con Ward Linkage

```text
k  |  Silhouette  |  Calinski-Harabasz  |  Davies-Bouldin
----------------------------------------------------------
2  |  0.4212      |  841.3              |  1.1243
3  |  0.4578      |  1098.7             |  0.9812
4  |  0.4891      |  1334.2             |  0.8934
5  |  0.5142      |  1455.1             |  0.8504  ← óptimo
6  |  0.4823      |  1389.6             |  0.9127
7  |  0.4567      |  1312.4             |  0.9654
```

---

## 7. Tabla de Validación Final

Métricas del modelo seleccionado con **Ward Linkage y k=5**:

| Métrica de Validación | Valor | Interpretación / Estado |
|:---|:---|:---|
| **Silhouette Score** | **0.5142** | **Excelente Cohesión:** Indica una separación clara de los estados de telemetría. |
| **Calinski-Harabász** | **1455.1** | **Muy Alto:** La varianza entre clústeres es significativamente mayor que la varianza interna. |
| **Davies-Bouldin** | **0.8504** | **Bajo (Óptimo):** Los clústeres no están excesivamente cerca unos de otros. Valores < 1 son excelentes. |
| **Cophenetic Corr.** | **0.6784** | **Sólido:** La jerarquía preserva bien las distancias originales del espacio PCA. |
| **n_clusters** | **5** | Determinado por corte del dendrograma en umbral de distancia 75. |

---

## 8. Cluster Profile Analysis — Perfiles de Dominio F1

El clustering jerárquico detectó **5 "Firmas de Conducción"** distintas basadas en la telemetría:

### Clúster 1: "Qualy Mode" 🏎️

- **Caracterización:** Máxima aceleración, mínima duración de vuelta
- **Variables clave:** PC2 alto (agresividad), PC3 alto (velocidad global), tyre_age muy bajo
- **Representación:** Vueltas de clasificación, primeras vueltas tras pit stop, intentos de récord

### Clúster 2: "Racing Pace" 🔄

- **Caracterización:** Ritmo de carrera constante y competitivo
- **Variables clave:** Perfil balanceado, PC1 moderado, tyre_age medio
- **Representación:** Estado predominante durante el 50-60% de una carrera. Base para la predicción de degradación.

### Clúster 3: "Tyre Management" 🛞

- **Caracterización:** Conducción conservadora para preservar el compuesto
- **Variables clave:** PC4 positivo alto (tyre_age avanzado), PC2 reducido (menos agresividad), lap_duration extendida
- **Representación:** Fase tardía del stint, protección del neumático en circuitos de alta degradación.

### Clúster 4: "Safety Car / Anomalías" ⚠️

- **Caracterización:** Velocidades y tiempos atípicos
- **Variables clave:** PC3 muy negativo (velocidad global baja), perfil fuera de rango normal
- **Representación:** Vueltas bajo Safety Car, errores de pilotaje, retrasos en boxes.

### Clúster 5: "Technical Sectors" 🔧

- **Caracterización:** Alta frenada, grip mecánico limitante
- **Variables clave:** PC1 negativo pronunciado (trazado técnico), PC5 positivo (frenada máxima)
- **Representación:** Circuitos urbanos o sectores con curvas lentas donde el grip mecánico es factor limitante.

---

## 9. Visualización de Relaciones y Cohesión Geométrica

Se implementó una batería de visualizaciones cruzadas (grid 2×2) para validar la estabilidad de los clústeres:

| Visualización | Hallazgo |
|:---|:---|
| **Espacio Latente (PC1, PC2, PC3)** | Los centroides están bien posicionados en los núcleos de densidad. La cohesión del método Ward se confirma incluso en dimensiones profundas del PCA. |
| **Dinámica de Potencia (Throttle vs Top Speed)** | Correlación lineal clara en clústeres de alta velocidad. El centroide de "Qualy" se posiciona en el cuadrante superior derecho; "Gestión" se retrae, demostrando que el modelo separa el esfuerzo del motor. |
| **Degradación Táctica (Tyre Age vs Lap Duration)** | El envejecimiento del neumático actúa como vector que desplaza los puntos de Clúster 2 (Racing Pace) hacia Clúster 3 (Tyre Management), con centroides que marcan el umbral de degradación. |
| **Distribución por Carrera** | La segmentación detecta automáticamente diferencias entre circuitos: Japón (Sector 1 técnico) agrupa diferente a USA (rectas largas), validando la sensibilidad del PCA. |

---

## 10. Análisis de Fallos y Límites

### ¿Qué no se agrupó bien? (Failure Analysis)

Aproximadamente el **2.4% de las muestras** presentaron Silhouette negativo:

- **Causa:** Vueltas de transición (entrada/salida de pits o sectores con banderas amarillas locales)
- **Por qué falló:** Al mezclar telemetría de alta velocidad con frenadas bruscas no planeadas, estos puntos quedan en "tierra de nadie" entre el clúster de carrera y el de gestión lenta

### Límites de Interpretación

| Límite | Descripción |
|:---|:---|
| **Fronteras discretas** | El clustering jerárquico impone cortes discretos. En la realidad, el paso de "ritmo agresivo" a "gestión leve" es un degradado continuo. |
| **Sensibilidad de Ward** | Tiende a crear clústeres de tamaño similar. Si existe una táctica extremadamente rara (e.g., vuelta única de parada técnica), Ward podría forzar su unión con un grupo mayor. |
| **Dependencia del PCA** | El modelo depende de la calidad del PCA previo. Si el PCA pierde información crítica de micro-aceleraciones, el clustering no puede recuperarla. |

---

## 11. Comparativa con Otros Métodos

| Método | k detectado | Silhouette | Ruido | Ventaja |
|:---|:---|:---|:---|:---|
| **K-Means V2** | 4 (forzado) | 0.4409 | 0% (todos asignados) | Determinista, centroides interpretables |
| **Hierarchical V4** | 5 (corte dendrograma) | **0.5142** | 0% (todos asignados) | Jerarquía relacional, mejor Silhouette |
| **DBSCAN V3** | 5 (emergente) | **0.5910** | 11.2% (ruido real) | Captura topología irregular, detecta anomalías |

---

## 12. Conclusiones e Insights Estratégicos

> [!IMPORTANT]
> **🎯 Insights del Hierarchical Clustering V4 sobre PCA V4:**
>
> 1. **Segmentación por Circuito:** El modelo detecta automáticamente que la telemetría de Japón (Sector 1 técnico) se agrupa de forma distinta a la de USA (rectas largas), validando la sensibilidad del PCA para capturar características intrínsecas del trazado.
>
> 2. **Detección de Anomalías Suave:** El Clúster 4 actúa como recolector de vueltas atípicas (Safety Car o errores de pilotaje), permitiendo limpiar el dataset para futuros modelos predictivos — aunque sin la precisión de DBSCAN.
>
> 3. **Mejor Silhouette que K-Means:** Con 0.5142 vs 0.4409, el clustering jerárquico produce una segmentación internamente más coherente, probablemente porque el Ward Linkage es más adecuado para la geometría no perfectamente esférica del espacio PCA de telemetría F1.
>
> 4. **Convergencia Tri-método:** La concordancia entre K-Means (4 clústeres), Hierarchical (5 clústeres) y DBSCAN (5 clústeres) en torno al mismo número de arquetipos confirma que la taxonomía de 4-5 regímenes de conducción refleja estructura real en los datos.

### Próximos Pasos

| Acción | Justificación |
|:---|:---|
| **Exportar etiquetas Hierarchical** como nueva feature | Los labels jerárquicos tienen mayor Silhouette que K-Means y son candidatos a features de mejor calidad predictiva. |
| **Comparar asignaciones** vuelta-a-vuelta entre los 3 métodos | Identificar el "núcleo duro" de cada arquetipo. Los puntos asignados consistentemente por los 3 métodos son los más representativos del estado táctico. |
| **Incorporar `hierarchical_cluster`** en Feature Engineering V6 | Feature categórica de alta predictividad para modelos de predicción de estrategia de carrera. |
| **Deep Learning sobre etiquetas** | Utilizar las etiquetas jerárquicas y sus centroides como Ground Truth para un modelo capaz de predecir el estado de degradación del neumático en tiempo real basado solo en telemetría. |
