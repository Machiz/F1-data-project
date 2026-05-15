# Documentación Técnica: Reducción Dimensional por Embeddings (Manifold Learning)

Este reporte detalla la implementación y los descubrimientos de la generación de Embeddings no lineales mediante t-SNE, aplicados sobre el dataset de alta dimensionalidad de eventos tácticos de F1.

## 1. Contexto: ¿Por qué Embeddings en Tabular Data?
Mientras que el **PCA** (Principal Component Analysis) ya implementado en el proyecto es excelente para eliminar correlaciones lineales (multicolinealidad), las tácticas de carrera en la Fórmula 1 suelen tener límites de decisión curvos o topologías complejas que un modelo lineal no puede desenredar completamente.

Utilizar **t-SNE** (t-Distributed Stochastic Neighbor Embedding) permite aplicar un algoritmo de *Manifold Learning* que proyecta los ~500 features en un "espacio latente" de 2D o 3D. El objetivo de este algoritmo es preservar la "vecindad local": si dos maniobras de ataque ocurrieron con telemetrías y desgastes casi idénticos, t-SNE asegura que aterricen matemáticamente una al lado de la otra en la proyección final.

## 1.1. ¿Cómo funciona t-SNE matemáticamente?

El algoritmo t-SNE no comprime varianza de forma lineal (como PCA), sino que intenta mapear y replicar "probabilidades de vecindad". Su funcionamiento se rige por los siguientes pasos clave:

1. **Similitud en Alta Dimensionalidad:** Primero, t-SNE mide la distancia geométrica entre todos los eventos en el espacio original (nuestras ~500 variables). Estas distancias se convierten en probabilidades usando una distribución Gaussiana. Si dos tácticas son casi idénticas, tienen una probabilidad altísima de elegirse mutuamente como "vecinos".
2. **Similitud en Baja Dimensionalidad (Distribución t-Student):** Luego, el algoritmo proyecta puntos aleatorios en un lienzo 2D o 3D. Aquí vuelve a medir las distancias, pero en lugar de usar una campana de Gauss, usa una distribución **t-Student**. Al tener "colas más pesadas", la t-Student permite que los eventos que son muy diferentes se "empujen" y se separen más agresivamente en el mapa, evitando que todo colapse en el centro (*Crowding Problem*).
3. **Optimización (Kullback-Leibler):** El objetivo matemático es que la matriz de probabilidades del lienzo 2D/3D sea exactamente igual a la matriz del espacio original. Para lograrlo, t-SNE utiliza descenso de gradiente para mover iterativamente los puntos en el gráfico, minimizando la diferencia entre ambas matrices de probabilidad (medida mediante la Divergencia de Kullback-Leibler).
4. **Resultado Final (Embedding):** Cuando el modelo converge y el error es mínimo, las coordenadas estáticas en las que se detienen los puntos se convierten en nuestro Embedding final.

## 2. Proceso de Generación y Archivos
*   **Notebook Principal:** `project/notebooks/dimensionality reduction/tSNE_Embeddings_Manifold_Learning.ipynb`
*   **Pipeline de Preprocesamiento:** 
    *   Filtro estricto para remover variables de identificación o el Target (`pos_change`), previniendo sesgos algorítmicos.
    *   Imputación de mediana (`SimpleImputer`) para manejar variables de telemetría faltantes.
    *   Estandarización estricta (`StandardScaler`) para homogeneizar las magnitudes de tiempos de vuelta vs velocidades.
*   **Parámetros Topológicos:** El modelo t-SNE se configuró con una **Perplexity de 30**, lo que balancea la atención del algoritmo entre la estructura local inmediata y la estructura global del dataset.

## 2.1. Justificación de la Cantidad de Dimensiones (2D y 3D)

La elección de proyectar el espacio latente estrictamente en **2 y 3 dimensiones** responde a dos justificaciones técnicas críticas (una algorítmica y otra de negocio):

1. **Limitación Matemática (Algoritmo Barnes-Hut):** La implementación óptima de t-SNE (utilizada a través de `scikit-learn`) emplea la aproximación *Barnes-Hut*. Este algoritmo reduce la complejidad computacional masivamente (de $O(N^2)$ a $O(N \log N)$), lo que permite que el código corra rápido en datasets densos. Sin embargo, matemáticamente, **Barnes-Hut está confinado exclusivamente a 2 o 3 componentes**. Forzar el modelo a 4 o más dimensiones requeriría cambiar al método `exact`, el cual es computacionalmente prohibitivo e inviable para el volumen de telemetría de Fórmula 1.
2. **Objetivo Dual (Feature Engineering + Interpretabilidad):** A diferencia de PCA (donde extrajimos 15 dimensiones), el objetivo fundamental de Manifold Learning es la interpretación topológica. El análisis táctico exige validación visual humana. Extraer 3 componentes maximiza la cantidad de información latente que le daremos a los algoritmos predictivos posteriores (Machine Learning), al mismo tiempo que se mantiene en el límite de lo que un analista puede explorar, graficar y rotar en un entorno tridimensional.

## 2.2. Comparativa Estratégica: t-SNE(Raw Features) vs t-SNE(PCA Components)

El pipeline de embeddings se ejecutó de forma paralela en dos fuentes de datos distintas para comparar sus resultados topológicos:
1. **t-SNE sobre ~500 Raw Features:** Se inyectó la matriz de variables original estandarizada. El resultado es un mapa latente que preserva la vecindad usando todas las variables del sistema, pero que a menudo presenta alta dispersión debido al "ruido" de la multicolinealidad.
2. **t-SNE sobre 15 PCA Components:** Se inyectó la matriz reducida extraída del `pca_scores.parquet`. En este flujo, el modelo PCA actúa primero como un filtro de ruido masivo.

> [!NOTE]
> **Hallazgo Clave (Insight):** El notebook demuestra visualmente que aplicar t-SNE sobre los componentes del PCA genera agrupamientos **mucho más definidos, compactos y con menos dispersión estocástica**. Al eliminar primero las redundancias lineales con PCA, liberamos a t-SNE para que enfoque todo su poder matemático en clasificar puramente la "señal táctica real".

## 3. Exploración Visual del Espacio Latente
El notebook genera 4 perspectivas críticas sobre el espacio topológico descubierto:

### 3.1. Separación de Tácticas (2D Scatter)
El primer panel proyecta las coordenadas latentes coloreadas por `event_type` y tasa de éxito. A diferencia del PCA, donde los eventos pueden parecer una sola "nube", t-SNE genera estructuras de islas o "penínsulas". Se puede observar que los *Undercuts* se proyectan en cuadrantes aislados de los *Ataques Físicos*, validando que a nivel de datos (sin decirle al modelo qué es qué), la matemática subyacente de la F1 los considera mundos separados.

### 3.2. Mapa de Densidad (Hexbin)
El gráfico de *Hexbin Plot* responde a la pregunta de volumen: ¿dónde ocurre la mayoría de la acción? Las celdas brillantes ("puntos calientes") revelan los centros de gravedad de las tácticas más estándar de la F1 moderna, en contraste con las tácticas exóticas que caen en las afueras de la galaxia 2D.

### 3.3. Feature Overlay (Temperatura de Velocidad)
El mapa coloreado por la variable continua de la velocidad del atacante (`att_st_speed_mean`) prueba de forma espectacular que la dimensión latente tiene sentido físico. Una zona entera del espacio latente se tiñe de color extremo (alta velocidad), demostrando que el modelo ha agrupado de forma no supervisada todos los eventos ocurridos en rectas de alta energía (ej. Bakú, Monza).

### 3.4. Topología en 3 Dimensiones (3D Scatter)
El notebook incluye una proyección tridimensional con capacidades rotativas visuales (si se usa de forma interactiva). En datasets tan complejos (500 variables), la proyección 2D puede sufrir el problema de "apelotonamiento" (*crowding problem*). Al inyectar una dimensión $Z$, el modelo t-SNE logra desenredar grupos que en 2D parecían chocar, confirmando que la topología real de las carreras requiere al menos 3 dimensiones para apreciarse.

## 4. Conclusiones y Próximos Pasos (Dataset Generado)
El proceso termina guardando las coordenadas puras (`Emb_2D_X, Emb_2D_Y, Emb_3D_Z`) en un nuevo artefacto de datos (`../data/features/tactical_embeddings.parquet`). 

> [!TIP]
> **Siguiente Paso Estratégico:**
> Estos vectores latentes ultra-densos se pueden inyectar directamente como variables predictoras en un modelo de clasificación final (como un Random Forest o XGBoost). Al estar ya destiladas las similitudes topológicas complejas, el clasificador posterior podrá aprender las reglas de éxito de la F1 de forma exponencialmente más rápida y con mucho menos ruido que si usara las 500 variables originales.
