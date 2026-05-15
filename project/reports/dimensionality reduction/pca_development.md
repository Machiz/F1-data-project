# Documentación Técnica: Análisis de Componentes Principales (PCA)

Este documento detalla la implementación, metodología y hallazgos del Análisis de Componentes Principales (PCA) desarrollado para el proyecto de datos de la Fórmula 1. Este paso es fundamental para procesar la alta dimensionalidad generada durante la fase de Feature Engineering.

## 1. Archivos Clave
El desarrollo completo del PCA se encuentra en:
*   `project/notebooks/PCA_V3.ipynb`: Notebook principal que contiene el flujo end-to-end de reducción de dimensionalidad, desde la limpieza hasta la interpretación de componentes.

## 2. Objetivo del PCA

> [!NOTE]
> **Contexto:** El proceso de Feature Engineering resultó en la creación de una matriz de datos de **alta dimensionalidad (aproximadamente 540 variables)**. Un dataset tan grande puede generar "la maldición de la dimensionalidad" y ruido durante el entrenamiento de modelos de Machine Learning.

El objetivo del PCA es:
1.  **Reducir la dimensionalidad**: Transformar las cientos de variables en un conjunto reducido de "Componentes Principales".
2.  **Eliminar Multicolinealidad**: Remover la alta correlación existente entre variables temporales (ej. medias móviles de 3, 5 y 10 vueltas).
3.  **Preservar la Varianza**: Retener la mayor cantidad de información y patrones relevantes (varianza) con la menor cantidad de variables posibles.
4.  **Preparar los Datos**: Generar un dataset óptimo, sin ruido y ortogonal, para los algoritmos de Clustering posteriores (K-Means, DBSCAN).

## 3. Preparación y Limpieza de Datos

Antes de aplicar el algoritmo matemático del PCA, los datos numéricos pasaron por un riguroso pipeline de preparación implementado con `scikit-learn`:

*   **Filtrado por Nulos**: Las columnas con más de un 50% de valores nulos fueron evaluadas (y eventualmente eliminadas si no aportaban valor).
*   **Imputación de Valores Faltantes (`SimpleImputer`)**: Los valores nulos restantes fueron rellenados utilizando la **mediana** de cada columna, evitando así que los *outliers* sesguen la imputación.
*   **Filtro de Varianza Cero**: Se eliminaron variables que contenían un valor constante para todos los eventos (varianza cero), reduciendo el dataset a 492 características efectivas.

> [!IMPORTANT]
> **Escalado Estandarizado (`StandardScaler`)**: Paso crítico para el PCA. Todas las variables fueron escaladas para tener una **media de 0 y una desviación estándar de 1**. Sin este paso, las variables con magnitudes altas dominarían artificialmente a los componentes principales.

---

## 4. Desarrollo del Modelo Matemático y Resultados

Se aplicó el modelo `sklearn.decomposition.PCA` sobre la matriz de características escalada. 

*   **Varianza Explicada**: El modelo reveló que el primer componente (PC1) captura por sí solo alrededor del **26.08%** de toda la varianza del dataset, mientras que PC1 y PC2 combinados superan el **35.86%**.
*   **Selección de Componentes**: A través de la evaluación de la varianza acumulada, se determinó que **15 componentes principales** son suficientes para retener entre el **80% y 90% de la varianza explicada**. Esto representa una reducción de dimensionalidad masiva (de ~492 a 15), reteniendo casi toda la señal predictiva.

### 4.1. Scree Plot: Varianza Explicada
El siguiente gráfico traza la varianza explicada por cada componente individual y la curva de varianza acumulada. Las líneas de referencia visuales en el 80% y 90% confirman la viabilidad de usar 15 componentes.

![Scree Plot - Varianza Explicada](images/pca_scree.png)

### 4.2. Heatmap de Loadings
El *Heatmap* muestra los *pesos* o coeficientes del top 20 de variables más influyentes. Permite identificar de forma rápida qué variables originales construyen y dan peso a cada componente principal.

![Heatmap de Loadings](images/pca_heatmap_loadings.png)

### 4.3. Biplot (Componente 1 vs Componente 2)
El Biplot es un gráfico de dispersión bidimensional que mapea los eventos tácticos en el espacio transformado, superponiendo los vectores de dirección de las características originales más fuertes. Ayuda a ver cómo se distribuyen los eventos y qué variables tiran de ellos en diferentes direcciones.

![Biplot PCA](images/pca_biplot.png)

---

## 5. Interpretación de los Componentes Principales

> [!TIP]
> Uno de los logros del notebook es el bloque de código automatizado para la **Interpretación de Componentes**. Dado que los componentes (PC1, PC2, etc.) son constructos matemáticos abstractos, el algoritmo extrae el **Top 3 de variables (loadings absolutos)** para cada componente. 

Esto permite darles un **"significado de negocio"**:
*   *Ejemplo práctico*: Si el componente está dominado por variables de desgaste de neumáticos de las últimas 5 vueltas, se le podría bautizar como el *"Componente de Degradación"*.
*   *Ejemplo práctico*: Si el componente está liderado por las variables *Delta* de velocidad punta, podría llamarse *"Componente de Ventaja de Motor"*.

## 6. Resultados y Siguientes Pasos

Al finalizar, el notebook genera un **Dataset Transformado**. En lugar de usar características raw, este nuevo dataframe contiene las variables categóricas identificadoras de la carrera y del evento, unidas únicamente a las columnas `PC1`, `PC2`, ..., `PC15`.

> [!NOTE]
> **Próximo Paso Analítico**: Este dataset reducido y descorrelacionado será inyectado directamente en el pipeline de clustering, con la expectativa de que los componentes principales agrupen de forma natural los distintos tipos de tácticas (ej. "ataque en recta por diferencial de llanta fresca" vs "overcut por retención de ritmo en sector trabado").
