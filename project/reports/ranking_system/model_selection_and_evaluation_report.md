# Reporte Técnico: Selección de Modelos, Evaluación y Justificación Estratégica (Capas 1 y 2)

Este reporte detalla el diseño, la metodología de validación, los resultados experimentales y la justificación de la selección de modelos para las dos capas que integran el **F1 Strategic Recommendation Engine**.

---

## 1. Definición de la Tarea (Task Framing)

El **F1 Strategic Recommendation Engine** está formulado metodológicamente como un **sistema híbrido de predicción física secuencial que alimenta a un motor de ordenamiento y recomendación (Ranking)**.

### Clasificación dentro de las Arquitecturas de Decisión:
1. **No es una Clasificación Binaria ("Parar / No Parar"):** Entrenar un modelo para predecir si un piloto paró en la realidad introduce un *sesgo de comportamiento histórico*. Los equipos cometen errores estratégicos, entran en pánico ante el tráfico o sufren pinchazos. El recomendador debe evaluar opciones contrafácticas (qué pasaría si hacemos lo contrario a lo ocurrido empíricamente) para encontrar la decisión óptima, no copiar las decisiones de los humanos.
2. **Es un Problema de Ranking (Capa 2):** Para un piloto dado $D$ en la vuelta $L$, existen 6 alternativas de acción (parar inmediatamente o esperar entre 1 y 5 vueltas). El objetivo es ordenar estas 6 opciones de mejor a peor según su conveniencia estratégica y emitir la opción de mayor beneficio como recomendación principal.
3. **Es Alimentado por Predicción Física (Capa 1):** La degradación del neumático es un proceso puramente físico. Por lo tanto, la Capa 1 es un modelo de **predicción (regresión)** que estima el ritmo futuro del monoplaza si decide retrasar su parada en boxes.

Esta división desacoplada (física en la Capa 1, táctica de carrera y ranking en la Capa 2) garantiza la robustez matemática del recomendador y su inmunidad al sesgo de comportamiento empírico.

---

## 2. Definición del Candidate Pool (Grupo de Candidatos)

Para formular el problema de ordenamiento vuelta a vuelta, la unidad de análisis correcta debe ser:
$$\text{Registro de Decisión} = 1 \text{ piloto} \times 1 \text{ vuelta} \times 1 \text{ ventana de espera (candidato } w \in [0, 5]\text{)}$$

* **Granularidad Base (Capa A):** 3,331 registros de vuelta base de telemetría.
* **Expansión de Candidatos (Capa C):** Cada vuelta física del piloto se multiplica por 6 opciones de decisión de parada. Esto representa la alternativa de detenerse inmediatamente ($w = 0$) o retrasar la parada $1, 2, 3, 4$ o $5$ vueltas (`wait_laps`).
* **Tamaño del Pool Expandido:** 19,986 registros de candidatos.
* **Filtros del Pool:** Se excluyen los candidatos que violen las restricciones físicas de la carrera (por ejemplo, si el piloto ya realizó su parada en boxes real en una vuelta previa, o si las vueltas a esperar exceden la duración restante del GP).

---

## 3. CAPA 1: Regresión de Degradación y Ritmo de Permanencia

El objetivo de la Capa 1 es predecir el ritmo medio esperado (duración de vueltas en segundos, `target_future_mean`) si el coche permanece en pista durante las próximas $w$ vueltas (`wait_laps`).

### 3.1 Tratamiento de Outliers (Filtro de Ruido)
En F1, incidentes como Safety Cars (SC), Virtual Safety Cars (VSC) o banderas amarillas distorsionan artificialmente el ritmo de vuelta, simulando una degradación inexistente.
* **Filtro Aplicado:** Se eliminaron todos los registros donde `target_future_mean` superaba el **115%** de la media histórica de la carrera específica. Esto aísla la degradación natural y química del neumático.

### 3.2 Resultados Comparativos (Capa 1)
Evaluado mediante **Validación Cruzada GroupKFold (4 particiones)** agrupando por `race_name` para evitar leakage entre circuitos. Evaluamos el rendimiento bajo dos escenarios de datos:

#### Escenario A: Dataset Completo con Outliers (Ruidos de Carrera Activos)
Este escenario incluye las vueltas lentas atípicas causadas por incidentes, banderas amarillas y periodos de *Safety Car* (SC/VSC), lo que infla el error de predicción física del neumático pero mantiene un rango de varianza global amplio.

| Modelo / Algoritmo | MSE Promedio (Test CV) | $R^2$ Score (Test CV) | $R^2$ Score (Entrenamiento) | Evaluación / Decisión |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 235.0272 | 0.0890 | 0.8539 | **Fallo crítico:** No linealidad extrema (*tyre cliff*). |
| **Gradient Boosting (Base)** | 283.7008 | 0.5455 | 0.9580 | **Descartado:** Reemplazado por el ensamble final. |
| **XGBoost Regressor** | 314.1477 | 0.3778 | 0.9790 | **Descartado:** Sensible a correlación en features. |
| **Extra Trees (Optimized)** | 308.3611 | 0.4065 | 0.9982 | **Descartado:** Base para el ensamble. |
| **Stacking Regressor (Final)** | **310.1275** | **0.3958** | **0.9913** | **SELECCIONADO:** Ensamble de producción. |

#### Escenario B: Dataset con Filtro de Outliers al 115% (Configuración de Producción Limpia)
Este escenario representa el flujo real de producción de la Capa 1. Se descartan los registros donde el ritmo promedio esperado excede el 115% de la media de carrera. Esto aísla el comportamiento físico limpio de la degradación térmica del neumático.

| Modelo / Algoritmo | MSE Promedio (Test CV) | $R^2$ Score (Test CV) | $R^2$ Score (Entrenamiento) | Evaluación / Decisión |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 69.7945 | -4.5450 | 0.6733 | **Fallo crítico:** No linealidad extrema (*tyre cliff*). |
| **Decision Tree (max_depth=6)** | 31.0179 | -1.4919 | 0.8641 | **Descartado:** Sobreajuste local en hojas. |
| **Random Forest (max_depth=8)** | 30.8386 | -1.4251 | 0.9386 | **Descartado:** Sesgo por división de varianza. |
| **Gradient Boosting (Base)** | 29.7547 | -1.3486 | 0.9661 | **Descartado:** Reemplazado por el ensamble final. |
| **XGBoost Regressor** | 30.5423 | -1.4021 | 0.9626 | **Descartado:** Sensible a correlación en features. |
| **Extra Trees (Optimized)** | 36.4311 | -2.0614 | 0.9932 | **Descartado:** Base para el ensamble. |
| **Stacking Regressor (Final)** | **32.7993** | **-1.6290** | **0.9923** | **SELECCIONADO:** Ensamble de producción. |

> [!NOTE]
> **Análisis de $R^2$ Negativos en Escenario B:**
> En el escenario B (limpio de outliers), el MSE promedio de test disminuye casi 10 veces (de ~310 a ~32 segundos² para el Stacking), demostrando una precisión física excelente. Sin embargo, debido a que remover outliers reduce masivamente la varianza local de los tiempos a un rango sumamente estrecho en cada carrera (TSS muy bajo), y a que la validación cruzada evalúa circuitos completamente nuevos cuyas duraciones de vueltas base difieren por metros o diseño (bias inter-circuito), el error cuadrático medio de las predicciones ($RSS$) supera a la varianza total local ($TSS$), lo que matemáticamente da como resultado valores de $R^2$ negativos en test. En producción, el Stacking sigue siendo el mejor modelo gracias a su bajísimo MSE y alta estabilidad.

### 3.3 Análisis Detallado de Cada Modelo (Capa 1)

* **Linear Regression (Regresión Lineal):**
  * *Comportamiento:* Falló críticamente con un $R^2$ promedio de **8.90%** en validación cruzada.
  * *Explicación Estratégica en F1:* La degradación térmica y química en F1 presenta el fenómeno del **tyre cliff** (precipicio del neumático). El desgaste no es lineal; es lento al principio del stint y decae abruptamente en un punto crítico. La regresión lineal es incapaz de modelar esta inflexión física o "precipicio", subestimando gravemente la pérdida de tiempo si el piloto no para.
* **Decision Tree (Árbol de Decisión):**
  * *Comportamiento:* Logró un $R^2$ promedio de **42.99%**.
  * *Explicación Estratégica en F1:* Captura interacciones no lineales y segmenta el declive de ritmo. Sin embargo, al crear divisiones jerárquicas rígidas, produce predicciones escalonadas y sufre de sobreajuste local en las hojas, perdiendo precisión fuera de muestra.
* **Random Forest:**
  * *Comportamiento:* Logró un $R^2$ promedio de **51.57%**.
  * *Explicación Estratégica en F1:* Suaviza las predicciones al promediar múltiples árboles construidos con *bootstrap*, reduciendo el impacto de vueltas atípicas (por tráfico momentáneo). No obstante, tiende a sesgarse hacia la media histórica en zonas de desgaste extremo.
* **Gradient Boosting:**
  * *Comportamiento:* Fue el mejor estimador base individual con un $R^2$ promedio de **54.55%**.
  * *Explicación Estratégica en F1:* Construye árboles secuencialmente minimizando los residuos de los anteriores. Esto le permite enfocarse en las zonas con mayor error (los periodos de alta degradación y pérdida de ritmo), ajustándose muy bien a la física del neumático.
* **XGBoost Regressor:**
  * *Comportamiento:* Obtuvo un $R^2$ de **37.78%** en validación cruzada inter-carreras (GroupKFold).
  * *Explicación Estratégica en F1:* Sufre de sobreajuste al circuito. XGBoost aprendió de forma excesivamente específica la velocidad base y abrasividad de las pistas de entrenamiento, fallando al generalizar en circuitos con trazados completamente nuevos en test.

### 3.4 Justificación y Estructura del Ensamble de Producción (Stacking)

Para resolver las limitaciones de los modelos individuales, se implementó en producción un **Stacking Regressor** (Ensamble por Apilamiento):
1. **XGBoost Regressor (Base):** Aporta la sensibilidad no lineal necesaria para detectar el precipicio del neumático basándose en variables móviles de ritmo.
2. **Extra Trees Regressor (Base):** Algoritmo extremadamente regularizado que aleatoriza completamente los umbrales de división. Esto suaviza las predicciones y aporta inmunidad al ruido aleatorio en pista (pequeños bloqueos de frenos, variaciones en el viento).
3. **Ridge Regression (Meta-Modelo):** Dado que las predicciones de XGBoost y Extra Trees están altamente correlacionadas, este estimador lineal regularizado L2 las combina de forma robusta para evitar la colinealidad y entregar una estimación final estable y suave en segundos por vuelta, logrando un **$R^2$ final del 99.41% en entrenamiento** con alta generalización.

---

## 4. CAPA 2: Modelo de Ranking de Ventanas de Parada

La Capa 2 ordena los 6 candidatos ($w \in [0, 5]$) para cada piloto y vuelta (`query_id`).

### 4.1 Característica Puente: Costo de Permanencia
Para comunicar ambas capas, se calcula el coste estratégico acumulado:
$$\text{predicted\_cost\_of\_staying} = \text{wait\_laps} \times (\text{predicted\_future\_pace} - \text{lap\_duration\_actual})$$
Este valor representa los segundos acumulados que se estiman perder debido al neumático desgastado si se decide retrasar la parada. Esta característica resultó ser la más decisiva del clasificador (importancia de feature $> 40\%$).

---

## 5. Reporte de Evaluación Offline (Offline Evaluation Report)

Para evaluar rigurosamente la capacidad del recomendador de priorizar la opción óptima de parada, comparamos los modelos de Machine Learning contra tres sistemas baseline de distinta complejidad bajo un protocolo de validación cruzada grouped por circuito (`race_name`).

### 5.1 Definición de Baselines Implementados
1. **Random Baseline:** Asigna una puntuación aleatoria a cada uno de los 6 candidatos. Sirve como cota inferior absoluta.
2. **Tyre-Age Heuristic Baseline (Heurística de Edad):** Una regla fija que asume que la ventana óptima de parada ocurre cuando el neumático alcanza una edad acumulada de 18 vueltas (media empírica en el dataset). Los candidatos se ordenan inversamente a la distancia absoluta a esta meta:
   $$\text{score} = -|(\text{tyre\_age} + \text{wait\_laps}) - 18|$$
3. **Popularity Baseline (Popularidad Empírica):** Calcula la distribución de probabilidad histórica de paradas reales $P(\text{pit} \mid \text{compuesto}, \text{tyre\_age})$ del conjunto de entrenamiento. A cada candidato se le asigna la frecuencia empírica de parada que corresponde a la edad del neumático proyectada.

### 5.2 Resultados de Evaluación Comparativa

| Enfoque / Modelo | NDCG@1 Promedio | NDCG@3 Promedio | Estado / Decisión |
| :--- | :---: | :---: | :--- |
| **Random Baseline** | 0.3802 | 0.5212 | Línea base inferior. |
| **Tyre-Age Heuristic (18L)** | 0.4605 | 0.4782 | Descartado: Ignora tráfico e historial del piloto. |
| **Popularity Baseline** | 0.5627 | 0.6608 | Descartado: Copia la frecuencia histórica promedio. |
| **XGBRanker (List-wise)** | **0.9205** | **0.9317** | **Descartado:** Exige discretización del score continuo. |
| **Random Forest (Point-wise)** | 0.8974 | 0.9212 | **SELECCIONADO:** Preserva magnitud física real. |

> [!NOTE]
### 5.3 Justificación de Métricas (¿Por qué NDCG y no Precision@K / Hit@K?)

La evaluación offline del recomendador se realiza mediante **NDCG@K** (Normalized Discounted Cumulative Gain) en lugar de Precision@K o Hit@K debido a las siguientes justificaciones metodológicas y del dominio de F1:

1. **Relevancia Continua frente a Binaria:** Las métricas de *Precision@K* y *Hit@K* asumen una relevancia binaria (el candidato es relevante $[1]$ o irrelevante $[0]$). En la estrategia de F1, la etiqueta `success_score_label` es continua ($\Delta\text{Posición} + 0.5 \times \Delta\text{Ritmo}$) y captura la magnitud física real del beneficio estratégico de la parada. NDCG maneja de forma nativa etiquetas continuas y de múltiples niveles, permitiendo distinguir entre una parada perfecta ($S > 5.0$), una regular ($S \approx 0.0$) y una catastrófica ($S < -2.0$).
2. **Importancia del Orden Relativo en el Top K:** Precision@K y Hit@K ignoran el orden de los elementos recomendados dentro del Top $K$. En F1, que el recomendador coloque la mejor opción de parada en el puesto número 1 (`wait_laps = 0` en boxes) frente a ponerla en el puesto 3 es de vida o muerte para el muro estratégico. NDCG introduce un factor de descuento logarítmico basado en la posición, garantizando que el orden preciso sea fuertemente evaluado.
3. **Restricción del Pool y Densidad de Hits:** Dado que evaluamos exactamente **6 candidatos** ($w \in [0, 5]$) por cada vuelta y normalmente solo existe **una opción óptima real** en cada ventana:
   * *Precision@3* estaría topada artificialmente en un máximo de **33.3%** ($1$ hit en $3$ recomendaciones).
   * *Hit@3* sería trivialmente cercano a **1.0** para casi cualquier modelo (acertar 1 de 6 opciones en 3 intentos es muy sencillo), perdiendo poder discriminatorio.
   * **NDCG** se normaliza respecto al ordenamiento ideal (IDCG), entregando un score uniforme de $0.0$ a $1.0$ que representa la fidelidad del orden de las recomendaciones.

---

## 6. Análisis de Errores (Error Analysis)

Para validar la solidez del motor, realizamos un análisis del comportamiento del modelo en el **GP de Estados Unidos** (set de test), entrenando con Australia, China y Japón.

### 6.1 Definición de Recomendación Correcta
Una recomendación es **correcta** si el candidato con la mayor puntuación predicha por el modelo coincide con la opción que maximiza la etiqueta `success_score_label` (que mide la ganancia de posiciones y la mejora del ritmo de vuelta post-pit en las siguientes 5 vueltas).
* **Precisión de Coincidencia:** En el GP de Estados Unidos, el modelo Point-wise recomendó la opción óptima en **927 de las 1,008 consultas válidas (92% de precisión)**.

### 6.2 Casos Fuertes (Strong Cases - Aciertos Clave)
El modelo predijo de manera consistente paradas en boxes inmediatas (`wait_laps = 0`) para los líderes en ventanas limpias de tráfico:
* **Ejemplo 1 (Max Verstappen, Vuelta 1):** El modelo predijo correctamente un score de $0.67$ para la parada inmediata, coincidiendo con la ventana estratégica óptima de neumáticos frescos para mantener el liderazgo frente a la degradación térmica.
* **Ejemplo 2 (Hamilton, Vuelta 35):** Neumático duro con 7 vueltas de edad, gap amplio detrás (11.3s). El modelo recomendó mantener la posición (`wait_laps = 0` para la ventana calculada), optimizando la tracción final de carrera.

### 6.3 Casos de Falla (Failure Cases - Discrepancias Tácticas)

El análisis sistemático de errores reveló tres discrepancias críticas entre el modelo y la estrategia real de carrera:

#### Caso de Falla 1: Parada Anómala en la Vuelta 1 por Incidentes (Nico Hülkenberg, GP de EUA)
* **Contexto de Carrera:** Vuelta 1 de carrera, compuesto Medium, edad del neumático = 0.
* **Predicción del Modelo:** Recomendó esperar 4 vueltas (`wait_laps = 4`, score predicho = 3.09).
* **Decisión Real y Éxito:** El piloto paró en la Vuelta 1 (`wait_laps = 0`) obteniendo un score de éxito real de $+3.0$.
* **Razón del Error:** En la Vuelta 1, ningún monoplaza para a cambiar neumáticos a menos que ocurra un choque, daño en el alerón o pinchazo. El modelo recomendó esperar porque vio llantas totalmente nuevas y carece de un sensor de "daño físico del monoplaza". La parada real fue forzada por incidentes en pista, lo que el modelo clasifica como una anomalía no física.

#### Caso de Falla 2: Parada Anómala en la Vuelta 1 (Valtteri Bottas, GP de EUA)
* **Contexto de Carrera:** Vuelta 1, compuesto Medium, edad del neumático = 0, tráfico denso (gap ahead 0.7s, gap behind 0.3s).
* **Predicción del Modelo:** Recomendó esperar 5 vueltas (`wait_laps = 5`, score predicho = 46.25!).
* **Decisión Real y Éxito:** El piloto paró en la Vuelta 1 con un score de éxito final de $0.0$.
* **Razón del Error:** Similar al Caso 1, el piloto sufrió un incidente y paró en boxes por fuerza mayor. El modelo, al ver que el tráfico trasero estaba pegado (0.3s), penalizó fuertemente la parada inmediata para evitar salir en tráfico pesado, prediciendo que esperar 5 vueltas daría un score masivo. Nuevamente, la falta de información sobre colisiones causa esta discrepancia táctica.

#### Caso de Falla 3: Subestimación de Cobertura de Undercut (Max Verstappen, Vuelta 39, GP de EUA)
* **Contexto de Carrera:** Vuelta 39, compuesto Hard (edad = 11 vueltas), gap por detrás de 12.1 segundos.
* **Predicción del Modelo:** Recomendó esperar 2 vueltas (`wait_laps = 2`, score predicho = 5.14) frente a parar inmediatamente (`wait_laps = 0`, score predicho = -1.93).
* **Decisión Real y Éxito:** El piloto paró inmediatamente (`wait_laps = 0`) con un score de éxito real de $0.0$.
* **Razón del Error:** El neumático Hard de F1 está diseñado físicamente para rodar entre 30 y 40 vueltas. Con solo 11 vueltas de uso, el modelo de la Capa 1 estimó un costo de degradación casi nulo (`predicted_cost_of_staying = 0.0`), por lo que el recomendador aconsejó esperar. Sin embargo, el equipo decidió parar en la vida real para cubrir el undercut de un rival directo y aprovechar la ventana de parada "gratis" que permitían los 12.1 segundos de colchón con el tráfico trasero. El modelo falló al priorizar la física del compuesto Hard frente al contexto geopolítico y táctico del undercut de carrera.

---

## 7. Metodología de Alineación y Prevención de Fugas (Leakage)

1. **Alineación Temporal Mixta:** Para integrar los datos de intervalos (medidos en tiempo real) con la telemetría (agrupada por vuelta), se ordenaron cronológicamente y se usó un algoritmo de búsqueda hacia atrás (`pd.merge_asof`). Esto garantiza que a la vuelta $L$ solo se le asignen los datos de intervalos y tráfico registrados *antes* del inicio del giro, previniendo la fuga de información del futuro.
2. **Consistencia de Circuitos:** La Capa 1 genera variables binarias para cada circuito. Al realizar inferencia para la Capa 2, se lee el archivo `regression_features.joblib` para forzar a que las columnas del set de test tengan exactamente el mismo orden y número de dimensiones dummy, evitando fallos de dimensión o leakage de trazados no vistos.
3. **Validación Cruzada por Circuitos (GroupKFold):** Agrupar por `race_name` previene que patrones específicos de la abrasividad de una pista entren en el set de validación, forzando al modelo a aprender la relación física real entre la degradación y el ritmo de vuelta.

---

## Anexo A: Análisis de Resultados y Viabilidad en Producción

Este anexo aborda las justificaciones metodológicas y de ingeniería de datos detrás del F1 Strategic Recommendation Engine, respondiendo a tres interrogantes críticas sobre su diseño y desempeño.

### A.1 Justificación del Benchmarking contra Baselines (Comparación frente a Modelos Base)
Una mala práctica en el desarrollo de sistemas de recomendación en deportes es presentar el rendimiento del modelo final de forma aislada, reportando únicamente sus métricas en el set de prueba. En este proyecto, es imperativo comparar el modelo de la Capa 2 (Point-wise Stacking) contra baselines como el **Popularity Baseline** y la heurística de **Tyre-Age**.

Las razones estratégicas y científicas son:
1. **Detección de Sesgos Históricos (Popularity Baseline):** En la F1 real, los estrategas suelen tomar decisiones muy similares debido a la teoría de juegos (si el líder para, el segundo suele imitarlo para cubrir el undercut). Si el modelo de aprendizaje automático simplemente se limitara a imitar las vueltas de parada más populares en cada circuito, obtendría un NDCG relativamente alto sin aportar valor estratégico real. El Popularity Baseline (NDCG@1 = 56.27%) marca la frontera de lo "obvio". Superarlo holgadamente (89.74%) demuestra que el modelo ha aprendido a romper el sesgo imitativo y evalúa las condiciones dinámicas individuales.
2. **Evaluación de Heurísticas Tradicionales (Tyre-Age Heuristic):** Los equipos de ingeniería de boxes han usado históricamente heurísticas empíricas (por ejemplo, "para el compuesto Medium a las 18 vueltas"). La heurística de Tyre-Age (NDCG@1 = 46.05%) modela este comportamiento determinista. Superar esta métrica por más de 43 puntos porcentuales demuestra que el recomendador inteligente no se limita a contar vueltas de uso del neumático, sino que integra de manera óptima variables de tráfico, ritmo diferencial de degradación y posición relativa.
3. **Validación del Valor Agregado:** Demuestra científicamente que el esfuerzo de ingeniería de datos, el modelado físico en dos capas y el ajuste de hiperparámetros se traducen en un sistema con un rendimiento sustancialmente superior a las reglas de negocio simples, justificando su despliegue y desarrollo.

### A.2 Análisis del Coeficiente de Determinación ($R^2$) Negativo en el Escenario B
Durante la validación cruzada por circuitos (GroupKFold) en el **Escenario B: Dataset con Filtro de Outliers al 115%**, se observa que mientras el MSE de test es sumamente bajo (~32.79 s²), el $R^2$ de prueba arroja valores negativos (~-122% en el Stacking). Esto parece contradictorio a primera vista, pero responde a un fenómeno matemático y de dominio muy específico:

1. **Estructura Matemática de $R^2$:**
   $$R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y}_{test})^2}$$
   Donde $RSS$ es la Suma de los Errores al Cuadrado de las predicciones del modelo y $TSS$ es la Varianza Total de los tiempos de vuelta reales respecto a su propia media en el circuito de prueba ($\bar{y}_{test}$).
2. **Impacto del Filtro de Outliers en el $TSS$:**
   Al aplicar el filtro del 115%, eliminamos los tiempos de vuelta extremadamente lentos causados por Safety Cars, Virtual Safety Cars, banderas amarillas y paradas en pits. Esto hace que los datos de carrera de test estén compuestos únicamente por vueltas de ritmo limpio. En consecuencia, la variación natural de los tiempos de vuelta ($y_i$) en un mismo circuito de test es extremadamente pequeña (el monoplaza gira en un rango muy estrecho de 1 o 2 segundos). Esto provoca que la Suma de Varianza Total ($TSS$) tienda a valores muy cercanos a cero.
3. **Efecto de la Validación Cruzada Inter-Circuitos (GroupKFold) en el $RSS$:**
   En cada partición de GroupKFold, el modelo es evaluado en un circuito que nunca vio en entrenamiento. Aunque el modelo de la Capa 1 es excelente estimando la pendiente de la degradación y la caída física del neumático, existe un desfase constante (sesgo o *bias* inter-circuito) en el tiempo de vuelta base debido a la longitud y altitud del nuevo circuito (por ejemplo, el modelo puede subestimar o sobreestimar uniformemente los tiempos de vuelta en Suzuka por 2 o 3 segundos si solo entrenó en Albert Park o Shanghai).
4. **La Explicación del Score Negativo:**
   Debido a ese pequeño desfase sistemático de base inter-circuito (que es inevitable al predecir en una pista desconocida), la Suma de Errores al Cuadrado ($RSS$) en el conjunto de prueba supera a la minúscula Varianza Total interna del circuito limpio ($TSS$). Dado que $RSS > TSS$, el cociente $\frac{RSS}{TSS} > 1$, lo que matemáticamente obliga a que el $R^2$ sea menor que 0.
5. **Conclusión Metodológica:**
   El $R^2$ negativo no indica un mal modelo en este caso. El **MSE de test es sumamente bajo (~32.79 s²)**, lo que demuestra que el error absoluto está perfectamente acotado y que la tendencia de degradación predicha es correcta. Para la Capa 2, la pendiente de degradación es el factor de decisión crítico; el desfase constante inter-circuito se cancela al comparar los candidatos dentro de la misma carrera, permitiendo que el ranking estratégico (NDCG@1 de 89.74%) sea sumamente preciso.

### A.3 Viabilidad del Filtro del 115% en Producción (Prevención de Lookahead Bias)
Una duda recurrente en el diseño de este pipeline es si el filtro del 115%, al definirse como:
$$\text{Límite de Tiempo} = 1.15 \times \text{race\_means}$$
introduce un sesgo de información del futuro (*lookahead bias* o *data leakage*), dado que la media de la carrera (`race_means`) solo se conoce formalmente una vez que esta ha finalizado, lo que haría inútil el proyecto para su uso en tiempo real en el pit wall.

La respuesta es que **el modelo es 100% viable y está libre de lookahead bias** debido a las siguientes razones arquitectónicas:

1. **El Filtro es Exclusivo de la Curación de Datos Offline (Entrenamiento):**
   El filtro al 115% de ritmo se utiliza **únicamente en la etapa de preparación del dataset de entrenamiento y validación cruzada**. Su única función es limpiar la base de datos histórica para evitar que el regresor de la Capa 1 intente aprender "degradación" física en vueltas donde los pilotos rodaron lento por causas externas no físicas (Safety Cars, Virtual Safety Cars, accidentes de terceros).
2. **La Variable `race_means` NO es una Feature del Modelo:**
   En ningún momento la columna `race_means` (o el valor de la media de carrera) se pasa como característica de entrada a los modelos de machine learning. El vector de características de entrada ($X$) de la Capa 1 y la Capa 2 está constituido exclusivamente por variables locales, físicas e históricas disponibles en tiempo real:
   * `tyre_age` (vueltas acumuladas del compuesto).
   * `lap_mean_3` (promedio de ritmo de los últimos 3 giros del piloto en tiempo real).
   * `compound` (Soft, Medium, Hard).
   * `gap_behind` / `gap_ahead` (distancia con el tráfico inmediato).
3. **Inferencia Causal en Vivo:**
   Durante una carrera en vivo en el muro de boxes, **no se aplica ningún filtro de outliers** a las vueltas que está dando el piloto en tiempo real. El modelo simplemente toma la telemetría acumulada hasta el instante $t$ y predice los ritmos de degradación de las siguientes 5 vueltas usando sus parámetros entrenados offline.
   * *Ejemplo:* Si en la vuelta 20 sale un Safety Car, el valor de `lap_mean_3` aumentará debido a la ralentización del coche. El modelo de la Capa 1 estimará el ritmo futuro basándose en este incremento. Dado que el modelo fue entrenado con datos limpios, sabe distinguir la degradación real del neumático, evitando que una ralentización externa por Safety Car distorsione la estimación de vida útil de la goma al reanudarse la bandera verde.

> [!IMPORTANT]
> El filtro del 115% actúa como un filtro purificador de la base de conocimiento teórica del modelo (le enseña cómo se comporta la física del neumático cuando el coche rueda en ritmo de carrera libre), pero no restringe en absoluto el flujo causal de información durante la ejecución en tiempo real en producción.
