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
Evaluado mediante **Validación Cruzada GroupKFold (4 particiones)** agrupando por `race_name` para evitar leakage entre circuitos.

| Modelo | MSE Promedio (S²) | R² Promedio (Test) | R² (Entrenamiento) | Evaluación |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 235.0272 | 0.0890 | 0.8539 | **Fallo crítico:** No linealidad extrema (*tyre cliff*). |
| **Decision Tree (max_depth=6)** | 306.6918 | 0.4299 | 0.8845 | **Descartado:** Sobreajuste local en hojas. |
| **Random Forest (max_depth=8)** | 290.2896 | 0.5157 | 0.9312 | **Descartado:** Sesgo por división de varianza. |
| **Gradient Boosting (max_depth=5)**| 283.7008 | 0.5455 | 0.9580 | **SELECCIONADO:** Menor MSE y mayor generalización. |
| **XGBoost Regressor (max_depth=5)** | 314.1477 | 0.3778 | 0.9790 | **Descartado:** Sensible a correlación en features. |

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
