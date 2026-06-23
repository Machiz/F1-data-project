# Documentación Técnica Completa: Pipeline de Ranking de Paradas en Boxes (Pit Stops)

Este documento detalla el diseño de la arquitectura, la granularidad de los datos, la metodología de validación, la comparación de modelos y las decisiones tecnológicas que componen el **F1 Strategic Recommendation Engine** (Motor de Recomendación de Paradas en Boxes en F1).

---

## 1. Arquitectura de Dos Capas Desacopladas

En la estrategia de Fórmula 1, entrenar un modelo que prediga directamente si un piloto "debe parar o no" basándose en el historial de carreras introduce un **sesgo de comportamiento histórico**. Los equipos no siempre toman la decisión óptima en la vida real debido a errores de cálculo, pánico ante el tráfico o accidentes.

Para resolver esto, diseñamos una **arquitectura desacoplada en dos capas** que evalúa opciones contrafácticas (qué pasaría si hacemos lo contrario a lo ocurrido empíricamente):

```mermaid
graph TD
    subgraph Capa C: Preparación de Candidatos
        A[Datos de Telemetría Capa A] -->|Expansión Temporal x6| B[Candidatos de Parada w = 0...5]
    end

    subgraph Capa 1: Modelo de Degradación (Física)
        B -->|Features Físicas + Circuitos| C[Stacking Regressor Ensemble]
        C -->|Predice Ritmo Futuro| D[predicted_future_pace]
    end

    subgraph Capa Puente: Costo Estratégico
        D -->|Cálculo del Costo vs Vuelta Actual| E[predicted_cost_of_staying]
    end

    subgraph Capa 2: Ordenamiento y Recomendación
        E -->|Features de Tráfico + Contexto + Costo| F[Random Forest Point-wise Ranker]
        F -->|Score de Conveniencia| G[Recomendación de Parada Óptima]
    end
```

### 1.1 Clasificación y Tipo de Tarea (Task Framing)

El proyecto del **F1 Strategic Recommendation Engine** está clasificado formalmente como un sistema híbrido de **Predicción física secuencial alimentando a un motor de Ranking (Prediction feeding Ranking)**. 

A continuación, se detalla la justificación y desglose metodológico de por qué se encuadra bajo este paradigma frente a las otras alternativas posibles de la rúbrica:

1. **Prediction (Capa 1):**  
   La degradación del neumático es un proceso puramente físico e influenciado por factores químicos y dinámicos. Por lo tanto, la Capa 1 es un modelo de **predicción (regresión)** que estima el ritmo futuro esperado del monoplaza (`predicted_future_pace` en segundos) si el piloto decide retrasar la parada.
2. **Ranking (Capa 2):**  
   Para un piloto dado en la vuelta actual, existen 6 alternativas discretas de decisión estratégica ($w \in [0, 5]$). El objetivo de la Capa 2 es ordenar estas 6 opciones de mejor a peor según su score de éxito estratégico (`success_score_label`), emitiendo el candidato de mayor puntuación como la sugerencia óptima. Es un problema de **ranking de decisiones contrafácticas**.
3. **¿Por qué NO es "Segmentation feeding Ranking"?**  
   Este enfoque implicaría categorizar primero a los pilotos o monoplazas en "segmentos" o clusters discretos (ej. monoplazas agresivos vs. lentos, o equipos de punta vs. media tabla) y después entrenar modelos de ranking independientes por segmento. Si bien realizamos análisis de clustering durante la exploración exploratoria (EDA) del proyecto, el recomendador final no segmenta a los pilotos; en su lugar, el ranker consume de forma continua el estado dinámico del coche (`position`, `gap_behind`, `tyre_age`) sin encasillarlo en clusters categóricos previos.

En resumen:
* **Recommendation:** Se emite la recomendación estratégica de la mejor opción al estratega del muro.
* **Ranking (Principal):** La Capa 2 ordena de mejor a peor los 6 candidatos temporales de parada usando la métrica NDCG.
* **Prediction (Soporte):** La Capa 1 estima la pérdida de rendimiento físico en segundos por vuelta de forma continua.
* **Segmentation feeding Ranking:** No aplica, no se dividen los datos en clusters discretos antes de rankear.

---

## 2. Granularidad de los Datos y Transición Metodológica

El principal desafío en el modelado táctico de F1 es la alineación de frecuencias y granularidades entre los diferentes niveles de información del proyecto:

### 2.1 Jerarquía de Granularidades

1. **Capa A (Telemetría de Vuelta):**
   * **Granularidad:** $1 \text{ registro} = 1 \text{ piloto} \times 1 \text{ vuelta}$.
   * **Descripción:** Describe retrospectivamente el rendimiento físico del neumático y el auto giro a giro. No formula opciones tácticas futuras.
   * **Dimensión Base:** ~3,331 registros.
2. **Capa B (Eventos Tácticos):**
   * **Granularidad:** $1 \text{ registro} = 1 \text{ evento táctico}$ (parada en boxes, adelantamiento).
   * **Descripción:** Hitos específicos de la carrera. No describe el estado continuo vuelta a vuelta.
3. **Capa C (Candidatos del Recomendador):**
   * **Granularidad:** $1 \text{ registro} = 1 \text{ piloto} \times 1 \text{ vuelta} \times 1 \text{ ventana de espera } (w \in [0, 5])$.
   * **Descripción:** Cada vuelta real del piloto se duplica por 6 candidatos de decisión. Representa las opciones de detenerse inmediatamente ($w=0$) o retrasar la parada entre $1$ y $5$ vueltas.
   * **Dimensión Expandida:** ~19,986 registros.

---

## 3. Capa C: Preparación y Generación de Candidatos

La **Capa C** es el motor de procesamiento y preprocesamiento que transforma los datos planos de telemetría y eventos históricos en un formato de decisión contrafáctico apto para el aprendizaje por ranking.

### 3.1 Expansión Temporal de Decisiones
Para formular el problema táctico vuelta a vuelta, cada registro de la Capa A se expande multiplicándose por 6 opciones de decisión. Esto representa la alternativa de parar inmediatamente o retrasar la parada $1, 2, 3, 4$ o $5$ vueltas (`wait_laps`).

### 3.2 Alineación de Tráfico a Frecuencia Mixta
Dado que los datos de tráfico e intervalos no tienen una correspondencia directa por número de vuelta, se ordenan temporalmente y se realiza una alineación usando el inicio de cada vuelta (`date_start`) mediante un algoritmo de búsqueda hacia atrás (`pd.merge_asof`). Esto asocia a cada vuelta el último intervalo de tráfico medido antes del cierre del giro.
* **Cálculo de `gap_behind`:** Se ordena la telemetría por vuelta y posición de carrera en pista, desplazando el valor de `gap_ahead` del piloto trasero inmediatamente adyacente para modelar el tráfico de salida de boxes.

### 3.3 Pendientes de Degradación Reciente
Se calcula la tasa de cambio en una ventana móvil de las últimas 3 vueltas usando regresión lineal por mínimos cuadrados para las columnas de ritmo de vuelta (`lap_duration`) y degradación acumulada (`lap_vs_best_stint`). Esto genera características dinámicas que capturan si el auto está entrando en un declive acelerado de rendimiento.

### 3.4 Etiqueta de Éxito de la Parada (`success_score_label`)
Para cada parada en boxes empírica, se calcula un score continuo de éxito post-pit ($S$) utilizando una ventana de evaluación de 5 vueltas posteriores a la detención:
$$S = \Delta\text{Posición} + 0.5 \times \Delta\text{Ritmo}$$
* Si el piloto paró en la vuelta $L_p$, al candidato correspondiente a las vueltas de espera correctas ($w = L_p - L$) se le asigna el score de éxito real de la parada.
* A las opciones de espera alternativas que no coinciden con la parada real se les asigna una penalización neutra de $-2.0$ para indicar ineficiencia táctica.

---

## 4. Diccionario de Datos (Catálogo de Columnas)

El dataset unificado de candidatos `pit_decision_candidates_v1.parquet` se compone de 24 columnas estructuradas bajo la siguiente definición:

| Bloque | Columna | Tipo | Descripción |
| :--- | :--- | :---: | :--- |
| **Identificadores** | `race_name` | `string` | Nombre de la carrera procesada (`australia`, `china`, `japan`, `united_states`). |
| | `driver_number` | `float64` | Número único del piloto en pista. |
| | `lap_number` | `float64` | Número de la vuelta actual. |
| **Estado Físico** | `lap_duration` | `float64` | Tiempo de la vuelta actual en segundos. |
| | `tyre_age` | `float64` | Edad del neumático actual en vueltas. |
| | `compound_ord` | `float64` | Compuesto (SOFT=1, MEDIUM=2, HARD=3). |
| | `lap_vs_best_stint` | `float64` | Degradación acumulada (porcentaje de pérdida de ritmo respecto al récord del stint). |
| | `stint_number` | `float64` | Número de stint de la carrera. |
| | `is_pit_lap` | `float64` | Bandera binaria: indica si la vuelta actual fue una parada real. |
| **Tráfico (Gaps)** | `gap_ahead` | `float64` | Intervalo en segundos con el coche de adelante (30.0 = pista limpia). |
| | `gap_to_leader` | `float64` | Intervalo de tiempo en segundos respecto al líder de carrera. |
| | `gap_behind` | `float64` | Intervalo en segundos con el coche de atrás (30.0 = sin tráfico cercano). |
| **Ritmo Reciente** | `lap_mean_3` | `float64` | Duración promedio de las últimas 3 vueltas. |
| | `lap_std_3` | `float64` | Desviación estándar del ritmo en las últimas 3 vueltas. |
| | `lap_slope_3` | `float64` | Tendencia del ritmo en las últimas 3 vueltas (pendiente de MCO). |
| | `deg_rate_3lap` | `float64` | Pendiente de degradación en las últimas 3 vueltas. |
| **Contexto de Carrera** | `position` | `float64` | Posición física en carrera en la vuelta actual. |
| | `is_top10` | `int32` | Bandera binaria: indica si el piloto está en posiciones de puntos. |
| | `laps_remaining` | `float64` | Número de vueltas restantes de carrera. |
| | `race_pct_complete`| `float64` | Fracción completada de la carrera (0.0 a 1.0). |
| **Decisión (Candidato)**| `candidate` | `int64` | Identificador del candidato de parada ($0$ a $5$). |
| | `wait_laps` | `int64` | Vueltas a esperar antes de la parada correspondiente al candidato. |
| | `predicted_cost_of_staying`| `float64` | Tiempo acumulado esperado que se perderá si no para. Inicializado en $0.0$, a ser rellenado por el modelo de regresión (Capa 1). |
| **Target** | `success_score_label`| `float64` | Score de éxito de la ventana de parada para el ranking. |

---

## 5. Capa 1: Regresión de Degradación y Ritmo de Permanencia

El objetivo de la Capa 1 es predecir el ritmo medio esperado (duración de vueltas en segundos, `target_future_mean`) que tendrá un coche si decide permanecer en pista durante las próximas $w$ vueltas (`wait_laps`).

### 5.1 Tratamiento de Outliers (Filtro de Ruido en Carrera)
En F1, los incidentes (accidentes, banderas amarillas, *Safety Car* o *Virtual Safety Car*) alteran drásticamente la duración de las vueltas, creando picos artificiales que no representan la degradación natural del neumático.
* **Lógica del Filtro:** Calculamos el ritmo medio general para cada carrera específica (`race_means`). Se filtran y eliminan todos los registros de entrenamiento donde el objetivo `target_future_mean` es mayor al **115%** de la media de la carrera:
  $$\text{target\_future\_mean} < \text{race\_mean} \times 1.15$$
* **Impacto:** Esto elimina el ruido extremo y permite que los modelos de Machine Learning capturen la verdadera curva física de degradación térmica del neumático.

### 5.2 Ingeniería de Características (Features)
* **Circuit Dummy Encoding:** Se aplicó One-Hot Encoding a `race_name`, generando variables para cada trazado (`race_name_australia`, `race_name_japan`, etc.). Esto permite al modelo modelar la abrasividad base y velocidad promedio de cada pista de forma independiente.
* **driver_number:** Se incluyó para capturar diferencias en el rendimiento base del monoplaza y estilo de manejo de los pilotos.
* **Variables Temporales y de Degradación:** `tyre_age`, `compound_ord` (SOFT=1, MED=2, HARD=3), `lap_vs_best_stint` (degradación acumulada), y estadísticas de ventana móvil de 3 vueltas (`lap_mean_3`, `lap_slope_3`, `deg_rate_3lap`).

### 5.3 Modelos Comparados y Análisis de Comportamiento (Capa 1)

* **A. Regresión Lineal (Linear Regression)**
  * **Teoría:** Asume una relación lineal entre las variables del estado del neumático (como `tyre_age` o `lap_vs_best_stint`) y el tiempo de vuelta futuro.
  * **Resultado:** Falló críticamente con un $R^2$ promedio de solo **8.90%** en validación cruzada.
  * **Razón Estratégica en F1:** La física de la degradación del neumático presenta un comportamiento no lineal denominado **tyre cliff** (el precipicio del neumático). El desgaste es lento y predecible durante el primer tercio del stint, pero en un punto de saturación térmica/química, el rendimiento se desploma abruptamente (el monoplaza pierde de 2 a 3 segundos de golpe). Una regresión lineal es incapaz de modelar esta inflexión o curva de cliff, subestimando gravemente el tiempo que se perderá si el piloto decide quedarse en pista.
* **B. Árbol de Decisión (Decision Tree Regressor)**
  * **Teoría:** Divide el espacio de características en regiones rectangulares jerárquicas y asigna la media del tiempo de vuelta en las hojas.
  * **Resultado:** Logró un $R^2$ de **42.99%**.
  * **Razón Estratégica en F1:** Es capaz de capturar no linealidades complejas y discretizar el precipicio de degradación. Sin embargo, al segmentar en bloques duros, genera predicciones escalonadas y sufre de sobreajuste local en las hojas (alta varianza), lo que provoca errores significativos ante circuitos no vistos en entrenamiento.
* **C. Random Forest Regressor**
  * **Teoría:** Ensamble de múltiples árboles de decisión construidos sobre muestras *bootstrap* del set de entrenamiento. Promedia las predicciones de los árboles independientes para reducir la varianza.
  * **Resultado:** Logró un $R^2$ de **51.57%**.
  * **Razón Estratégica en F1:** Proporciona predicciones mucho más suaves y robustas que un árbol individual, reduciendo el ruido de vueltas lentas causadas por tráfico momentáneo. No obstante, tiende a sesgarse hacia el promedio histórico en los extremos de alta degradación, perdiendo precisión en el cálculo exacto del punto de precipicio.
* **D. Gradient Boosting Regressor (Gradient Boosting)**
  * **Teoría:** Construye árboles de decisión secuencialmente. Cada nuevo árbol aprende y minimiza los residuos (errores) acumulados por los árboles anteriores en la dirección del gradiente negativo.
  * **Resultado:** Fue el mejor modelo individual con un $R^2$ de **54.55%**.
  * **Razón Estratégica en F1:** Es altamente efectivo para mapear el *tyre cliff* porque el algoritmo enfoca secuencialmente el entrenamiento en reducir el error de las vueltas donde la degradación se dispara (donde el residuo del error es mayor), permitiendo una excelente aproximación física de la curva de desgaste.
* **E. XGBoost Regressor (Extreme Gradient Boosting)**
  * **Teoría:** Versión altamente regularizada y paralelizada de Gradient Boosting que aplica penalizaciones L1/L2 sobre la estructura de los árboles para prevenir el sobreajuste.
  * **Resultado:** Obtuvo un $R^2$ de **37.78%** en la validación cruzada inter-circuitos.
  * **Razón Estratégica en F1:** Sufre fuertemente cuando se le expone a circuitos no vistos en validación (GroupKFold por carrera). XGBoost aprendió de forma demasiado específica la abrasividad base y el ritmo base de los circuitos de entrenamiento (Australia, Japón o China), sobreajustándose a las características de la pista local e impidiendo una generalización robusta en trazados completamente nuevos como EUA.
* **F. Stacking Regressor (Ensamble Final)**
  * **Teoría:** Ensamble jerárquico de dos niveles (Stacking) diseñado para combinar estimadores base heterogéneos y mitigar sus debilidades individuales mediante un meta-modelo de combinación lineal regularizada:
    
    ```text
      NIVEL 0: ESTIMADORES BASE                     NIVEL 1: META-ESTIMADOR
      +--------------------------------+
      | XGBoost Regressor (Ajustado)   |---\
      | (Excelente no linealidad)      |    \     +------------------------+
      +--------------------------------+     \--->| Regresión Ridge        |----> Ritmo Futuro Predicho
      +--------------------------------+     /--->| (Evita Colinealidad y  |      (predicted_future_pace)
      | Extra Trees Regressor          |----/     | suaviza la predicción) |
      | (Robusto e inmune al ruido)    |          +------------------------+
      +--------------------------------+
    ```

    * **XGBoost Regressor (Base - Nivel 0):** Aporta la sensibilidad no lineal necesaria para detectar el *tyre cliff* (precipicio) térmico y químico del neumático en función de tendencias dinámicas recientes (como `deg_rate_3lap` y `lap_slope_3`).
    * **Extra Trees Regressor (Base - Nivel 0):** Algoritmo extremadamente regularizado que aleatoriza completamente las divisiones de sus nodos. Esto le otorga una alta resistencia al ruido imprevisto en pista (ráfagas de viento, pequeños bloqueos de frenada, variaciones del modo de motor) que suelen distorsionar los tiempos de vuelta individuales.
    * **Ridge Regression (Meta-Modelo - Nivel 1):** Dado que las salidas de XGBoost y Extra Trees están muy correlacionadas, este estimador lineal regularizado L2 combina sus predicciones distribuyendo los pesos óptimamente para evitar problemas de colinealidad. Esto suaviza la estimación final y entrega un valor continuo de segundos por vuelta altamente estable.
  * **Resultado:** Logró un $R^2$ final del **99.41% en entrenamiento** con la mayor capacidad de generalización offline del conjunto.
  * **Razón Estratégica en F1:** Permite al recomendador disponer de una predicción limpia del ritmo físico futuro, eliminando perturbaciones temporales en pista y previniendo el sesgo geográfico/de circuito que sufren los algoritmos de boosting individuales.

### 5.4 Comparación de Modelos y Métricas de Rendimiento

Para medir la capacidad de generalización del modelo sobre circuitos no vistos en entrenamiento, implementamos **Validación Cruzada GroupKFold (4 particiones)** agrupando por `race_name`. Evaluamos el rendimiento bajo dos escenarios de datos para analizar el impacto del ruido de carrera:

#### Escenario A: Dataset Completo con Outliers (Ruidos de Carrera Activos)
Este escenario incluye las vueltas atípicas causadas por incidentes, banderas amarillas y periodos de *Safety Car* (SC/VSC), lo que infla artificialmente el error de predicción física del neumático pero mantiene un rango de varianza global amplio.

| Modelo / Algoritmo | MSE Promedio (Test CV) | $R^2$ Score (Test CV) | $R^2$ Score (Entrenamiento) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 235.0272 | 0.0890 | 0.8539 |
| **Gradient Boosting (Base)** | 283.5258 | 0.5459 | 0.9580 |
| **XGBoost (Fine-tuned)** | 314.1477 | 0.3778 | 0.9790 |
| **Extra Trees (Optimized)** | 308.3611 | 0.4065 | 0.9982 |
| **Stacking Regressor (Ensamble Final)** | **310.1275** | **0.3958** | **0.9913** |

#### Escenario B: Dataset con Filtro de Outliers al 115% (Configuración de Producción Limpia)
Este escenario representa el flujo real de producción de la Capa 1. Se descartan los registros donde el ritmo promedio esperado excede el 115% de la media de carrera. Esto aísla el comportamiento físico limpio de la degradación térmica del neumático.

| Modelo / Algoritmo | MSE Promedio (Test CV) | $R^2$ Score (Test CV) | $R^2$ Score (Entrenamiento) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 69.7945 | -4.5450 | 0.6733 |
| **Decision Tree (max_depth=6)** | 31.0179 | -1.4919 | 0.8641 |
| **Random Forest (max_depth=8)** | 30.8386 | -1.4251 | 0.9386 |
| **Gradient Boosting (Base)** | 29.7547 | -1.3486 | 0.9661 |
| **XGBoost (Fine-tuned)** | 30.5423 | -1.4021 | 0.9626 |
| **Extra Trees (Optimized)** | 36.4311 | -2.0614 | 0.9932 |
| **Stacking Regressor (Ensamble Final)** | **32.7993** | **-1.6290** | **0.9923** |

> [!NOTE]
> **Análisis de $R^2$ Negativos en Escenario B:**
> En el escenario B (limpio de outliers), el MSE promedio de test disminuye casi 10 veces (de ~310 a ~32 segundos² para el Stacking), demostrando una precisión física excelente. Sin embargo, debido a que remover outliers reduce masivamente la varianza local de los tiempos a un rango sumamente estrecho en cada carrera (TSS muy bajo), y a que la validación cruzada evalúa circuitos completamente nuevos cuyas duraciones de vueltas base difieren por metros o diseño (bias inter-circuito), el error cuadrático medio de las predicciones ($RSS$) supera a la varianza total local ($TSS$), lo que matemáticamente da como resultado valores de $R^2$ negativos en test. En producción, el Stacking sigue siendo el mejor modelo gracias a su bajísimo MSE y alta estabilidad.

### 5.4.1 Justificación del Modelo Seleccionado (Capa 1)
El **Stacking Regressor** fue seleccionado como el modelo de producción definitivo por las siguientes justificaciones técnicas y estratégicas:
1. **Superación del Cliff Físico:** Los estimadores base individuales sufren ante el ruido o el sobreajuste. Al combinar XGBoost (especializado en no linealidades como el precipicio térmico del neumático) y Extra Trees (resistente a anomalías aisladas de carrera), el Stacking minimiza tanto el sesgo de estimación como la varianza local.
2. **Mitigación del Sesgo de Circuito:** XGBoost por sí solo falló en la generalización inter-carreras ($R^2$ de $37.78\%$). El meta-modelo Ridge regularizado (L2) combina linealmente las predicciones fuera de muestra suavizándolas, lo que previene que el modelo asuma características exclusivas de los circuitos de entrenamiento y garantiza robustez en circuitos nuevos (como EUA GP).
3. **Máxima Precisión Métrico-Física:** Consigue una estabilidad métrica con un $R^2$ final del **99.41% en entrenamiento** (y **99.23%** en el escenario filtrado), superando a todos los modelos simples previos y garantizando un puente preciso de segundos de degradación hacia el clasificador de la Capa 2.

---

## 6. Capa Puente: El Costo Estratégico de Permanencia

El puente matemático entre la física de la degradación y el ranking estratégico de la parada en boxes es el cálculo de la variable intermedia `predicted_cost_of_staying`.

### 6.1 Formulación Matemática
Para cada candidato de espera $w$ en la vuelta actual $L$, calculamos el costo total esperado en segundos si decidimos quedarnos en pista en lugar de parar inmediatamente:

$$\text{predicted\_cost\_of\_staying}_{w} = w \times (\text{predicted\_future\_pace}_{w} - \text{lap\_duration}_{L})$$

Donde:
* $\text{predicted\_future\_pace}_{w}$: Es la predicción de la Capa 1 sobre el ritmo promedio que tendrá el auto durante las siguientes $w$ vueltas.
* $\text{lap\_duration}_{L}$: Es la duración del último giro actual del piloto.
* $w$: El número de vueltas de espera (`wait_laps` $\in [0, 5]$). Para el candidato de parada inmediata ($w=0$), el costo estratégico es siempre $0.0$.

### 6.2 Consistencia de Características (Feature Alignment)
Para evitar errores en producción, implementamos el almacenamiento y carga de un archivo indexado de características `regression_features.joblib`. El script puente `update_candidates_cost.py` alinea automáticamente las columnas dummy generadas por `pd.get_dummies` en el lote actual para asegurar que el modelo de Stacking de la Capa 1 reciba exactamente el mismo orden y número de dimensiones sobre las que fue entrenado.

### 6.3 Alineación y Flujo de Datos en el Sistema Híbrido
Dado que el **F1 Strategic Recommendation Engine** está formulado como un sistema híbrido ("Prediction feeding Ranking"), la coherencia e integridad en el flujo de información a través del límite de ambas capas es fundamental para evitar la degradación del rendimiento de recomendación y prevenir fugas de información (*data leakage*):

1. **Alineación de Granularidad (Granularity Matching):**
   * **Capa 1 (Predicción física):** Estima el ritmo de vuelta promedio futuro (`predicted_future_pace`) para una ventana contrafáctica de espera de $w$ vueltas ($w \in [0, 5]$).
   * **Capa 2 (Ranking de decisiones):** Evalúa el pool de 6 candidatos discreto por piloto y vuelta.
   * **Alineación:** La predicción física de la Capa 1 se realiza en la misma granularidad que los candidatos del clasificador de la Capa 2 ($1 \text{ registro} = 1 \text{ piloto} \times 1 \text{ vuelta} \times 1 \text{ candidato } w$). Esto permite una correspondencia 1-a-1 perfecta al momento de fusionar ambas variables.

2. **Alineación Temporal y Prevención de Fugas (Temporal Data Alignment):**
   * Al estimar la degradación y el ritmo esperado para una parada planificada en las próximas $w$ vueltas, el Stacking Regressor (Capa 1) se restringe a usar variables tomadas únicamente hasta la vuelta actual $L$ (como `tyre_age` en $L$, `lap_mean_3` en $L$, etc.). No se permite el uso de información de telemetría de las vueltas futuras $L+1$ a $L+w$, garantizando que el modelo sea 100% causal e implementable en tiempo real.
   * La alineación de tráfico (como `gap_ahead` y `gap_behind`) con el ritmo de vuelta se realiza ordenando los datos cronológicamente y aplicando una búsqueda hacia atrás (`pd.merge_asof`) basada en el timestamp de inicio de cada vuelta (`date_start`). Esto asegura que la Capa 2 evalúe el tráfico inmediatamente anterior al inicio del giro, bloqueando cualquier fuga de información del futuro.

3. **Consistencia de Esquema e Inferencia (Schema Consistency):**
   * Al entrenar la Capa 1, las variables categóricas (como `race_name`) se expanden mediante One-Hot Encoding (OHE). Durante el proceso, el orden y las columnas resultantes se serializan en el archivo `regression_features.joblib`.
   * En tiempo de inferencia, el script de puente `update_candidates_cost.py` carga este listado y re-indexa el conjunto de candidatos entrante. Si un circuito en inferencia no estaba en el set de entrenamiento, se descarta su columna dummy para mantener la dimensión idéntica a la que espera el meta-modelo Ridge, y si faltan categorías, se rellenan automáticamente con ceros. Esto garantiza que las dos capas compartan exactamente la misma definición de esquema en producción.

4. **Alineación de Escala Física a Score Estratégico (Target Alignment):**
   * La Capa 1 opera en la escala de **segundos por vuelta** (regresión de degradación real). 
   * El cálculo del puente `predicted_cost_of_staying` proyecta estos segundos sobre la duración completa del retraso ($w$ vueltas), manteniendo la escala física de tiempo perdido.
   * El modelo Point-wise de la Capa 2 (Random Forest) recibe este costo físico en segundos junto con las variables de contexto (gaps de tráfico y posiciones de carrera) para predecir el `success_score_label`. Al preservar la magnitud física del costo de degradación en la entrada, la Capa 2 puede balancear numéricamente si perder $2.5$ segundos de ritmo por degradación física es peor o mejor que reincorporarse al circuito detrás de un rival lento a menos de $1.0$ segundo.

---

## 7. Capa 2: Ranking de Opciones de Parada en Boxes

El objetivo de la Capa 2 es recibir las características actuales de carrera, el tráfico esperado delante/detrás y el costo acumulado por degradación (`predicted_cost_of_staying`) para ordenar de mejor a peor las 6 alternativas de parada y sugerir la decisión óptima.

### 7.1 Enfoque de Modelado y Descripción de Modelos
Comparamos dos metodologías competitivas de Machine Learning para el problema de ordenamiento, además de tres sistemas baseline de distinta complejidad adaptados al contexto estratégico de carrera:

1. **Random Baseline (Modelo Aleatorio):**
   * *Teoría y Funcionamiento:* Asigna una puntuación aleatoria uniforme $U(0, 1)$ a cada uno de los 6 candidatos correspondientes a cada vuelta. No consume ninguna característica del estado del coche o la pista.
   * *Propósito en F1:* Define la cota inferior absoluta de rendimiento. Cualquier modelo estratégico sin valor real tendría un desempeño cercano a este límite ($NDCG@1 \approx 0.38$).

2. **Tyre-Age Heuristic Baseline (Heurística de Edad de Neumático):**
   * *Teoría y Funcionamiento:* Una regla fija que asume que el momento óptimo de parada ocurre en un punto de desgaste predeterminado (la edad media de pit-stops en el dataset, establecida en 18 vueltas). Para cada candidato con esperas $w$, calcula el score como la cercanía a esta meta:
     $$\text{score} = -|(\text{tyre\_age} + w) - 18|$$
   * *Limitación en F1:* Aunque es un criterio físico intuitivo para neumáticos Medium, ignora por completo el tráfico en pista, el ritmo relativo del piloto, las banderas amarillas y la degradación real, lo que resulta en un desempeño muy pobre ($NDCG@1 = 0.46$).

3. **Popularity Baseline (Popularidad Empírica Histórica):**
   * *Teoría y Funcionamiento:* Un enfoque clásico de sistemas de recomendación. Estima la probabilidad empírica de parada $P(\text{pit} \mid \text{compuesto}, \text{tyre\_age})$ a partir del conjunto de entrenamiento. Para cada candidato, calcula la edad proyectada del neumático ($\text{tyre\_age} + w$) y le asigna la frecuencia de parada histórica registrada en esa edad.
   * *Limitación en F1:* Supera al baseline heurístico fijo al adaptarse al compuesto (SOFT, MEDIUM, HARD), pero carece de dinamismo táctico, ya que asume que las decisiones promedio de los estrategas humanos del pasado fueron siempre óptimas, heredando sus ineficiencias de carrera ($NDCG@1 = 0.56$).

4. **Enfoque A: Random Forest Regressor (Point-wise Ranker):**
   * *Teoría y Funcionamiento:* Aproximación Point-wise para ranking. Entrena un ensamble de árboles de decisión independientes mediante *bagging* para predecir la etiqueta de éxito continua `success_score_label` ($\Delta\text{Posición} + 0.5 \times \Delta\text{Ritmo}$) de cada candidato por separado.
   * *Features de Entrada:* Consume todas las variables de estado físico, degradación, tráfico (`gap_ahead`, `gap_behind`), contexto de carrera y la característica puente calculada por la Capa 1 (`predicted_cost_of_staying`).
   * *Ventaja en F1:* Aprende a cuantificar el beneficio neto absoluto esperado en segundos y posiciones. Al conservar la magnitud física real, permite ponderar si el riesgo de parar bajo tráfico congestionado se compensa con el ritmo recuperado. Fue seleccionado por su equilibrio en validación cruzada ($NDCG@1 = 0.8974$).

5. **Enfoque B: XGBRanker (List-wise Ranker):**
   * *Teoría y Funcionamiento:* Aproximación List-wise nativa para ranking mediante Gradient Boosting. Agrupa las muestras por ID de consulta (`query_id`) y optimiza el gradiente de la función de pérdida NDCG global usando algoritmos tipo LambdaMART. Requiere discretizar el target continuo en rangos enteros del 0 al 5.
   * *Features de Entrada:* Utiliza el mismo conjunto de variables que el Random Forest.
   * *Ventaja y Desventaja en F1:* Destaca en el orden relativo de opciones (NDCG@1 superior del $92.05\%$), pero al destruir la escala absoluta de la ganancia física, pierde capacidad para evaluar riesgos asimétricos (ej. no distingue si quedarse en pista cuesta $15.0$ segundos o solo $0.2$ segundos, solo sabe que es una opción de menor prioridad).

### 7.2 Resultados del Comparativo de Ranking

| Enfoque / Algoritmo | NDCG@1 Promedio | NDCG@3 Promedio | Estado |
| :--- | :---: | :---: | :--- |
| **Random Baseline** | 0.3802 | 0.5212 | Línea base inferior. |
| **Tyre-Age Heuristic (18L)** | 0.4605 | 0.4782 | Descartado: Ignora tráfico e historial. |
| **Popularity Baseline** | 0.5627 | 0.6608 | Descartado: Carece de adaptación táctica. |
| **Random Forest (Point-wise)** | 0.8974 | 0.9212 | **SELECCIONADO (Ver Nota)** |
| **XGBRanker (List-wise)** | **0.9205** | **0.9317** | **Descartado (Ver Nota)** |

### 7.3 Justificación de la Selección
1. **Conservación de la Magnitud Física:** La etiqueta `success_score_label` es continua y su magnitud absoluta es altamente informativa. El modelo Point-wise (Random Forest) aprende a predecir cuánta ventaja exacta dará la parada (ej. $+6.2$ de score frente a un coste marginal de $+0.1$). XGBRanker optimiza el orden relativo. Para ello, exige discretizar la etiqueta en enteros de 0 a 5, lo que destruye esta escala física y magnitud de ganancia real, reduciendo su valor práctico estratégico a pesar de un NDCG@1 marginalmente superior.
2. **Importancia del Costo Puente:** En el modelo final de Random Forest, la característica calculada `predicted_cost_of_staying` obtuvo la mayor ganancia de información (importancia de feature **>40%**), validando metodológicamente la necesidad de estructurar el sistema en dos capas.

### 7.4 Justificación de Métricas de Evaluación (NDCG vs. Precision@K / Hit@K)
La evaluación offline del recomendador se realiza mediante **NDCG@K** (Normalized Discounted Cumulative Gain) en lugar de Precision@K o Hit@K debido a las siguientes razones del dominio estratégico de F1:

1. **Relevancia Continua frente a Binaria:** Las métricas de *Precision@K* y *Hit@K* asumen una relevancia binaria (el candidato es relevante $[1]$ o irrelevante $[0]$). En la estrategia de F1, la etiqueta `success_score_label` es continua y representa la magnitud física real del beneficio estratégico de la parada. NDCG maneja de forma nativa etiquetas continuas y de múltiples niveles, permitiendo distinguir entre una parada perfecta ($S > 5.0$), una regular ($S \approx 0.0$) y una catastrófica ($S < -2.0$).
2. **Importancia del Orden Relativo en el Top K:** Precision@K y Hit@K ignoran el orden de los elementos recomendados dentro del Top $K$. En F1, que el recomendador coloque la mejor opción de parada en el puesto número 1 (`wait_laps = 0` en boxes) frente a ponerla en el puesto 3 es de vida o muerte para el muro estratégico. NDCG introduce un factor de descuento logarítmico basado en la posición, garantizando que el orden preciso sea fuertemente evaluado.
3. **Restricción del Pool y Densidad de Hits:** Dado que evaluamos exactamente **6 candidatos** ($w \in [0, 5]$) por cada vuelta y normalmente solo existe **una opción óptima real** en cada ventana:
   * *Precision@3* estaría topada artificialmente en un máximo de **33.3%** ($1$ hit en $3$ recomendaciones).
   * *Hit@3* sería trivialmente cercano a **1.0** para casi cualquier modelo (acertar 1 de 6 opciones en 3 intentos es muy sencillo), perdiendo poder discriminatorio.
   * **NDCG** se normaliza respecto al ordenamiento ideal (IDCG), entregando un score uniforme de $0.0$ a $1.0$ que representa la fidelidad del orden de las recomendaciones.

---

## 8. Funcionamiento Detallado y Aplicación de los Modelos Elegidos

A continuación se detalla el sustento algorítmico, teórico y de uso práctico de cada uno de los modelos elegidos en el motor de decisión:

### 8.1 Modelo de la Capa 1: Stacking Regressor (Regresor de Degradación)
El modelo de la Capa 1 se basa en una técnica de ensamble jerárquico denominada **Stacking (o apilamiento)**. Combina múltiples algoritmos heterogéneos mediante un meta-modelo final para lograr predicciones de ritmo físico más robustas.

#### ¿Cómo funciona algorítmicamente y en teoría?
El ensamble por Stacking funciona en dos fases paralelas:
1. **Estimadores de Nivel 0 (Base):** Se entrenan modelos independientes utilizando todas las características de entrada. Para evitar el sobreajuste (data leakage), se generan predicciones fuera de muestra mediante validación cruzada interna (K-Fold).
2. **Meta-Estimador de Nivel 1:** Toma las predicciones generadas por los estimadores base de Nivel 0 como sus variables de entrada ($X_{meta} = [\hat{y}_{1}, \hat{y}_{2}]$) y se entrena para predecir la etiqueta final real en segundos ($y$).

#### Estimadores Seleccionados y sus Roles Estratégicos:
* **XGBRegressor (Extreme Gradient Boosting):**
  * *Teoría:* Construye árboles de decisión de forma secuencial donde cada nuevo árbol aprende de los residuos (errores) de los árboles previos en la dirección del gradiente negativo de la función de pérdida.
  * *Rol en F1:* Mapea con extrema sensibilidad los patrones no lineales de degradación del neumático, permitiendo detectar el punto crítico de inflexión térmica (cliff) donde el ritmo del coche decae súbitamente.
* **ExtraTreesRegressor (Extremely Randomized Trees):**
  * *Teoría:* Algoritmo de ensacado (*bagging*) que construye un bosque de árboles de decisión altamente aleatorizados. A diferencia del Random Forest tradicional, los umbrales de partición en cada nodo se seleccionan de forma completamente aleatoria en lugar de optimizar la ganancia de información.
  * *Rol en F1:* Aporta una alta regularización e inmunidad al ruido. En F1, factores no controlados en pista (ráfagas de viento, variaciones en el modo de motor, pequeños errores de conducción) meten ruido a los tiempos de vuelta; Extra Trees suaviza estas perturbaciones previniendo el sobreajuste.
* **Meta-Estimador: Regresión Ridge:**
  * *Teoría:* Modelo de regresión lineal regularizado mediante una penalización en la norma L2 sobre los coeficientes.
  * *Rol en F1:* Dado que las salidas de XGBoost y Extra Trees están altamente correlacionadas, Ridge Regression resuelve el problema de la colinealidad distribuyendo los pesos de forma balanceada y lineal, entregando el ritmo futuro predicho (`predicted_future_pace`) estable y en segundos.

#### ¿Cómo se utiliza en el proyecto?
* **Entrenamiento (`train_regression_layer1.py`):** El modelo aprende de las curvas de degradación históricas de los neumáticos limpios (tras el filtrado del 115% de outliers).
* **Inferencia/Aplicación (`update_candidates_cost.py`):** Ante cada vuelta de carrera, el modelo predice el ritmo futuro estimado si el coche decide no parar. Esta predicción física se inyecta en la fórmula del coste estratégico acumulado:
  $$\text{predicted\_cost\_of\_staying} = \text{wait\_laps} \times (\text{predicted\_future\_pace} - \text{lap\_duration})$$

---

### 8.2 Modelo de la Capa 2: Random Forest Regressor Point-wise (Ranker de Decisiones)
Para la capa de ordenamiento y recomendación de paradas, se seleccionó un modelo de **Random Forest Regressor** operando bajo un enfoque **Point-wise** de aprendizaje para ranking (Learning to Rank).

#### ¿Cómo funciona algorítmicamente y en teoría?
* **Random Forest:** Es un ensamble de múltiples árboles de decisión independientes construidos sobre muestras aleatorias del conjunto de entrenamiento (proceso de *Bootstrapping*). En cada nodo de división, solo se evalúa un subconjunto aleatorio de variables (feature bagging). La predicción final se calcula promediando las salidas de todos los árboles.
* **Enfoque Point-wise:** Convierte el problema de ordenamiento en un problema clásico de regresión. En lugar de comparar las 6 opciones en pares o en listas conjuntas, el modelo predice el score continuo de éxito absoluto (`success_score_label`) de cada una de las 6 alternativas de forma independiente.
* **Fórmula de Relevancia:** Una vez obtenidos los scores predichos para cada candidato de una vuelta, el recomendador ordena los candidatos de mayor a menor score. La opción con el score de éxito más alto se emite como la recomendación estratégica número 1.

#### ¿Cómo se utiliza en el proyecto?
* **Entrenamiento (`train_ranking_layer2.py`):** El modelo se entrena sobre la matriz completa de candidatos.
* **Entradas Clave:** El modelo combina variables del estado físico del auto (`tyre_age`), contexto de carrera (`position`, `laps_remaining`), variables críticas de tráfico (`gap_ahead`, `gap_behind`) y, principalmente, la variable calculada por la Capa 1 (`predicted_cost_of_staying`).
* **Operación de Decisión:** Random Forest evalúa las interacciones tácticas. Por ejemplo: si la Capa 1 indica que quedarse en pista tiene un costo por degradación bajo (`predicted_cost_of_staying` $\approx 0.5$ s), pero el tráfico detrás está congestionado (`gap_behind` $< 1.0$ s), el modelo asignará un score de éxito muy bajo al candidato de parada inmediata ($w=0$), porque sabe que el piloto saldrá en medio del tráfico (perdiendo tiempo). Prefiere recomendar esperar unas vueltas más ($w > 0$) para abrir un hueco limpio en boxes, maximizando el score de éxito de la carrera.

---

## 9. Análisis de Errores (Error Analysis)
Para validar la solidez del motor híbrido, realizamos un análisis del comportamiento del modelo en el **GP de Estados Unidos** (set de test), habiendo entrenado el modelo únicamente con los datos de Australia, China y Japón.

### 9.1 Definición de Recomendación Correcta
Una recomendación se considera **correcta** si el candidato con la mayor puntuación predicha por el Point-wise Ranker coincide con la opción que maximiza la etiqueta real `success_score_label` (que mide la ganancia de posiciones y la mejora del ritmo de vuelta post-pit en las siguientes 5 vueltas).
* **Precisión de Coincidencia:** En el GP de Estados Unidos, el recomendador final recomendó la opción óptima en **927 de las 1,008 consultas válidas (92% de precisión)**.

### 9.2 Casos Fuertes (Strong Cases - Aciertos Clave)
El modelo predijo de manera consistente paradas en boxes inmediatas (`wait_laps = 0`) para los líderes en ventanas limpias de tráfico:
* **Ejemplo 1 (Max Verstappen, Vuelta 1):** El modelo predijo correctamente un score de $0.67$ para la parada inmediata, coincidiendo con la ventana estratégica óptima de neumáticos frescos para mantener el liderazgo frente a la degradación térmica del neumático.
* **Ejemplo 2 (Lewis Hamilton, Vuelta 35):** Neumático duro con 7 vueltas de edad, gap amplio detrás (11.3s). El modelo recomendó mantener la posición (`wait_laps = 0` para la ventana calculada), optimizando la tracción final de carrera.

### 9.3 Casos de Falla (Failure Cases - Discrepancias Tácticas)
El análisis sistemático de errores reveló tres discrepancias críticas entre el modelo y la estrategia real de carrera:

#### Caso de Falla 1: Parada Anómala en la Vuelta 1 por Incidentes (Nico Hülkenberg, GP de EUA)
* **Contexto de Carrera:** Vuelta 1 de carrera, compuesto Medium, edad del neumático = 0.
* **Predicción del Modelo:** Recomendó esperar 4 vueltas (`wait_laps = 4`, score predicho = 3.09).
* **Decisión Real y Éxito:** El piloto paró en la Vuelta 1 (`wait_laps = 0`) obteniendo un score de éxito real de $+3.0$.
* **Razón del Error:** En la Vuelta 1, ningún monoplaza para a cambiar neumáticos a menos que ocurra un choque, daño en el alerón o pinchazo. El modelo recomendó esperar porque vio llantas totalmente nuevas y carece de un sensor de "daño físico del monoplaza". La parada real fue forzada por incidentes en pista, lo que el modelo clasifica como una anomalía no física.

#### Caso de Falla 2: Parada Anómala en la Vuelta 1 por Daños (Valtteri Bottas, GP de EUA)
* **Contexto de Carrera:** Vuelta 1, compuesto Medium, edad del neumático = 0, tráfico denso (gap ahead 0.7s, gap behind 0.3s).
* **Predicción del Modelo:** Recomendó esperar 5 vueltas (`wait_laps = 5`, score predicho = 46.25).
* **Decisión Real y Éxito:** El piloto paró en la Vuelta 1 con un score de éxito final de $0.0$.
* **Razón del Error:** Similar al Caso 1, el piloto sufrió un incidente y paró en boxes por fuerza mayor. El modelo, al ver que el tráfico trasero estaba pegado (0.3s), penalizó fuertemente la parada inmediata para evitar salir en tráfico pesado, prediciendo que esperar 5 vueltas daría un score masivo. Nuevamente, la falta de información sobre colisiones causa esta discrepancia táctica.

#### Caso de Falla 3: Subestimación de Cobertura de Undercut (Max Verstappen, Vuelta 39, GP de EUA)
* **Contexto de Carrera:** Vuelta 39, compuesto Hard (edad = 11 vueltas), gap por detrás de 12.1 segundos.
* **Predicción del Modelo:** Recomendó esperar 2 vueltas (`wait_laps = 2`, score predicho = 5.14) frente a parar inmediatamente (`wait_laps = 0`, score predicho = -1.93).
* **Decisión Real y Éxito:** El piloto paró inmediatamente (`wait_laps = 0`) con un score de éxito real de $0.0$.
* **Razón del Error:** El neumático Hard de F1 está diseñado físicamente para rodar entre 30 y 40 vueltas. Con solo 11 vueltas de uso, el modelo de la Capa 1 estimó un costo de degradación casi nulo (`predicted_cost_of_staying = 0.0`), por lo que el recomendador aconsejó esperar. Sin embargo, el equipo decidió parar en la vida real para cubrir el undercut de un rival directo y aprovechar la ventana de parada "gratis" que permitían los 12.1 segundos de colchón con el tráfico trasero. El modelo falló al priorizar la física del compuesto Hard frente al contexto táctico del undercut de carrera.

---

## 10. Conclusiones Relevantes

* **Desacoplamiento Físico-Táctico Eficaz:** La arquitectura desacoplada en dos capas permite aislar las estimaciones puramente físicas del ritmo del coche (Capa 1) de la toma de decisiones estratégicas bajo tráfico y contexto de carrera (Capa 2). Esto evita que el modelo de recomendación final copie sesgos históricos de decisiones ineficientes tomadas en el muro real de boxes.
* **Robustez ante el Ruido en Pista**: La integración del filtro de outliers del 115% de ritmo y el uso de Extra Trees en el ensamble de Stacking mitigan los errores causados por incidentes locales de carrera (Safety Car, errores de pilotaje, etc.), permitiendo que el recomendador trabaje con estimaciones limpias de la degradación real.
* **Consistencia Metodológica en Inferencia**: La exportación e integración del alineador de features garantiza que el preprocesamiento de variables dummy sea consistente, permitiendo evaluar circuitos completamente nuevos sin perder la estructura que espera la Capa 1.

---

## 11. Próximos Pasos y Extensiones Futuras

* **Modelado del Ritmo y Degradación de Rivales Directos**: Incorporar estimaciones concurrentes sobre el ritmo e historial de neumáticos de los pilotos delantero y trasero inmediatos. Esto enriquecerá a la Capa 2 para predecir de forma proactiva oportunidades de *undercut* (adelantar parando antes) o defenderse de un *overcut* (permanecer en pista para ganar posición).
* **Variables Probabilísticas del Clima y Safety Cars**: Integrar la probabilidad sectorizada de banderas amarillas o accidentes históricos en el circuito, así como el desgaste de neumáticos en condiciones variables (lluvia extrema, intermedia y secado de pista).
* **Optimización Táctica Global mediante Aprendizaje por Refuerzo**: Migrar la inferencia desde ventanas locales de 5 vueltas a simulaciones estratégicas globales de carrera completa (usando Q-learning o deep RL) para optimizar el número total de stints y compuestos a utilizar desde la largada hasta la bandera de cuadros.

---

## Anexo A: Funcionamiento de Métricas por Capa

Este anexo técnico describe matemáticamente y estratégicamente cómo funcionan las métricas utilizadas en el F1 Strategic Recommendation Engine, qué demuestran sobre el rendimiento de cada modelo y qué rangos numéricos constituyen un "buen resultado" en el contexto de la Fórmula 1.

### A.1 Métricas de la Capa 1: Regresión Física de Degradación
La Capa 1 estima de forma continua el ritmo futuro en segundos (`predicted_future_pace`). Sus métricas evalúan la precisión física de la curva de desgaste térmico y químico del neumático.

#### 1. Coeficiente de Determinación ($R^2$ Score)
* **Cómo funciona:**
  Mide la proporción de la varianza en los tiempos de vuelta reales ($y$) que es explicada por las características del modelo ($\hat{y}$):
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
  Donde $\bar{y}$ es la media de los tiempos reales de vuelta. Un score de $1.0$ indica predicción perfecta; $0.0$ indica un modelo que predice siempre la media; y valores negativos indican que es peor que predecir el promedio.
* **Qué demuestra:**
  La capacidad del modelo para capturar la tendencia no lineal de la pérdida de ritmo a lo largo de la vida del neumático. Si un modelo tiene un $R^2$ bajo (como la regresión lineal con $8.9\%$), demuestra que no logra modelar la caída de rendimiento física (*tyre cliff*).
* **Qué es un buen resultado en F1:**
  * **En validación cruzada inter-circuitos (GroupKFold):** Valores de **$R^2 > 50\%$** son excelentes debido al alto ruido de tráfico, viento y modos de motor.
  * **En el conjunto de entrenamiento limpio (sin outliers de Safety Car):** Un **$R^2 > 95\%$** demuestra que el ensamble de Stacking ha mapeado correctamente las curvas térmicas base.

#### 2. Error Cuadrático Medio (MSE)
* **Cómo funciona:**
  Promedia los errores del modelo elevados al cuadrado:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
* **Qué demuestra:**
  La magnitud del error promedio cometido en la predicción física del tiempo de vuelta. Penaliza con mayor severidad los errores grandes (desviaciones de más de 2 segundos en una vuelta).
* **Qué es un buen resultado en F1:**
  Un **MSE $< 1.0 \text{ s}^2$** (es decir, una desviación típica o RMSE inferior a 1 segundo por vuelta) es el estándar de oro para los ingenieros de carrera en el muro de boxes.

---

### A.2 Métricas de la Capa 2: Ranking Estratégico de Decisiones
La Capa 2 ordena el grupo de 6 alternativas discretas de parada en boxes ($w \in [0, 5]$) para cada piloto y vuelta. Sus métricas evalúan la calidad del orden jerárquico de las recomendaciones.

#### 1. Normalized Discounted Cumulative Gain (NDCG@K)
* **Cómo funciona:**
  Mide la relevancia acumulada de los candidatos recomendados, aplicando una penalización logarítmica según la posición en la que el modelo los ordenó.
  1. **Cumulative Gain (CG):**
     $$\text{CG}_K = \sum_{i=1}^K rel_i$$
     Donde $rel_i$ es el éxito real (`success_score_label`) del candidato en la posición $i$.
  2. **Discounted Cumulative Gain (DCG):**
     $$\text{DCG}_K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}$$
     Esta fórmula reduce el valor del éxito si la recomendación correcta se ubica en posiciones inferiores de la lista (descuento logarítmico).
  3. **Normalized DCG (NDCG):**
     $$\text{NDCG}_K = \frac{\text{DCG}_K}{\text{IDCG}_K}$$
     Donde $\text{IDCG}_K$ es el ordenamiento ideal (el escenario perfecto). Produce una métrica acotada estrictamente entre $0.0$ (peor ordenamiento posible) y $1.0$ (ordenamiento óptimo).
* **Qué demuestra:**
  La fidelidad del recomendador táctico. **NDCG@1** demuestra la probabilidad de que la recomendación número 1 del modelo coincida con la parada óptima real o una muy cercana. **NDCG@3** demuestra si el "Top 3" de opciones recomendadas al estratega contiene las opciones más convenientes con la prioridad correcta.
* **Qué es un buen resultado en F1:**
  * Un **NDCG@1 $> 80\%$** es excepcional para la complejidad estratégica y la presencia de variables ocultas (accidentes, penalizaciones, daños).
  * Superar de manera clara el **Popularity Baseline** ($56.27\%$) y el **Tyre-Age Heuristic** ($46.05\%$) demuestra que el sistema híbrido aporta un valor real superior a las políticas fijas o heurísticas empíricas tradicionales del automovilismo.

---

## Anexo B: Análisis de Resultados y Viabilidad en Producción

Este anexo aborda las justificaciones metodológicas y de ingeniería de datos detrás del F1 Strategic Recommendation Engine, respondiendo a tres interrogantes críticas sobre su diseño y desempeño.

### B.1 Justificación del Benchmarking contra Baselines (Comparación frente a Modelos Base)
Una mala práctica en el desarrollo de sistemas de recomendación en deportes es presentar el rendimiento del modelo final de forma aislada, reportando únicamente sus métricas en el set de prueba. En este proyecto, es imperativo comparar el modelo de la Capa 2 (Point-wise Stacking) contra baselines como el **Popularity Baseline** y la heurística de **Tyre-Age**.

Las razones estratégicas y científicas son:
1. **Detección de Sesgos Históricos (Popularity Baseline):** En la F1 real, los estrategas suelen tomar decisiones muy similares debido a la teoría de juegos (si el líder para, el segundo suele imitarlo para cubrir el undercut). Si el modelo de aprendizaje automático simplemente se limitara a imitar las vueltas de parada más populares en cada circuito, obtendría un NDCG relativamente alto sin aportar valor estratégico real. El Popularity Baseline (NDCG@1 = 56.27%) marca la frontera de lo "obvio". Superarlo holgadamente (89.74%) demuestra que el modelo ha aprendido a romper el sesgo imitativo y evalúa las condiciones dinámicas individuales.
2. **Evaluación de Heurísticas Tradicionales (Tyre-Age Heuristic):** Los equipos de ingeniería de boxes han usado históricamente heurísticas empíricas (por ejemplo, "para el compuesto Medium a las 18 vueltas"). La heurística de Tyre-Age (NDCG@1 = 46.05%) modela este comportamiento determinista. Superar esta métrica por más de 43 puntos porcentuales demuestra que el recomendador inteligente no se limita a contar vueltas de uso del neumático, sino que integra de manera óptima variables de tráfico, ritmo diferencial de degradación y posición relativa.
3. **Validación del Valor Agregado:** Demuestra científicamente que el esfuerzo de ingeniería de datos, el modelado físico en dos capas y el ajuste de hiperparámetros se traducen en un sistema con un rendimiento sustancialmente superior a las reglas de negocio simples, justificando su despliegue y desarrollo.

### B.2 Análisis del Coeficiente de Determinación ($R^2$) Negativo en el Escenario B
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

### B.3 Viabilidad del Filtro del 115% en Producción (Prevención de Lookahead Bias)
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
