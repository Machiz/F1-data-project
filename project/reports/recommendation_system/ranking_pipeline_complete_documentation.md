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

### 5.3 Comparación de Modelos y Métricas de Rendimiento
Para medir la capacidad de generalización del modelo sobre circuitos no vistos en entrenamiento, implementamos **Validación Cruzada GroupKFold (4 particiones)** agrupando por `race_name`.

| Iteración / Modelo | MSE Promedio | $R^2$ Score (Test) | $R^2$ Score (Entrenamiento) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 235.0272 | 0.0890 | 0.8539 |
| **Gradient Boosting (Base)** | 283.5258 | 0.5459 | 0.9580 |
| **XGBoost (Fine-tuned)** | 314.1477 | 0.3778 | 0.9790 |
| **Extra Trees (Optimized)** | - | - | 0.9819 |
| **Stacking Regressor (Ensamble Final)** | **-** | **-** | **0.9941** |

> [!NOTE]
> La regresión lineal simple fallaba críticamente ($R^2$ promedio en validación cruzada de solo **8.9%**) debido al comportamiento no lineal de la degradación (conocido en F1 como el "precipicio del neumático" o *tyre cliff*).

### 5.4 Justificación del Ensamble por Stacking
Se seleccionó una arquitectura de **Stacking Regressor** que combina la potencia de predicción local y global:
1. **Estimador 1 (XGBoost Regressor):** Excelente para mapear interacciones no lineales complejas entre el desgaste del neumático (`tyre_age`) y la pendiente de degradación reciente (`deg_rate_3lap`).
2. **Estimador 2 (Extra Trees Regressor):** Algoritmo extremadamente robusto ante el ruido, que reduce la varianza del ensamble mediante la aleatorización extrema de los umbrales de decisión.
3. **Meta-Regresor (Ridge Regression):** Modela linealmente las predicciones de los dos algoritmos previos para evitar el sobreajuste y estabilizar la salida en segundos.

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

---

## 7. Capa 2: Ranking de Opciones de Parada en Boxes

El objetivo de la Capa 2 es recibir las características actuales de carrera, el tráfico esperado delante/detrás y el costo acumulado por degradación (`predicted_cost_of_staying`) para ordenar de mejor a peor las 6 alternativas de parada y sugerir la decisión óptima.

### 7.1 Enfoque de Modelado: Point-wise vs. List-wise
Comparamos dos metodologías competitivas para el problema de ordenamiento:

1. **Enfoque A: Random Forest Regressor (Point-wise)**
   * Entrena un regresor robusto para predecir la etiqueta de éxito continua `success_score_label` de forma aislada para cada candidato, y posteriormente los ordena de mayor a menor dentro de cada vuelta.
2. **Enfoque B: XGBRanker (List-wise)**
   * Utiliza lambdas de pérdida orientadas a optimizar directamente la métrica NDCG agrupando los candidatos por ID de consulta (`race_name_driver_number_lap_number`). Requiere la discretización de la etiqueta continua en rangos enteros (0 a 5).

### 7.2 Resultados del Comparativo de Ranking

| Enfoque / Algoritmo | NDCG@1 Promedio | NDCG@3 Promedio | Estado |
| :--- | :---: | :---: | :---: |
| **Point-wise (Random Forest Regressor)** | **0.9342** | **0.9453** | **SELECCIONADO** |
| **List-wise (XGBRanker)** | 0.8179 | 0.8727 | **Descartado** |

### 7.3 Justificación de la Selección
1. **Conservación de la Magnitud Física:** La etiqueta `success_score_label` es continua ($\Delta\text{Posición} + 0.5 \times \Delta\text{Ritmo}$) y su magnitud absoluta es altamente informativa. El modelo Point-wise aprende a predecir cuánta ventaja exacta dará la parada (ej. $+6.2$ de score frente a un coste marginal de $+0.1$). Discretizar esta variable para XGBRanker destruye esta escala física y reduce la precisión del ranking a un **81.79%** de NDCG@1.
2. **Importancia del Costo Puente:** En el modelo final de Random Forest, la característica calculada `predicted_cost_of_staying` obtuvo la mayor ganancia de información (importancia de feature **>40%**), validando metodológicamente la necesidad de estructurar el sistema en dos capas.

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

## 9. Conclusiones Relevantes

* **Desacoplamiento Físico-Táctico Eficaz:** La arquitectura desacoplada en dos capas permite aislar las estimaciones puramente físicas del ritmo del coche (Capa 1) de la toma de decisiones estratégicas bajo tráfico y contexto de carrera (Capa 2). Esto evita que el modelo de recomendación final copie sesgos históricos de decisiones ineficientes tomadas en el muro real de boxes.
* **Robustez ante el Ruido en Pista**: La integración del filtro de outliers del 115% de ritmo y el uso de Extra Trees en el ensamble de Stacking mitigan los errores causados por incidentes locales de carrera (Safety Car, errores de pilotaje, etc.), permitiendo que el recomendador trabaje con estimaciones limpias de la degradación real.
* **Consistencia Metodológica en Inferencia**: La exportación e integración del alineador de features garantiza que el preprocesamiento de variables dummy sea consistente, permitiendo evaluar circuitos completamente nuevos sin perder la estructura que espera la Capa 1.

---

## 10. Próximos Pasos y Extensiones Futuras

* **Modelado del Ritmo y Degradación de Rivales Directos**: Incorporar estimaciones concurrentes sobre el ritmo e historial de neumáticos de los pilotos delantero y trasero inmediatos. Esto enriquecerá a la Capa 2 para predecir de forma proactiva oportunidades de *undercut* (adelantar parando antes) o defenderse de un *overcut* (permanecer en pista para ganar posición).
* **Variables Probabilísticas del Clima y Safety Cars**: Integrar la probabilidad sectorizada de banderas amarillas o accidentes históricos en el circuito, así como el desgaste de neumáticos en condiciones variables (lluvia extrema, intermedia y secado de pista).
* **Optimización Táctica Global mediante Aprendizaje por Refuerzo**: Migrar la inferencia desde ventanas locales de 5 vueltas a simulaciones estratégicas globales de carrera completa (usando Q-learning o deep RL) para optimizar el número total de stints y compuestos a utilizar desde la largada hasta la bandera de cuadros.
