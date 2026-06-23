# Documentación Técnica: Pipeline de Características para Recomendador de Paradas en Boxes (Capa C)

Este documento describe la arquitectura, lógica matemática, algoritmo de alineación e implementación detallada del script de ingeniería de características [f1_recommender_pipeline.py](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/src/f1_recommender_pipeline.py). 

Este pipeline genera el dataset unificado y expandido [pit_decision_candidates_v1.parquet](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/data/recommendation/pit_decision_candidates_v1.parquet), diseñado específicamente para entrenar y evaluar los modelos de regresión y ranking de paradas en boxes.

---

## 1. El Problema de la Granularidad y el Cambio Metodológico

Las dos capas iniciales del proyecto son insuficientes para un motor de recomendación de decisiones:
* **Capa A (Telemetría):** Granularidad de $1 \text{ fila} = 1 \text{ piloto} \times 1 \text{ vuelta}$. Describe el estado físico en retrospectiva, pero no las opciones hacia adelante.
* **Capa B (Táctica):** Granularidad de $1 \text{ fila} = 1 \text{ evento táctico}$ (adelantamiento, pit stop). Describe hitos en carrera, no decisiones vuelta a vuelta.

El recomendador táctico debe resolver el siguiente dilema estratégico:  
> **"Para el piloto $D$ en la vuelta $L$, ¿es óptimo parar inmediatamente o es mejor retrasar la parada $1, 2, 3, 4$ o $5$ vueltas?"**

Para formular esto como un problema de aprendizaje por ranking (**Learning to Rank**), la unidad de análisis correcta debe ser:
$$\mathbf{1\text{ fila}} = \mathbf{1\text{ piloto}} \times \mathbf{1\text{ vuelta}} \times \mathbf{1\text{ ventana de espera (candidato } 0 \dots 5\mathbf{)}}$$

El pipeline toma las **3,331** filas base de la Capa A y las expande multiplicando por las 6 opciones de decisión de parada, generando un dataset final estructurado y robusto de **19,986 filas** de entrenamiento y test.

---

## 2. Flujo de Ingesta y Procesamiento de Datos

El pipeline sigue un flujo modular carrera por carrera para mantener el aislamiento del contexto físico de cada circuito:

```mermaid
graph TD
    subgraph Fuentes de Entrada
        A[telemetry_features_v4.parquet] -->|Filtro de columnas base| E[Dataframe Piloto-Vuelta]
        B[laps.csv] -->|date_start| F[Alineador merge_asof]
        C[intervals.csv] -->|date & interval| F
    end

    subgraph Pipeline de Ingesta (process_race_recommender)
        F -->|gap_ahead| G[Calculador de Tráfico]
        G -->|Shift por posicion| H[gap_behind]
        E -->|Join por driver & lap| I[Dataset Unificado]
        H -->|Join por driver & lap| I
        I -->|Rolling 3 Laps| J[Cálculo de Slopes y Tendencias]
        J -->|Paradas en boxes reales| K[Evaluación de Éxito Post-Pit]
        K -->|Duplicación x6| L[Expansión de Candidatos y Targets]
    end

    subgraph Artefacto de Salida
        L -->|Exportar Parquet| M[pit_decision_candidates_v1.parquet]
    end
```

---

## 3. Algoritmos y Procesamientos Detallados

### 3.1 Alineación de Gaps de Tráfico (Frecuencia Mixta)
El archivo `intervals.csv` contiene registros de intervalos a alta frecuencia medidos en bucles de la pista, mientras que el dataset maestro requiere un único valor representativo de tráfico por vuelta.

1. **Alineación Temporal con `pd.merge_asof`:**
   Dado que no hay llaves directas de vuelta en los intervalos, ordenamos los registros cronológicamente y los emparejamos usando el inicio de la vuelta del piloto (`date_start`):
   ```python
   intervals_with_lap = pd.merge_asof(
       intervals,
       laps[["driver_number", "lap_number", "date_start"]],
       left_on="date",
       right_on="date_start",
       by="driver_number",
       direction="backward"
   )
   ```
   Esto asigna cada intervalo a la última vuelta que inició el piloto (es decir, cualquier intervalo posterior a la hora de inicio de la vuelta $L$, pero anterior a la de la vuelta $L+1$, se asigna a la vuelta $L$).

2. **Extracción del Gap Final:**
   Agrupamos por `driver_number` and `lap_number` y seleccionamos el último intervalo registrado de la vuelta. Este representa la distancia en segundos frente al auto de adelante en la entrada a la meta/boxes:
   ```python
   lap_intervals = intervals_with_lap.groupby(["driver_number", "lap_number"]).agg(
       gap_ahead=("interval", "last"),
       gap_to_leader=("gap_to_leader", "last")
   )
   ```

3. **Cálculo de `gap_behind` por Desplazamiento de Posiciones:**
   Para obtener el hueco libre que el piloto tiene detrás (crítico para saber si saldrá del pit en medio del tráfico), el pipeline ordena la tabla por vuelta y posición de carrera y calcula el desplazamiento inverso:
   ```python
   base_df = base_df.sort_values(["lap_number", "position"])
   base_df["gap_behind"] = base_df.groupby("lap_number")["gap_ahead"].shift(-1)
   ```
   * *Ejemplo:* Si el piloto P1 va adelante y el piloto P2 va detrás, el `gap_ahead` del piloto P2 (la distancia entre P2 y P1) representa exactamente el `gap_behind` del piloto P1.

---

### 3.2 Pendientes de Tendencia de Ritmo y Degradación
Para que el modelo estime si el auto está en un "precipicio" de rendimiento del neumático, no basta con medias estáticas. El pipeline calcula la tasa de cambio usando **Regresión Lineal por Mínimos Cuadrados** en una ventana móvil de las últimas 3 vueltas:

La pendiente $\beta$ se calcula mediante la fórmula:
$$\beta = \frac{N\sum(xy) - \sum x \sum y}{N\sum(x^2) - (\sum x)^2}$$

Donde $y$ son los valores observados (`lap_duration` o `lap_vs_best_stint`) y $x = [0, 1, 2]$.
* **`lap_slope_3`:** Tasa de pérdida o ganancia de ritmo en segundos por vuelta. Una pendiente positiva alta indica pérdida acelerada de ritmo.
* **`deg_rate_3lap`:** Pendiente del desgaste del neumático medido sobre la característica `lap_vs_best_stint` de la Capa A.

---

### 3.3 El Score de Éxito de la Parada (La Etiqueta Target)
Para cada parada en boxes real identificada (`is_pit_lap == 1`), calculamos un score continuo de éxito post-pit ($S$) utilizando una ventana de evaluación de **5 vueltas después de la parada**:

$$S = \Delta\text{Posición} + 0.5 \times \Delta\text{Ritmo}$$

Donde:
* **$\Delta\text{Posición} = \text{Posición en vuelta de parada} - \text{Posición 5 vueltas después}$.** Una ganancia de posiciones representa un score positivo (overcut/undercut exitoso).
* **$\Delta\text{Ritmo} = \text{Ritmo medio previo (3 vueltas)} - \text{Ritmo medio posterior (5 vueltas)}$.** Mide cuántos segundos por vuelta recuperó el coche al pasar a neumáticos frescos.

#### Mapeo a las Ventanas Candidatas
Al duplicar cada vuelta de piloto por los 6 candidatos (`wait_laps` de 0 a 5):
* Si el piloto en la vida real paró en la vuelta $L_p$, entonces para la vuelta $L = L_p - w$, la decisión correcta era esperar $w$ vueltas. Por lo tanto, al candidato $w$ se le asigna el score de éxito real de esa parada ($S$).
* A los candidatos alternativos se les penaliza con un score de $-2.0$ (indicando que pararon demasiado tarde o temprano respecto a la estrategia real de carrera).

---

## 4. Catálogo Completo de Columnas (Diccionario de Datos)

El dataset de salida [pit_decision_candidates_v1.parquet](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/data/recommendation/pit_decision_candidates_v1.parquet) se compone de 24 columnas estructuradas en bloques estratégicos:

| Bloque | Columna | Tipo | Descripción |
| :--- | :--- | :---: | :--- |
| **Identificadores** | `race_name` | `string` | Nombre de la carrera procesada (`australia`, `china`, `japan`, `united_states`). |
| | `driver_number` | `float64` | Número único del piloto. |
| | `lap_number` | `float64` | Número de la vuelta actual. |
| **Estado Físico** | `lap_duration` | `float64` | Tiempo de la vuelta actual en segundos. |
| | `tyre_age` | `float64` | Edad del neumático actual en vueltas. |
| | `compound_ord` | `float64` | Tipo de compuesto (codificado ordinal: SOFT=1, MEDIUM=2, HARD=3). |
| | `lap_vs_best_stint` | `float64` | Degradación acumulada (porcentaje más lento que el récord del stint). |
| | `stint_number` | `float64` | Número de stint de la carrera. |
| | `is_pit_lap` | `float64` | Bandera binaria indicadora de si la vuelta actual fue la parada real. |
| **Tráfico (Gaps)** | `gap_ahead` | `float64` | Intervalo en segundos con el coche de adelante (30.0 = aire limpio). |
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
| | `predicted_cost_of_staying`| `float64` | **Feature puente:** Tiempo acumulado esperado que se perderá si no para. Inicializado en $0.0$, a ser rellenado por la predicción del modelo de regresión (Capa 1). |
| **Target** | `success_score_label`| `float64` | **Etiqueta continua:** Score de conveniencia y éxito de la ventana de parada. |

---

## 5. Tolerancia a Fallas e Ingesta Robusta

El pipeline incorpora dos medidas de tolerancia a fallas de grado industrial para soportar las limitaciones reales de la API OpenF1:
1. **Ausencia de datos de intervalos (Ej: GP de Japón):** Si una carrera no contiene el archivo `intervals.csv`, el pipeline lo detecta automáticamente, emite una advertencia en consola y rellena las columnas de tráfico (`gap_ahead`, `gap_to_leader`, `gap_behind`) con valores neutrales (30.0 segundos). Esto previene que se descarte la carrera completa y mantiene el tamaño de los datos de entrenamiento.
2. **Estandarización de nombres:** Mapea automáticamente nombres de carpetas inconsistentes (ej: `united_states_2026` se traduce correctamente a la clave `united_states` en el dataset de telemetría Capa A).

---

## 6. Flujo del Modelo de Dos Capas

La estructura de este dataset permite la comunicación secuencial de los dos modelos propuestos por el profesor:

```text
               +--------------------------------------+
               |      Dataset de Entrada (Capa C)      |
               | (19,986 filas x 24 características)  |
               +--------------------------------------+
                                  |
                                  v
                    +---------------------------+
                    | CAPA 1: Modelo Regresión  |
                    |                           |
                    | Predice la degradación    |
                    | futura para wait_laps     |
                    +---------------------------+
                                  |
                                  v
            +-------------------------------------------+
            | Rellena: predicted_cost_of_staying        |
            +-------------------------------------------+
                                  |
                                  v
                    +---------------------------+
                    |  CAPA 2: Modelo Ranking   |
                    |                           |
                    | Ordena candidatos 0 a 5   |
                    | optimizando NDCG@K        |
                    +---------------------------+
```

1. **La Capa 1 (Regresión)** se entrena para mapear el estado actual del auto (`tyre_age`, `deg_rate_3lap`, compuesto, ritmo) a los tiempos futuros perdidos por degradación. La predicción de este modelo se inserta en la columna `predicted_cost_of_staying`.
2. **La Capa 2 (Ranking)** consume las características del estado actual, tráfico, contexto y la estimación de coste de degradación de la Capa 1 para ordenar los candidatos (0 a 5) y emitir la recomendación óptima.
