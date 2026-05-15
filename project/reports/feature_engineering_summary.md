# 🏎️ F1 Race Intelligence - Resumen de Feature Engineering

Este documento detalla el proceso de **Feature Engineering** (Ingeniería de Características) aplicado en el proyecto para analizar los eventos tácticos de la Fórmula 1 (como adelantamientos en pista y undercuts).

El objetivo principal de este proceso es transformar los datos de telemetría en crudo y los eventos detectados en una matriz estructurada donde **cada fila representa un evento táctico** y **cada columna describe el contexto profundo** de ese evento (llegando a construir más de 500 variables). Esto prepara la información para aplicar técnicas de Machine Learning como Análisis de Componentes Principales (PCA) y Clustering.

---

## 📂 1. Archivos Clave en el Proceso

El feature engineering se lleva a cabo en diferentes fases, repartidas en los siguientes archivos:

1. **`project/src/f1_events_pipeline.py`**:
   - **Propósito**: Ejecuta el preprocesamiento global y la generación de la tabla maestra (`master_parquet`) y la extracción inicial de eventos (`events_parquet`).
   - **Acciones**: Estandariza tiempos de vuelta a segundos, cruza los datos de `laps.csv` con los datos de neumáticos (`stints.csv`) y las paradas en boxes (`pit.csv`). Además, realiza una **reconstrucción matemática de la posición** de los pilotos basada en el tiempo acumulado en caso de que este dato falte en el dataset original. Finalmente, detecta y exporta los eventos base (`On_Track_Overtake` y `Pit_Strategy`).

2. **`project/notebooks/feature_engineering_tactical_events.ipynb`**:
   - **Propósito**: Es el notebook core original donde se estructuró la extracción de características usando ventanas temporales (ej. analizando las 3 vueltas previas al evento).

3. **`project/notebooks/Feature_engineering_v3.ipynb`**:
   - **Propósito**: La versión más avanzada y completa del Feature Engineering, orientada a superar las 500 variables para un análisis PCA significativo. Implementa lógicas refinadas con `polars` para optimizar el rendimiento y añade bloques de contexto avanzado (eficiencias, ratios, campo completo).

---

## 🛠️ 2. ¿Cómo funciona el Feature Engineering? (Metodología)

El motor del proceso radica en comparar al **Atacante (Initiator)** contra el **Defensor (Target)** en los momentos previos al evento. La metodología se divide en los siguientes bloques (referenciados en la V3):

### A. Ventanas Temporales de Telemetría
Para cada evento (ej. Vuelta 15), se extrae una "ventana" de las vueltas previas (T-1, T-2, T-3, y ventanas extendidas de 5 y 10 vueltas) tanto para el atacante como para el defensor.

### B. Agregaciones Estadísticas sobre Sensores
Sobre los sensores disponibles de esas ventanas temporales (`lap_duration`, velocidades `i1_speed`, `i2_speed`, `st_speed`, duración de los sectores, `throttle`, `brake`, etc.), se calculan 5 estadísticos básicos:
- **Media (`mean`)**
- **Máximo (`max`) y Mínimo (`min`)**
- **Desviación Estándar (`std`)** para medir consistencia.
- **Pendiente / Tendencia (`slope`)** mediante regresión lineal, que indica si un piloto está mejorando o empeorando en ese sensor específico.

### C. Variables Delta (Diferenciales)
Se calcula la diferencia numérica exacta entre el Atacante y el Defensor para todas las estadísticas anteriores.
- *Ejemplo*: `delta_tyre_age` (diferencia de edad del neumático), `delta_lap_duration_mean`, `delta_degradation`. Esto le dice al modelo exactamente cuánta ventaja tenía el atacante sobre el defensor en esa métrica particular.

### D. Ratios, Eficiencias y Contexto de Carrera
Se crean variables de alto nivel para darle "inteligencia de carrera" al modelo:
- **Proporción Sectorial**: Qué porcentaje de la vuelta se pasa en cada sector.
- **Pace drop / Degradación**: El ratio de pérdida de tiempo dentro del "stint" (conjunto de vueltas con el mismo neumático) actual.
- **Eficiencias**: Ratios de aceleración vs frenado, y consistencia de velocidad.
- **Contexto del Campo (Pace vs Field)**: Se calcula la mediana del tiempo de vuelta de **todos** los pilotos en la carrera en ese instante y se compara el tiempo del atacante/defensor contra el resto de la parrilla.
- **Safety Car**: Se detectan posibles periodos de Safety Car midiendo anomalías en los tiempos globales del campo.
- **Fase de la Carrera**: Variables que indican si el evento ocurre al inicio, mitad o final de la carrera (`is_early_race`, `race_pct_complete`).

### E. Variable Objetivo (Target)
En el notebook de `tactical_events`, se define la variable `success` (éxito). Toma el valor `1` si el atacante logró consolidar la posición (la posición numérica resultante fue menor que la original) o si el undercut funcionó, y `0` si fracasó.

---

## 📈 3. Hallazgos del Proceso de Ingeniería

1. **Explosión Dimensional Exitosa**: Se ha logrado pasar de un conjunto de ~20 columnas básicas de telemetría a un dataset robusto de **540 variables explicativas** por evento (en la V3). Esto provee un lienzo excepcionalmente detallado para modelos multivariados como PCA.
2. **Reconstrucción de Datos**: El script de preprocesamiento demostró robustez al poder reconstruir matemáticamente las posiciones en pista (`f1_events_pipeline.py`) cruzando los deltas de tiempos acumulados, salvando problemas de completitud de datos originales.
3. **El poder de la "Tendencia" (`slope`)**: Una de las adiciones más significativas fue el cálculo de la pendiente de regresión lineal sobre las últimas vueltas. En lugar de ver si un piloto es "rápido" en promedio, la variable `slope` y `delta_slope` permite saber quién trae mayor **"momentum"** o ritmo ascendente justo antes del ataque.
4. **Validación de la Tasa de Éxito**: En la exploración preliminar de la variable target, la inmensa mayoría de los eventos detectados como "Adelantamientos en Pista" por el pipeline resultan en una tasa de éxito matemática cercana al 100%. Esto valida que el pipeline de eventos está capturando movimientos consolidados, pero también sugiere que si se desea predecir la *probabilidad* de éxito antes del evento, el modelo podría necesitar que se incluyan intentos fallidos (ataques que no terminaron en adelantamiento) en las futuras iteraciones de extracción para equilibrar las clases.
5. **Rendimiento**: Mover la lógica de cálculo pesada (estadísticas rodantes y regresiones lineales) a **Polars** en la versión V3 optimizó radicalmente el proceso, permitiendo calcular ~540 variables para cientos de eventos en cuestión de un par de minutos.

---

## 🗂️ 4. Diccionario de Variables Creadas (Selección Relevante)

A continuación, se presenta una tabla que resume las principales categorías de las más de 500 variables creadas a partir del *Feature Engineering*, indicando cómo se calcularon y por qué son importantes.

| Categoría de Variable | Ejemplos de Variables | ¿Cómo se Consiguió? | Datos y Justificación Relevante |
| :--- | :--- | :--- | :--- |
| **Estadísticas Base por Ventana** | `att_lap_duration_mean`<br>`def_st_speed_std`<br>`att_i1_speed_max` | Calculando agregaciones (`mean`, `max`, `min`, `std`) sobre la telemetría en las últimas 3, 5 y 10 vueltas. | Resume el ritmo y consistencia general del atacante (`att`) y defensor (`def`) antes del evento. Evita el ruido de una sola vuelta. |
| **Tendencia y Momentum** | `att_lap_duration_slope`<br>`def_throttle_slope` | Aplicando regresión lineal (`np.polyfit`) sobre los valores de la ventana temporal. | Un *slope* negativo en `lap_duration` indica que el piloto está bajando sus tiempos (mejorando el ritmo o *momentum*). |
| **Variables Diferenciales (Deltas)** | `delta_tyre_age`<br>`delta_lap_duration_mean`<br>`delta_lap_duration_slope` | Restando matemáticamente el valor del defensor al del atacante (`Atacante - Defensor`). | Mide la **ventaja directa** en pista. Un `delta_tyre_age` negativo significa que el atacante tiene llantas más frescas. |
| **Rendimiento Sectorial** | `att_sector1_pct`<br>`delta_sector3_pct` | Dividiendo el tiempo promedio del sector entre el tiempo de vuelta promedio (`sector_time / lap_duration`). | Revela en qué sector específico un piloto tiene ventaja. Ayuda a predecir *dónde* ocurrirá el adelantamiento. |
| **Degradación del Neumático** | `att_deg_rate`<br>`delta_deg_rate`<br>`pace_drop` | Calculando la pendiente del tiempo de vuelta desde el inicio del *stint* actual de neumáticos. | Clave para predecir si un piloto está cayendo en el *cliff* del neumático (pérdida severa de rendimiento). |
| **Contexto del Campo (Pace vs Field)** | `att_pace_vs_field`<br>`field_lap_duration_cv` | Dividiendo el tiempo de vuelta del piloto entre la mediana de tiempo de vuelta de toda la parrilla. | Normaliza el ritmo. Un safety car o bandera amarilla afecta a todos, esta métrica aísla el ritmo real frente al pelotón. |
| **Eficiencias de Conducción** | `speed_per_rpm`<br>`throttle_brake_ratio` | Ratios entre velocidad máxima (`st_speed`) y RPM, o uso de acelerador vs freno. | Describe el estilo de conducción y la eficiencia del monoplaza en la recta vs curvas. |
| **Variables Target y Categóricas** | `success` (Target)<br>`att_compound_ord`<br>`is_on_track_overtake` | Extracción de reglas de negocio (`P10 -> P7` = éxito) y *One-Hot Encoding* o mapeo ordinal. | Prepara los datos cualitativos (como el compuesto Medio=2, Suave=1) para que los algoritmos matemáticos puedan procesarlos. |
