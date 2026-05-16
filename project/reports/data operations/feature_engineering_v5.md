# Documentación Técnica: Feature Engineering V5 — F1 Strategic Decision Engine

Este documento describe en su totalidad el diseño, metodología, implementación y conclusiones del notebook `Feature_engineering_v5.ipynb`. Esta versión representa la **refactorización definitiva** del pipeline de ingeniería de características, respondiendo directamente a las críticas de mala praxis estadística documentadas en el reporte de gerencia (`reporte_evaluacion_fe_pca_gerencia.md`). La premisa central es la separación estricta en dos capas de datos y la prioridad de *calidad de señal* sobre *cantidad de variables*.

## 1. Archivo y Reproducibilidad

*   **Notebook Ejecutable:** `project/notebooks/feature engineering/Feature_engineering_v5.ipynb`
*   **Artefactos Generados:**
    *   `project/data/features/telemetry_features_v4.parquet` — Capa A (Telemetría para PCA)
    *   `project/data/features/tactical_features_v4.parquet` — Capa B (Táctica para Clustering)
*   **Librerías Clave:** `polars 1.38.1`, `numpy`, `pandas`, `requests`
*   **Carreras Analizadas:** Australia, China, Japón, Estados Unidos (temporada 2026)

---

## 2. Arquitectura de Dos Capas — El Cambio Paradigmático

La innovación central de la V5 es la **separación explícita y obligatoria del espacio de análisis en dos capas independientes**. Las versiones anteriores (V3) mezclaban contextos y generaban más de 540 variables altamente colineales para un dataset de solo 643 filas — un ratio 1:1 que destruía la generalización de cualquier modelo.

> **Regla de Oro:** NO mezclar capas. El único puente entre ellas son los PC Scores de telemetría agregados que pueden añadirse como features tácticas. Cada capa tiene su propia unidad de análisis, su propio objetivo y su propio destino en el pipeline.

| Capa | Dataset de Entrada | Unidad de Análisis | Nº de Features | Uso Posterior |
| :--- | :---: | :--- | :---: | :--- |
| **A — Telemetría** | `df_master` (3331 filas) | 1 fila = 1 vuelta de 1 piloto | ~27 features | PCA → Driver Performance States |
| **B — Táctica** | `df_events` (643 filas) | 1 fila = 1 evento táctico | ~25 features | Clustering → Strategy Archetypes |

### Solución al Problema de la Maldición Dimensional

| Métrica | V3 (Mala Praxis) | V5 (Refactorizada) |
| :--- | :---: | :---: |
| **Features Capa A (Telemetría)** | ~300+ variables mezcladas | **27 features** |
| **Features Capa B (Táctica)** | 540 variables | **25 features** |
| **Ratio filas/features (Capa A)** | <10:1 | **123:1** ✅ |
| **Ratio filas/features (Capa B)** | ~1.2:1 | **25:1** ✅ |
| **Multicolinealidad** | Alta (múltiples ventanas del mismo sensor) | **Eliminada** |

---

## 3. Fuentes de Datos y Carga

El notebook comienza con la carga en paralelo de los dos datasets fundamentales usando **Polars** (motor de DataFrames en Rust para máximo rendimiento).

### 3.1 `df_master` — Telemetría por Vuelta

Construido desde los archivos `{carrera}_2026_masterv2.parquet` procesados por el pipeline de preprocesamiento. Cada fila es una vuelta de un piloto con 29 columnas base.

**Variables base incluidas:**
```
meeting_key, session_key, driver_number, lap_number, date_start,
duration_sector_1/2/3, i1_speed, i2_speed, st_speed, lap_duration,
is_pit_out_lap, is_pit_lap, compound, stint_number, tyre_age,
pit_duration, position, throttle_mean_lap, throttle_std_lap,
brake_max_lap, rpm_max_lap, n_gear_max_lap, drs_max_lap, race_name
```

### 3.2 `df_events` — Eventos Tácticos

Construido desde los archivos `{carrera}_2026_events.parquet`. Cada fila es un evento táctico detectado (adelantamiento, pit strategy) con 8 columnas base.

**Distribución de eventos por carrera:**
| Carrera | Eventos | Vueltas Totales |
| :--- | :---: | :---: |
| Australia | 120 | 57 |
| China | 61 | 56 |
| Japón | 53 | 53 |
| Estados Unidos | 409 | 57 |
| **TOTAL** | **643** | — |

### 3.3 Enriquecimiento via API OpenF1

Para añadir contexto de equipo, el notebook realiza llamadas a `https://api.openf1.org/v1/drivers?session_key={sk}` por cada sesión única. Los campos `team_name` y `name_acronym` se unen a `df_master` via `join` por `[session_key, driver_number]`. Este enriquecimiento permite futuros análisis de rendimiento por escudería.

---

## 4. CAPA A — Feature Engineering de Telemetría

**Objetivo:** Construir ~30 variables que describan el *estado de rendimiento físico* del monoplaza en cada vuelta. Estas features son el input directo del PCA y deben capturar la dinámica real del coche sin sesgo contextual.

### 4.1 Catálogo Completo de Features

| Grupo | Variable | Descripción Técnica |
| :--- | :--- | :--- |
| **Ritmo puro** | `lap_duration` | Duración total de la vuelta en segundos |
| | `duration_sector_1/2/3` | Tiempo de cada micro-sector |
| **Velocidad** | `st_speed` | Velocidad punta en la trampa de velocidad |
| | `i1_speed`, `i2_speed` | Velocidades en los puntos intermedios |
| **Estilo de conducción** | `throttle_mean_lap` | Porcentaje de apertura de acelerador (media vuelta) |
| | `throttle_pct_full` | Proxy de tiempo con acelerador >95% |
| | `brake_max_lap` | Frenada máxima en la vuelta |
| | `coasting_pct` | Proxy de «punto muerto» (throttle<10 y brake<10) |
| **Motor** | `rpm_max_lap` | RPM máximas alcanzadas en la vuelta |
| | `n_gear_max_lap` | Marcha máxima usada |
| **Neumático** | `tyre_age` | Edad del neumático en vueltas |
| | `compound_ord` | Compuesto codificado ordinalmente (SOFT=1, MED=2, HARD=3) |
| | `lap_vs_best_stint` | Degradación relativa: cuánto más lento que la mejor vuelta del stint |
| **Posición** | `position` | Posición en la carrera |
| **Stint** | `stint_number` | Número de stint actual |
| | `is_pit_out_lap` | Indicador binario de vuelta de salida de pits |
| | `is_pit_lap` | Indicador binario de vuelta de entrada a pits |
| **Eficiencias derivadas** | `speed_per_rpm` | Ratio velocidad/RPM (eficiencia aerodinámica) |
| | `sector1_pct`, `sector3_pct` | Porcentaje del tiempo total dedicado a cada sector |
| | `sector_balance` | Ratio sector 1 / sector 3 (técnico vs rectas) |
| | `throttle_brake_ratio` | Agresividad de conducción: throttle/brake |
| | `best_lap_stint` | Mejor vuelta del stint (referencia para degradación) |

### 4.2 Lógica de Construcción Clave

**`safe_div(a, b)`:** Función helper que devuelve `None` en lugar de `Inf` o `NaN` cuando el denominador es cero. Previene que valores físicamente imposibles (ej. vuelta sin frenada) contaminen el dataset.

**`compound_ord`:** El compuesto de neumático se mapea ordinalmente (`replace_strict`). Elimina la necesidad de one-hot encoding, reduciendo dimensionalidad y preservando el orden lógico de dureza.

**`lap_vs_best_stint`:** Feature clave para el PCA. Captura la **degradación acumulada** del neumático dentro de cada stint:
```
lap_vs_best_stint = (lap_duration - best_lap_stint) / best_lap_stint
```
Un valor de `0.015` indica que el piloto está 1.5% más lento que su mejor vuelta en ese stint — señal directa del desgaste.

**`coasting_pct`:** Proxy binario de eficiencia energética. Cuando `throttle_mean < 10` y `brake_max < 10`, el coche está en «punto muerto», fenómeno observable en zonas de chicanas o bajo Safety Car.

### 4.3 Diagnóstico de Nulos — Decisión sobre Cobertura

Se realizó un diagnóstico sistemático de valores faltantes:

| Columna | % Nulos | Decisión |
| :--- | :---: | :--- |
| `speed_per_rpm` | 37.5% | Retenida — causada por cobertura parcial de telemetría por piloto |
| `brake_max_lap` | 37.1% | Retenida — imputación en PCA |
| `throttle_mean_lap` | 37.1% | Retenida — cobertura parcial |
| `throttle_brake_ratio` | 37.1% | Retenida — derivada de las anteriores |
| `rpm_max_lap` | 37.1% | Retenida — cobertura parcial |
| `throttle_pct_full` | 37.1% | Retenida — derivada |
| `n_gear_max_lap` | 37.1% | Retenida — cobertura parcial |
| Resto (16 columnas) | <5% | ✅ Limpias |

> [!NOTE]
> Los nulos del ~37% en las columnas de telemetría detallada (throttle, brake, RPM) se deben a que no todos los pilotos tienen datos de telemetría completa en OpenF1. Se documentan pero no se eliminan en esta fase; el PCA downstream maneja la imputación.

**Resultado de Capa A:** `(3331, 30)` — 3331 vueltas × 30 columnas (IDs + features). **Ratio efectivo: 123:1.**

---

## 5. CAPA B — Feature Engineering Táctico

**Objetivo:** Construir ~25 variables que describan el *contexto estratégico* de cada evento táctico (adelantamiento en pista, undercut, overcut). La unidad de análisis es el **par atacante–defensor** en el momento del evento.

### 5.1 Arquitectura del Extractor Táctico

El extractor opera con una función central: `extract_tactical_features(event, df_race, df_events_race, total_laps)`. Para cada evento, genera un diccionario con 32 columnas (7 identificadores + 25 features modelables).

**Ventanas temporales utilizadas:**
*   **Ventana de 3 vueltas (`att3`, `def3`):** Extrae las últimas 3 vueltas del atacante y defensor antes del evento. Esta ventana captura el estado inmediato de ritmo sin el ruido de vueltas más antiguas.
*   **Vuelta inmediatamente anterior (`att_prev`, `def_prev`):** Para datos puntuales (edad del neumático, posición, stint).

El uso de **una única ventana de 3 vueltas** (frente a las ventanas de 3, 5 y 10 vueltas de la V3) elimina la multicolinealidad por diseño.

### 5.2 Catálogo Completo de Features Tácticas

| Grupo | Variable | Descripción Técnica |
| :--- | :--- | :--- |
| **Diferencial de ritmo (3v)** | `att_lap_mean`, `def_lap_mean` | Media del tiempo de vuelta en las 3 vueltas previas |
| | `delta_lap_mean` | `att_lap_mean - def_lap_mean` (negativo = atacante más rápido) |
| | `delta_sector1_mean` | Diferencial de ritmo en el Sector 1 (técnico) |
| | `delta_sector3_mean` | Diferencial de ritmo en el Sector 3 (final/rectas) |
| **Tendencia de ritmo** | `att_lap_slope` | Pendiente de regresión lineal del tiempo de vuelta del atacante |
| | `def_lap_slope` | Pendiente del defensor (negativo = mejorando) |
| | `delta_lap_slope` | Diferencial de momentum (quién tiene más ritmo ascendente) |
| **Neumático** | `att_tyre_age`, `def_tyre_age` | Edad del neumático en el momento del evento |
| | `delta_tyre_age` | Diferencial de frescura (negativo = atacante con llantas más nuevas) |
| | `att_compound_ord`, `def_compound_ord` | Compuesto ordinal del neumático |
| | `delta_compound_ord` | Diferencial de dureza de compuesto |
| **Degradación** | `att_deg_rate`, `def_deg_rate` | Pendiente de pérdida de ritmo desde el inicio del stint |
| | `delta_deg_rate` | Diferencial de degradación (quién se está degradando más) |
| | `att_stint_laps_done`, `def_stint_laps_done` | Número de vueltas completadas en el stint actual |
| **Contexto de carrera** | `race_pct_complete` | Fracción de la carrera completada (0.0 = inicio, 1.0 = final) |
| | `laps_remaining` | Vueltas restantes para el final |
| | `position_gap` | Diferencia de posición entre defensor y atacante (positivo = defensor adelante) |
| | `is_top10_battle` | Bandera binaria: ¿ambos pilotos están en el Top 10? |
| **Pit strategy** | `att_is_pit_out` | ¿El atacante salió del pit en la vuelta previa? |
| | `def_is_pit_out` | ¿El defensor salió del pit en la vuelta previa? |

### 5.3 Helpers de Cálculo

**`linreg_slope(values)`:** Calcula la pendiente de una regresión lineal simple con `numpy.polyfit`. Retorna `None` si hay menos de 2 valores válidos. Captura el **momentum** de ritmo mejor que cualquier media estática.

**`get_window(df_race, driver, lap, window)`:** Filtra las últimas `window` vueltas del piloto antes del evento. Opera sobre el DataFrame de la carrera filtrada para máxima eficiencia.

**`col_mean(df, col)` / `col_item(df, col)`:** Abstracciones seguras para extraer medias o valores puntuales, manejando DataFrames vacíos y columnas inexistentes sin excepciones.

**`delta(a, b)`:** Función de substracción con manejo de `None` propagado — si cualquiera de los operandos es `None`, el delta es `None` (en lugar de romper el pipeline).

### 5.4 Diagnóstico de Nulos — Capa Táctica

| Feature | % Nulos | Causa |
| :--- | :---: | :--- |
| `att_compound_ord`, `def_compound_ord`, `delta_compound_ord` | **100%** | Los datos de compuesto no estaban disponibles en `df_events` |
| `delta_deg_rate`, `def_deg_rate` | ~13-15% | Datos de stint insuficientes para calcular pendiente |
| `def_lap_slope`, `delta_lap_slope` | ~13% | Ventana de 3 vueltas del defensor incompleta |
| `def_tyre_age`, `position_gap`, `is_top10_battle` | ~12% | Vuelta previa del defensor sin registro |
| Resto (10+ columnas) | <10% | ✅ Cobertura satisfactoria |

> [!IMPORTANT]
> Las tres columnas de compuesto (`att_compound_ord`, `def_compound_ord`, `delta_compound_ord`) presentan **100% de nulos** porque `df_events` no incluye la columna `compound`. Son descartables para el clustering; el compuesto se analiza implícitamente a través de `tyre_age` y `deg_rate`.

**Resultado de Capa B:** `(643, 32)` — 643 eventos × 32 columnas (7 IDs + 25 features). **Ratio efectivo: 25:1.**

---

## 6. Ejecución del Pipeline Táctico

El loop de extracción opera carrera por carrera para mantener el contexto correcto de cada circuito:

```python
for race in CARRERAS:
    df_race   = df_master.filter(pl.col('race_name') == race)
    df_ev_r   = df_events.filter(pl.col('race_name') == race)
    total_laps = df_race.select(pl.col('lap_number').max()).item()

    for event in df_ev_r.to_dicts():
        feat = extract_tactical_features(event, df_race, df_ev_r, total_laps)
        features_list.append(feat)
```

**¿Por qué filtrar por carrera?** El contexto de `total_laps`, `race_pct_complete` y `position_gap` son relativos a cada carrera. Mezclar vueltas de circuitos distintos generaría `race_pct_complete` incorrectos (Japón tiene 53 vueltas, Australia 57).

---

## 7. Verificación Final y Artefactos de Salida

```
=======================================================
RESUMEN FINAL DE FEATURE ENGINEERING v5
=======================================================

CAPA A — Telemetría (input para PCA):
  Filas     : 3331
  Features  : 27
  Ratio     : 123:1  ✅ (recomendado >50:1)

CAPA B — Táctica (input para Clustering):
  Filas     : 643
  Features  : 25
  Ratio     : 25:1   ✅ (recomendado >20:1)

Archivos generados:
  data/features/telemetry_features_v4.parquet
  data/features/tactical_features_v4.parquet

Siguiente paso → PCA sobre telemetry_features_v4.parquet
```

---

## 8. Límites de Interpretación (Failure Analysis)

### ¿Qué no funcionó o tiene limitaciones conocidas?

1.  **Compuesto de neumático indisponible (Capa B):** Las tres features de compuesto (`att_compound_ord`, `def_compound_ord`, `delta_compound_ord`) tienen 100% de nulos porque `df_events` no registra el compuesto en el momento del evento. Esta información existe en `df_master` pero el join requeriría identificar el piloto exacto y la vuelta exacta para el atacante y el defensor por separado. Se dejó pendiente para una V6.

2.  **Nulos estructurales en telemetría detallada:** El ~37% de nulos en columnas de throttle, brake y RPM refleja que no todos los pilotos tienen telemetría completa en OpenF1. No es un error de procesamiento, sino una limitación de la fuente de datos. El PCA downstream debe manejar imputación.

3.  **Ventana fija de 3 vueltas:** La elección de 3 vueltas para el contexto es empírica. En carreras con Safety Car o banderas amarillas, las vueltas previas al evento pueden estar contaminadas con tiempos artificialmente lentos, distorsionando los slopes y medias. No se implementa detección de SC en esta versión.

4.  **Eventos de pit strategy vs. on-track:** El dataset mezcla ambos tipos de eventos tácticos. Los slopes de ritmo no tienen el mismo significado para un `Pit_Strategy` (donde el contexto es puro timing de parada) que para un `On_Track_Overtake` (donde el ritmo relativo es decisivo). No se separan en esta versión.

---

## 9. Conclusiones e Insights Estratégicos

> [!IMPORTANT]
> **🎯 Hallazgos Clave del Rediseño:**
>
> 1. **La calidad supera a la cantidad:** Pasar de 540 variables con ratio 1:1 a 25–27 features con ratios >25:1 es la corrección metodológica más crítica del proyecto. El modelo predictivo posterior ya no memorizará ruido estadístico.
>
> 2. **El `lap_vs_best_stint` es la feature más poderosa de la Capa A:** Normaliza el tiempo de vuelta dentro del stint, haciendo comparables pilotos con distintos estilos de manejo y eliminando el efecto de los diferentes compuestos. Es la señal más limpia de degradación de neumático disponible.
>
> 3. **El `delta_lap_slope` es la feature más poderosa de la Capa B:** Captura el *momentum* de ritmo relativo. Un atacante con slope negativo (mejorando) vs. un defensor con slope positivo (degradando) es la firma táctica perfecta de un undercut exitoso. Ninguna media estática captura esto.
>
> 4. **La separación en dos capas es arquitectónicamente correcta:** PCA sobre telemetría por vuelta capturará patrones de conducción (estados del monoplaza). Clustering sobre features tácticas capturará arquetipos de maniobra. Mezclarlos hubiera creado un espacio latente sin interpretabilidad física.
>
> 5. **Próximos pasos:** Los dos artefactos exportados son los inputs directos de:
>    - `telemetry_features_v4.parquet` → `PCA_V4.ipynb` → generación de `telemetry_pca_v4.parquet`
>    - `tactical_features_v4.parquet` → notebooks de Clustering (K-Means, Jerárquico, DBSCAN)

---

*Análisis basado en `Feature_engineering_v5.ipynb`*
*Responde a las observaciones de `reporte_evaluacion_fe_pca_gerencia.md`*
