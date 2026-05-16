# Análisis y Desarrollo del PCA — Telemetría F1 (V4)

Este documento detalla la implementación, metodología y hallazgos del Análisis de Componentes Principales (PCA) desarrollado para la **Capa A (Telemetría)** del proyecto de datos de la Fórmula 1, Versión 4. El objetivo es reducir la dimensionalidad del dataset de telemetría por vuelta para capturar los **estados de rendimiento del piloto** de forma interpretable y descorrelacionada, preparando los datos para el clustering posterior.

---

## 1. Resumen y Archivos Clave

| Parámetro | Valor |
|---|---|
| **Notebook Principal** | `project/notebooks/dimensionality reduction/PCA_v4.ipynb` |
| **Dataset de Entrada** | `telemetry_features_v4.parquet` |
| **Dataset de Salida** | `telemetry_pca_v4.parquet` |
| **Dimensiones de Entrada** | 3331 filas × 30 columnas (26 features + 4 ID) |
| **Filas tras Filtrado** | 3004 (se excluyen pit laps y anomalías de pace) |
| **Features al PCA** | 24 variables numéricas |
| **Componentes Retenidos** | **6 PCs** → 78.7% de varianza explicada |
| **Umbral 80% varianza** | 7 PCs |
| **Umbral 90% varianza** | 9 PCs |
| **Temporada** | 2026 (Australia, China, Japón, Estados Unidos) |

---

## 2. Objetivo del PCA

> [!NOTE]
> **Contexto:** El pipeline de Feature Engineering V5 (Capa A - Telemetría) generó ~27 features de telemetría por vuelta. El PCA es el paso de reducción de dimensionalidad que convierte estas features en un conjunto compacto de **Componentes Principales** listos para el clustering.

El objetivo del PCA V4 es cuádruple:

1. **Capturar estados de rendimiento**: Representar de forma latente si el piloto está en modo *Push Lap*, *Energy Saving*, *Tyre Management* o *Pit Out*.
2. **Eliminar multicolinealidad**: Las features de telemetría presentan alta correlación (ej. `throttle_mean_lap` y `throttle_pct_full`). El PCA produce componentes ortogonales.
3. **Preservar varianza máxima**: Retener la mayor cantidad de señal predictiva con el menor número de componentes.
4. **Preparar inputs para clustering**: El output de scores `PC1–PC6` es el dataset que ingresa al modelo K-Means de telemetría.

### Estados de Rendimiento Esperados

| Estado | Descripción | Indicadores Clave |
|---|---|---|
| **Push Lap** | Vuelta a ritmo máximo | Throttle alto, RPM alto, `lap_duration` bajo, `tyre_age` bajo |
| **Energy Saving** | Vuelta de gestión energética | Throttle bajo, `coasting_pct` alto, pace sacrificado |
| **Tyre Management** | Gestión del neumático en degradación | `lap_vs_best_stint` alto, `deg_rate` creciente, ritmo decayendo |
| **Pit Out / Fresh Tyre** | Vuelta de salida de boxes | `tyre_age` = 1-2, `lap_duration` rápido, `stint_number` alto |

---

## 3. Pipeline de Preparación y Limpieza de Datos

### 3.1. Separación de Identificadores y Features

El dataset de entrada contiene columnas que no deben entrar al PCA porque son **identificadores** o **flags de estado binarios** sin valor de señal continua:

| Tipo | Columnas Excluidas | Razón |
|---|---|---|
| **Identificadores** | `race_name`, `driver_number`, `lap_number`, `team_name` | No aportan varianza de rendimiento |
| **Flags binarios** | `is_pit_out_lap`, `is_pit_lap`, `stint_number` | Variables de estado, no de rendimiento continuo |

### 3.2. Filtrado de Outliers Estructurales

Antes de aplicar el PCA se aplican dos filtros críticos:

```python
# Filtro 1: Eliminar pit laps (distorsionan la distribución de pace)
df_pd = df_pd[
    (df_pd['is_pit_lap'] != 1) &
    (df_pd['is_pit_out_lap'] != 1)
].reset_index(drop=True)

# Filtro 2: Eliminar vueltas anómalas (> 2σ de la mediana por carrera)
lap_median = df_pd.groupby('race_name')['lap_duration'].transform('median')
lap_std    = df_pd.groupby('race_name')['lap_duration'].transform('std')
df_pd = df_pd[
    df_pd['lap_duration'] <= lap_median + 2 * lap_std
].reset_index(drop=True)
```

**Resultado**: De 3331 filas iniciales se retienen **3004 filas** (327 filas excluidas, ~9.8%), correspondientes a pit laps, safety car laps extremos y vueltas de entrada/salida.

### 3.3. Diagnóstico de Nulos

| Feature | % Nulos | Acción |
|---|---|---|
| `speed_per_rpm` | 38.9% | Imputación con mediana |
| `brake_max_lap` | 38.5% | Imputación con mediana |
| `rpm_max_lap` | 38.5% | Imputación con mediana |
| `n_gear_max_lap` | 38.5% | Imputación con mediana |
| `throttle_brake_ratio` | 38.5% | Imputación con mediana |
| `throttle_pct_full` | 38.5% | Imputación con mediana |
| `throttle_mean_lap` | 38.5% | Imputación con mediana |
| `compound_ord` | 29.9% | Imputación con mediana |
| `i1_speed` | 16.8% | Imputación con mediana |
| `st_speed` | 2.2% | Imputación con mediana |
| `duration_sector_1` | 0.5% | Imputación con mediana |

> [!IMPORTANT]
> **Ninguna columna fue eliminada por exceso de nulos** (umbral: >40%). La presencia masiva de nulos en las features de throttle/RPM se debe a que los datos de telemetría granular no están disponibles en todas las vueltas. La imputación con **mediana** es robusta a outliers, apropiada para distribuciones de telemetría sesgadas.

### 3.4. Pipeline de Transformación Final

| Paso | Técnica | Justificación |
|---|---|---|
| **Filtrado de Outliers** | `is_pit_lap != 1`, `lap_duration ≤ μ + 2σ` | Elimina vueltas con dinámicas no representativas del rendimiento en carrera |
| **Diagnóstico de Nulos** | `null_pct > 0.40` → drop column | Conserva todas las features (ninguna supera el 40%) |
| **Imputación** | `SimpleImputer(strategy='median')` | Resistente a outliers extremos de telemetría |
| **Escalado** | `StandardScaler()` | **Obligatorio** para PCA: fuerza media=0 y std=1 por feature |
| **Eliminación no-numérica** | `select_dtypes(include='number')` | `name_acronym` se elimina antes del PCA |

**Matrix final**: **(3004, 24)** — 3004 vueltas × 24 features numéricas escaladas.

---

## 4. Modelo PCA y Varianza Explicada

### 4.1. Resultados del Scree Plot

Se ejecutó un PCA completo sobre la matriz (3004 × 24) para evaluar la distribución de varianza antes de seleccionar el número de componentes:

| PC | Varianza Individual | Varianza Acumulada | Hito |
|---|---|---|---|
| PC1 | **25.1%** | 25.1% | |
| PC2 | **16.0%** | 41.1% | |
| PC3 | **13.4%** | 54.5% | |
| PC4 | **9.8%** | 64.3% | |
| PC5 | **8.4%** | 72.7% | |
| PC6 | **6.0%** | 78.7% | ← **Seleccionado** |
| PC7 | 5.1% | 83.8% | ← 80% |
| PC8 | 3.7% | 87.5% | |
| PC9 | 3.2% | 90.7% | ← 90% |
| PC10 | 2.9% | 93.6% | |

> **→ 7 PCs explican el 80% de la varianza**
> **→ 9 PCs explican el 90% de la varianza**

### 4.2. Decisión de Componentes: N = 6

```python
# Lógica de selección: conservador para maximizar interpretabilidad
N_COMPONENTS = min(n_80, 6)  # = min(7, 6) = 6
```

Se eligieron **6 componentes** (78.7% de varianza) en lugar de 7 (83.8%) por una decisión deliberada de priorizar la **interpretabilidad** sobre la exhaustividad matemática:

- Cada PC debe corresponder a un estado de conducción reconocible.
- PC7 (~5%) aporta señal marginal con costo de interpretabilidad.
- 6 PCs mantiene un ratio de compresión **4:1** (24 → 6 dimensiones).

> [!TIP]
> Este patrón de varianza distribuida (sin un PC dominante con >50%) es **estadísticamente saludable**: indica que el espacio de rendimiento del piloto es genuinamente multidimensional y no está colapsado en un único eje.

---

## 5. Interpretación de los Componentes Principales

El notebook genera un bloque de interpretación automática basado en los **Top 3 loadings** (positivo y negativo) de cada PC:

### PC1 — Dominancia del Sector 3 en el Tiempo de Vuelta (25.1%)

| Dirección | Variable | Loading | Interpretación |
|---|---|---|---|
| **+** | `sector3_pct` | +0.398 | S3 pesa más en la vuelta |
| **+** | `duration_sector_3` | +0.372 | S3 lento en segundos |
| **−** | `sector_balance` | −0.371 | Inverso: equilibrio S1/S3 bajo |
| **−** | `duration_sector_2` | −0.357 | S2 relativamente rápido |
| **−** | `duration_sector_1` | −0.354 | S1 relativamente rápido |

**Etiqueta estratégica**: *Perfil de trazado — Circuito con peso en S3*. Un PC1 alto identifica circuitos o situaciones donde el sector 3 (típicamente de baja velocidad y frenadas) domina el tiempo total de vuelta. Es la firma del trazado, no del estilo de conducción.

---

### PC2 — Agresividad de Aplicación de Potencia (16.0%)

| Dirección | Variable | Loading | Interpretación |
|---|---|---|---|
| **+** | `throttle_brake_ratio` | +0.372 | Ratio gas/freno alto |
| **+** | `throttle_pct_full` | +0.370 | Porcentaje con throttle al 100% |
| **+** | `throttle_mean_lap` | +0.370 | Throttle medio de la vuelta alto |
| **+** | `i2_speed` | +0.359 | Velocidad en punto intermedio 2 alta |
| **+** | `sector1_pct` | +0.255 | S1 pesa más en la vuelta |

**Etiqueta estratégica**: *Intensidad de Aceleración — Push vs. Lift*. Un PC2 alto es la firma de una vuelta de ataque puro: gas al máximo, freno mínimo, velocidades intermedias altas. PC2 bajo corresponde a vueltas de gestión con *lift-and-coast*.

---

### PC3 — Velocidad Punta y Nivel de Frenada (13.4%)

| Dirección | Variable | Loading | Interpretación |
|---|---|---|---|
| **+** | `best_lap_stint` | +0.410 | Mejor vuelta del stint (en segundos, no en posición) |
| **+** | `st_speed` | +0.313 | Velocidad en speed trap alta |
| **+** | `brake_max_lap` | +0.311 | Frenada máxima intensa |
| **−** | `sector1_pct` | −0.307 | S1 pesa menos en la vuelta |
| **+** | `throttle_mean_lap` | +0.295 | Throttle medio alto |

**Etiqueta estratégica**: *Potencia de Motor y Perfil de Alta Velocidad*. PC3 captura circuitos o fases donde la velocidad punta (straight-line speed) es determinante. Los loadings combinados de `st_speed` y `brake_max_lap` son la firma clásica de un circuito de alta velocidad con frenadas tardías.

---

### PC4 — Degradación del Neumático (9.8%)

| Dirección | Variable | Loading | Interpretación |
|---|---|---|---|
| **+** | `lap_vs_best_stint` | +0.473 | Delta respecto a mejor vuelta del stint (degradación) |
| **+** | `lap_duration` | +0.449 | Vuelta lenta en absoluto |
| **−** | `st_speed` | −0.319 | Velocidad punta baja |
| **−** | `speed_per_rpm` | −0.252 | Eficiencia motor baja |
| **+** | `brake_max_lap` | +0.227 | Frenada fuerte |

**Etiqueta estratégica**: *Degradación de Neumático — Tire Cliff*. PC4 es el componente más directo de degradación: cuando el neumático pierde agarre, la vuelta se alarga, la velocidad punta cae y la eficiencia mecánica disminuye. Un PC4 alto indica un stint en su fase final o un neumático en cliff.

---

### PC5 — Velocidad Intermedia y Compuesto (8.4%)

| Dirección | Variable | Loading | Interpretación |
|---|---|---|---|
| **+** | `i2_speed` | +0.385 | Velocidad en punto intermedio 2 alta |
| **+** | `i1_speed` | +0.375 | Velocidad en punto intermedio 1 alta |
| **+** | `compound_ord` | +0.341 | Compuesto más duro |
| **−** | `position` | −0.311 | Posición delantera en carrera |
| **−** | `throttle_brake_ratio` | −0.286 | Ratio gas/freno moderado |

**Etiqueta estratégica**: *Velocidad en Secciones Intermedias + Estrategia de Compuesto*. PC5 une el rendimiento en zonas de velocidad media (chicanes, curvas rápidas) con el tipo de neumático. Un PC5 alto identifica pilotos en posiciones delanteras con compuesto duro, navegando curvas intermedias a alta velocidad.

---

### PC6 — Régimen de Motor (6.0%)

| Dirección | Variable | Loading | Interpretación |
|---|---|---|---|
| **+** | `rpm_max_lap` | **+0.629** | RPM máximo de la vuelta muy alto |
| **−** | `speed_per_rpm` | −0.496 | Eficiencia motor baja (mucho RPM, poca velocidad) |
| **−** | `n_gear_max_lap` | −0.282 | Marcha máxima baja |
| **−** | `lap_vs_best_stint` | −0.273 | Baja degradación |
| **+** | `brake_max_lap` | +0.243 | Frenadas fuertes |

**Etiqueta estratégica**: *Modo Motor — RPM Alto vs. Eficiencia*. PC6 captura la tensión entre rev alto y eficiencia de transmisión. Un PC6 alto indica circuitos de stop-and-go con RPM elevados pero marchas bajas (como Mónaco o Budapest), opuesto a circuitos de alta velocidad donde `speed_per_rpm` es elevado.

> [!NOTE]
> El loading dominante de `rpm_max_lap` (+0.629) es el más alto de todos los componentes, lo que indica que PC6 tiene una señal limpia y muy concentrada en una sola variable.

---

## 6. Heatmap de Loadings

El heatmap generado en el notebook (`loadings_heatmap_v4.png`) presenta la matriz completa de coeficientes PCA (24 variables × 6 componentes) codificada en una escala de color rojo-azul (RdBu_r):

```
RdBu_r: Rojo intenso → loading muy positivo (+1)
         Blanco       → loading neutro (0)
         Azul intenso → loading muy negativo (−1)
```

**Observaciones clave del heatmap**:
- PC1 y PC2 presentan **loadings distribuidos** entre múltiples variables → capturan patrones globales de trazado y estilo.
- PC6 muestra **concentración extrema** en `rpm_max_lap` → captura un fenómeno muy específico.
- Las variables de sector (`sector1_pct`, `sector3_pct`, `sector_balance`) tienen loadings opuestos en PC1 y PC2, confirmando que capturan dimensiones ortogonales del balance del trazado.

---

## 7. Scatter PC1 vs PC2 por Carrera

El scatter plot generado agrupa las 3004 vueltas filtradas en el espacio bidimensional (PC1, PC2), coloreadas por carrera. Es la prueba visual de si el PCA captura variación real de rendimiento:

**Interpretación esperada**:
- Si las carreras forman **nubes separadas** → los PCs capturan diferencias de trazado entre circuitos (efecto de circuito dominante).
- Si las carreras se **superponen** con dispersión interna → los PCs capturan variación dentro de cada carrera (estados de conducción real).
- La presencia de **ambos** patrones indica que PC1 captura el circuito y PC2 captura el comportamiento intra-carrera.

---

## 8. Resultados y Dataset de Salida

Al finalizar, el notebook exporta **`telemetry_pca_v4.parquet`** con la siguiente estructura:

| Columna | Tipo | Descripción |
|---|---|---|
| `race_name` | str | Identificador de carrera |
| `driver_number` | f64 | Número de piloto |
| `lap_number` | f64 | Número de vuelta |
| `team_name` | str | Equipo |
| `PC1` | f64 | Score componente 1 (25.1% varianza) |
| `PC2` | f64 | Score componente 2 (16.0% varianza) |
| `PC3` | f64 | Score componente 3 (13.4% varianza) |
| `PC4` | f64 | Score componente 4 (9.8% varianza) |
| `PC5` | f64 | Score componente 5 (8.4% varianza) |
| `PC6` | f64 | Score componente 6 (6.0% varianza) |

> [!IMPORTANT]
> Este dataset es el **input directo** del notebook `kmeans_telemetry_v2.ipynb`, donde los 6 PCs son los features de clustering. La separación en espacio PCA permite que el K-Means encuentre agrupaciones basadas en la estructura latente del rendimiento, no en correlaciones espurias entre variables crudas.

---

## 9. Conclusiones y Próximos Pasos

### 9.1. Validación del Pipeline

| Criterio | Resultado | Evaluación |
|---|---|---|
| Ratio filas/features | 3004 / 24 = **125:1** | ✅ Muy por encima del mínimo de 50:1 |
| Varianza retenida | **78.7%** con 6 PCs | ✅ Conservadora pero interpretable |
| Multicolinealidad eliminada | PCs ortogonales por construcción | ✅ Garantizado matemáticamente |
| Interpretabilidad | Cada PC mapea a un concepto de F1 | ✅ Todos los PCs tienen etiqueta estratégica |
| Nulos manejados | Mediana robusta, ninguna columna eliminada | ✅ Sin pérdida de features |

### 9.2. Mapa Conceptual de los 6 Componentes

```
PC1 (25.1%) — Perfil de Trazado: Peso del Sector 3
PC2 (16.0%) — Agresividad de Aplicación de Potencia (Push vs. Lift)
PC3 (13.4%) — Velocidad Punta y Nivel de Frenada
PC4 ( 9.8%) — Degradación de Neumático (Tire Cliff)
PC5 ( 8.4%) — Velocidad Intermedia + Estrategia de Compuesto
PC6 ( 6.0%) — Régimen de Motor (RPM Alto vs. Eficiencia)
```

### 9.3. Recomendaciones para el Clustering (K-Means V2)

1. **Usar los 6 PCs** como input: Varianza suficiente (78.7%) con interpretabilidad máxima.
2. **Probar k = 3 a 6**: Los 4 estados esperados (Push, Energy Saving, Tyre Mgmt, Pit Out) sugieren k=4 como punto de partida.
3. **Verificar con Silhouette Score**: Validar que los clusters no sean artefactos del circuito dominando PC1.
4. **Cruzar clusters con `lap_vs_best_stint`**: El cluster de Tyre Management debe tener los valores más altos.
5. **Analizar PC6 por separado**: El loading dominante de `rpm_max_lap` sugiere que PC6 puede separar circuitos callejeros de circuitos permanentes.

---

*Análisis basado en `PCA_v4.ipynb` y dataset `telemetry_features_v4.parquet`*
*Temporada 2026 — Carreras: Australia, China, Japón, Estados Unidos*
