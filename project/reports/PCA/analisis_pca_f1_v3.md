# Análisis del PCA — Eventos Tácticos F1 V3

## Resumen del dataset

| Parámetro | Valor |
|---|---|
| Fuente | `tactical_events_v3.parquet` |
| Variables de entrada | ~515 |
| Variables tras limpieza | ~400 |
| Componentes calculados | 15 |
| Tipos de evento | On-Track Overtake · Undercut · Overcut |
| Carreras | Australia, China, Japón, Estados Unidos (2026) |

---

## Varianza explicada

| PC | Varianza individual | Varianza acumulada |
|---|---|---|
| PC1 | ~26% | ~26% |
| PC2 | ~14% | ~40% |
| PC3 | ~10% | ~50% |
| PC4 | ~8% | ~58% |
| PC5 | ~6% | ~64% |
| PC6 | ~5% | ~69% |
| **PC7** | ~4% | **~73%** |
| PC8 | ~4% | ~77% |
| **PC9–11** | ~3% c/u | **~80–90%** |

> Para alcanzar el **80% de varianza** se necesitan ~7 componentes.  
> Para el **90%** se necesitan ~11 componentes.

Este patrón es saludable: el espacio táctico no está dominado por un único factor sino que tiene múltiples dimensiones reales.

---

## Interpretación de cada componente

### PC1 — Diferencial de rendimiento en pista (~26%)

Captura el contraste de ritmo directo entre atacante y defensor. Las variables con mayor loading son las que miden ventaja técnica neta.

**Variables dominantes (estimado):**

| Variable | Dirección | Interpretación |
|---|---|---|
| `delta_lap_time_mean` | + | Mayor diferencia de tiempos → ataque más sólido |
| `att_lap_time_mean` | − | Atacante más rápido → PC1 alto |
| `def_lap_time_mean` | + | Defensor más lento → facilita maniobra |
| `delta_grip` | + | Ventaja de compuesto del atacante |
| `position_gap_mean` | + | Brecha en pista al inicio del evento |

Un **PC1 alto** indica superioridad técnica clara del atacante. Un **PC1 bajo** indica maniobra táctica sin ventaja de ritmo.

---

### PC2 — Estrategia de pit (~14%)

Separa las maniobras puramente en pista de las tácticas de undercut/overcut. Captura el timing de parada y la fase del stint.

**Variables dominantes (estimado):**

| Variable | Dirección | Interpretación |
|---|---|---|
| `pit_delta_attacker` | + | Tiempo de pit del atacante |
| `pit_delta_defender` | − | Tiempo de pit del defensor |
| `stint_progress_att` | + | Qué tan avanzado está el stint |
| `def_tyre_age_mean` | + | Neumáticos más viejos del defensor |
| `att_grip` | + | Compuesto fresco del atacante |

El **PC2 distingue Undercut/Overcut de On-Track Overtakes**: los eventos de pit strategy deberían agruparse en valores altos de PC2.

---

### PC3 — Contexto de circuito y carrera (~10%)

Captura variación de carrera a carrera: tipo de trazado, temperatura, disponibilidad de DRS y fracción de vuelta.

**Variables dominantes (estimado):**

| Variable | Dirección | Interpretación |
|---|---|---|
| `race_lap_fraction` | + | Momento de la carrera |
| `circuit_type_enc` | − | Tipo de trazado (callejero vs. permanente) |
| `drs_zone_att` | + | DRS disponible para el atacante |
| `weather_temp_mean` | − | Temperatura de pista |
| `att_speed_trap_mean` | + | Velocidad punta (indica rectitud del trazado) |

---

### PC4 — Incidentes externos y condiciones especiales (~8%)

Recoge la influencia de safety car, VSC y condiciones climáticas extremas sobre la táctica.

**Variables dominantes (estimado):**

| Variable | Dirección | Interpretación |
|---|---|---|
| `safety_car_flag` | + | Presencia de Safety Car |
| `vsc_flag` | + | Virtual Safety Car activo |
| `weather_temp_mean` | − | Temperatura baja (condiciones mixtas) |
| `pos_change` | + | Cambio de posición efectivo |

---

## Mapa táctico (PC1 vs PC2)

El scatter de PC1 vs PC2 es el test más directo de si el PCA captura la diferencia táctica:

```
PC2 (pit strategy)
  ▲
  │  [Undercut]   [Overcut]
  │      ●●●         ▲▲▲
  │    ●●●●●●       ▲▲▲▲▲
  │
  │
  │              [On-Track Overtake]
  │                   ■■■■■
  │                  ■■■■■■■
  └──────────────────────────────▶ PC1 (rendimiento)
```

- **On-Track Overtakes** deberían concentrarse en PC1 alto (ventaja de ritmo clara).
- **Undercuts/Overcuts** deberían agruparse en PC2 alto (lógica de pit, no de ritmo puro).
- Si hay **solapamiento excesivo**, las variables de tiempos de vuelta están dominando sobre las de pit strategy en el espacio de features.

---

## Validación del pipeline

| Paso | Decisión | Valoración |
|---|---|---|
| Eliminar cols >50% nulos | ✅ Correcto | Evita imputation masiva que sesga PCA |
| Eliminar varianza cero | ✅ Correcto | Columnas constantes no aportan señal |
| Imputar con mediana | ✅ Robusto | Resistente a outliers en datos de telemetría |
| `StandardScaler` antes de PCA | ✅ Obligatorio | PCA es sensible a escala |
| 515 vars (V3) vs 197 (V2) | ⚠️ Mayor resolución, más ruido potencial | Verificar que PC1 no esté dominado por una variable ruidosa única |

---

## Recomendaciones para el clustering

1. **Usar 7–11 PCs** como input del clustering (80–90% varianza). Más componentes introducen ruido.
2. **Probar K-Means, DBSCAN y clustering jerárquico** — los eventos F1 pueden tener densidades irregulares.
3. **Verificar que los clusters no sean solo "carrera"**: si el biplot muestra que `race_name` domina la separación, los PCs capturan contexto, no táctica.
4. **Cruzar clusters con `pos_change`** para identificar qué patrones tácticos son más efectivos.
5. **Revisar el biplot**: si los vectores de Undercut/Overcut apuntan en PC2 y los de On-Track en PC1, el PCA está funcionando correctamente.

---

## Código sugerido para validación

```python
# Ver los loadings reales de PC1 a PC4
loadings[['PC1','PC2','PC3','PC4']].abs().sort_values('PC1', ascending=False).head(15)

# Verificar que ninguna variable domina PC1 de forma sospechosa
assert loadings['PC1'].abs().max() < 0.6, "Una variable domina PC1 — posible feature ruidosa"

# Preparar scores para clustering (7 PCs)
X_cluster = df.filter(like='PC').iloc[:, :7].values
```

---

*Análisis basado en `PCA_V3.ipynb` · Dataset `tactical_events_v3.parquet` · Mayo 2026*
