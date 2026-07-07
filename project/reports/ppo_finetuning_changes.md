# PPO Pit Strategy Fine Tuning Changes

## Objetivo

El objetivo del ajuste fue mejorar el agente PPO para que recomiende una ventana de pit stop mas realista. El modelo inicial tenia buen cumplimiento regulatorio, pero tendia a realizar demasiadas paradas y a concentrar los pit stops demasiado temprano.

Resultados observados durante el ajuste:

| Iteracion | Mean Reward | Mean Position | Mean Pit Stops | Reg Violations | Pit Lap Distribution |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Baseline reportado | -96.49 | 14.64 | 3.36 | 0.00% | No reportado |
| Reward mas estricto | -209.11 | 14.98 | 2.26 | 0.00% | No reportado |
| Balance intermedio | -173.51 | 14.86 | 2.32 | 0.00% | median=12, p25=9, p75=14 |

La lectura principal es que el PPO puede cumplir la regla de dos compuestos, pero si el simulador de ritmo premia demasiado el neumatico fresco, el agente aprende a parar temprano para explotar esa ventaja.

## Cambios en el entorno RL

Archivo: `project/src/models/f1_pit_env.py`

### 1. Penalizacion por paradas extra

Se aumento el costo de paradas por encima de dos stops para que el agente no convierta el problema en una politica de neumaticos siempre frescos.

La penalizacion final por mas de dos paradas se mantiene fuerte:

```python
if final_pits > 2:
    reward -= (final_pits - 2) * 500.0
```

### 2. Stints minimos mas realistas

Se agrego una penalizacion por parar antes de una vida util minima del stint:

```python
min_useful_stint = {1: 16, 2: 18, 3: 22}
```

Esto busca evitar ventanas artificiales como vueltas 9-14 cuando el neumatico todavia no deberia justificar una parada.

### 3. Control de primera y segunda parada temprana

Se agrego una penalizacion progresiva cuando:

- la primera parada ocurre antes del 30% de carrera;
- la segunda parada ocurre antes del 58% de carrera.

El objetivo es empujar la distribucion de pit laps hacia una ventana mas tardia sin bloquear por completo undercuts legitimos.

### 4. Bonus regulatorio mas controlado

El bonus inmediato por cumplir la regla de dos compuestos se mantuvo pequeno. La recompensa principal por cumplir la regla se paga al final, para que el agente no aprenda a parar temprano solo por recibir el bonus.

## Cambio en el simulador de ritmo

Archivo: `project/src/models/f1_pit_env.py`

Se reemplazo el `DecisionTreeRegressor` por un `RandomForestRegressor` compacto:

```python
RandomForestRegressor(
    n_estimators=40,
    max_depth=8,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

Motivo:

- un arbol unico genera predicciones bruscas;
- PPO puede explotar saltos artificiales de ritmo con neumatico fresco;
- Random Forest suaviza las predicciones al promediar varios arboles;
- `min_samples_leaf=20` reduce predicciones extremas basadas en pocos ejemplos.

Este cambio hace que el entorno sea mas realista, aunque puede reducir la velocidad de entrenamiento frente al arbol unico.

## Cambios en entrenamiento PPO

Archivo: `project/src/models/train_ppo_rl.py`

El script ahora hace fine tuning sobre un modelo existente si encuentra:

- `project/data/features/best_model.zip`
- `project/data/features/ppo_f1_pit_model.zip`

Tambien se agrego guardado explicito del modelo ajustado:

```text
project/data/features/ppo_f1_pit_model_finetuned.zip
```

Parametros actuales de fine tuning:

```python
learning_rate = 8e-5
ent_coef = 0.0015
gamma = 0.997
gae_lambda = 0.96
clip_range = 0.15
```

Tambien se agrego reporte de distribucion de pit laps del modelo:

```text
median, p25, p75, samples
```

Esta metrica es clave porque el objetivo del proyecto no es solo maximizar reward, sino identificar el lap/window optimo de parada.

## Cambios en Optuna

Archivo: `project/src/models/tune_ppo_optuna.py`

La funcion objetivo ya no maximiza solo recompensa media. Ahora usa una puntuacion estrategica que penaliza:

- mas de dos pit stops;
- violaciones regulatorias;
- posicion media demasiado mala.

Esto evita seleccionar politicas con buen reward pero comportamiento estrategico poco realista.

## Como repetir el experimento

Como se cambio el simulador base de ritmo, se recomienda entrenar desde cero:

```powershell
$env:F1_PPO_RESET="1"
python project\src\models\train_ppo_rl.py
```

Luego, si se quiere volver al comportamiento normal de fine tuning:

```powershell
Remove-Item Env:\F1_PPO_RESET
```

## Criterios de exito esperados

El siguiente entrenamiento deberia buscar:

| Metrica | Objetivo |
| :--- | :--- |
| Mean Pit Stops | 1.8 a 2.2 |
| Reg Violations | 0.00% |
| Pit Lap Median | >= 16 |
| Pit Lap P25 | >= 13 |
| Model Reward | Mejor que REAL |
| Mean Position | Cercana o mejor que REAL |

Si el modelo sigue parando demasiado temprano, el siguiente cambio recomendado es separar la accion en:

```text
pit / no pit
```

y elegir el compuesto con una regla externa. Eso reduce la capacidad del PPO de explotar cambios de compuesto como una accion tactica excesivamente flexible.
