# Auditoria de sesgo del ranking de pit stops (Capa 2)

Grupos de decision evaluados (carrera, piloto, vuelta): **3331**
Modelo: `models/ranking_layer2_model.pkl` (features: 21)
Dataset: `data/processed/recommendation/pit_decision_candidates_v1.parquet`
Acciones: `wait_laps` 0-5 (parar tras esperar w vueltas) + `wait_laps=6` (NO_PIT / STAY_OUT)

## Distribucion de la mejor accion real (ground truth)

| accion | wait_laps | n | % |
|---|---|---|---|
| Parar ahora (0) | 0 | 139 | 4.17% |
| Esperar 1 | 1 | 61 | 1.83% |
| Esperar 2 | 2 | 42 | 1.26% |
| Esperar 3 | 3 | 42 | 1.26% |
| Esperar 4 | 4 | 42 | 1.26% |
| Esperar 5 | 5 | 41 | 1.23% |
| NO_PIT | 6 | 2964 | 88.98% |

## Distribucion de la mejor accion predicha

| accion | wait_laps | n | % |
|---|---|---|---|
| Parar ahora (0) | 0 | 36 | 1.08% |
| Esperar 1 | 1 | 22 | 0.66% |
| Esperar 2 | 2 | 26 | 0.78% |
| Esperar 3 | 3 | 40 | 1.20% |
| Esperar 4 | 4 | 32 | 0.96% |
| Esperar 5 | 5 | 47 | 1.41% |
| NO_PIT | 6 | 3128 | 93.91% |

## Metricas

| Metrica | Valor |
|---|---|
| Accuracy global (accion exacta) | 0.9093 |
| Baseline "siempre NO_PIT (6)" | 0.8898 |
| Baseline "siempre parar ya (0)" (referencia historica) | 0.0417 |
| Accuracy de decision binaria (parar vs no parar) | 0.9147 |
| Grupos con parada optima real (optimo != 6) | 367 |
| Accuracy binaria en esos grupos (detecta que hay que parar) | 0.3896 |
| Accuracy exacta en esos grupos (offset correcto) | 0.3406 |

## Interpretacion

La accuracy global (0.9093) supera al baseline trivial 'siempre NO_PIT' (0.8898) por 1.95 puntos porcentuales.

En los 367 grupos donde la decision optima real fue una parada (offset 0-5), el modelo detecta correctamente la necesidad de parar (decision binaria parar/no parar) en el 38.96% de los casos y acierta el offset exacto en el 34.06%. La decision binaria es la metrica principal de utilidad del recomendador; el offset exacto es una exigencia mas estricta.

Con NO_PIT como accion explicita, el candidato wait_laps=0 deja de recibir la mejor etiqueta por defecto en las vueltas sin ventana de parada real, de modo que 'quedarse fuera' se aprende como una decision propia y no como un artefacto del etiquetado.

## Formulacion del target (corregida)

En `src/features/f1_recommender_pipeline.py`, cada grupo (carrera, piloto, vuelta)
genera siete candidatos. Si hubo una parada real en `lap + w` para algun `w` en
0-5, ese candidato recibe su `success_score`; el resto de offsets y NO_PIT reciben
`-2.0`. Si no hubo parada real en la ventana de 5 vueltas, NO_PIT (`wait_laps=6`)
recibe la etiqueta ganadora (`0.0`) y los offsets 0-5 reciben `-2.0`. Asi, el
umbral neutro de `0.0` para NO_PIT hace que el modelo prefiera quedarse fuera
antes que ejecutar una parada cuyo score esperado sea negativo.

## Limitaciones y trabajo futuro

El ground truth se deriva del propio esquema de etiquetado; la accuracy exacta
del offset esta acotada por la calidad del `success_score` proxy. La linea PPO
modela nativamente la decision secuencial de parada (incluida la accion de
quedarse fuera) evaluando la recompensa de la carrera simulada, y constituye la
via de mayor rigor una vez entrenado el agente.
