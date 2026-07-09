"""
Auditoria de sesgo del ranking de pit stops (Capa 2).

Version actualizada para el target de 7 acciones:
    wait_laps 0..5 -> parar tras esperar w vueltas
    wait_laps 6    -> NO_PIT / STAY_OUT (no parar en la ventana de 5 vueltas)

Con NO_PIT como accion explicita, el baseline trivial relevante deja de ser
"siempre wait_laps=0" y pasa a ser "siempre NO_PIT". La metrica de exito honesta
es la accuracy en los grupos donde la decision optima real fue una parada
(offset 0..5), es decir, donde el recomendador debe aportar valor.

Ubicacion prevista en el repo: project/src/models/audit_ranking_bias.py
Ejecucion:  python src/models/audit_ranking_bias.py   (desde project/)
Salida:     reports/ranking_system/ranking_bias_audit.md
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Rutas ---------------------------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent          # src/models
PROJECT_DIR = FILE_DIR.parent.parent                # project
DATA_PATH = PROJECT_DIR / "data" / "processed" / "recommendation" / "pit_decision_candidates_v1.parquet"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "ranking_layer2_model.pkl"
REPORT_DIR = PROJECT_DIR / "reports" / "ranking_system"
REPORT_PATH = REPORT_DIR / "ranking_bias_audit.md"

GROUP_KEYS = ["race_name", "driver_number", "lap_number"]
NO_PIT = 6

# Lista de features de la Capa 2 (identica a la usada en inferencia).
# Se prioriza el orden con el que fue entrenado el modelo si esta disponible.
FEATURES_RANK = [
    "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3",
    "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position",
    "is_top10", "laps_remaining", "race_pct_complete",
    "gap_ahead", "gap_behind", "wait_laps", "predicted_cost_of_staying",
]


def action_name(w):
    w = int(w)
    if w == NO_PIT:
        return "NO_PIT"
    if w == 0:
        return "Parar ahora (0)"
    return f"Esperar {w}"


def dist_table(series, n_total):
    """Construye una tabla markdown de distribucion sobre las acciones 0..6."""
    counts = series.value_counts().reindex(range(NO_PIT + 1), fill_value=0)
    lines = ["| accion | wait_laps | n | % |", "|---|---|---|---|"]
    for w, n in counts.items():
        pct = (n / n_total * 100) if n_total else 0.0
        lines.append(f"| {action_name(w)} | {w} | {int(n)} | {pct:.2f}% |")
    return "\n".join(lines)


def main():
    print("\n" + "=" * 70)
    print("Auditoria de sesgo del ranking (Capa 2) - target de 7 acciones")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"Error: no se encuentra el dataset {DATA_PATH}")
        return
    if not MODEL_PATH.exists():
        print(f"Error: no se encuentra el modelo {MODEL_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Orden de features: usa el del modelo si lo expone, de lo contrario la lista fija.
    feats = list(getattr(model, "feature_names_in_", FEATURES_RANK))
    missing = [c for c in feats if c not in df.columns]
    if missing:
        print(f"Error: faltan columnas de features en el dataset: {missing}")
        return

    # --- Prediccion del ranker sobre todos los candidatos ---------------------
    X = df[feats].copy()
    for c in feats:
        X[c] = X[c].fillna(0.0)
    df["pred_score"] = model.predict(X)

    # --- Mejor accion real y predicha por grupo -------------------------------
    idx_real = df.groupby(GROUP_KEYS)["success_score_label"].idxmax()
    idx_pred = df.groupby(GROUP_KEYS)["pred_score"].idxmax()

    real = (
        df.loc[idx_real, GROUP_KEYS + ["wait_laps"]]
        .rename(columns={"wait_laps": "wait_real"})
        .set_index(GROUP_KEYS)
    )
    pred = (
        df.loc[idx_pred, GROUP_KEYS + ["wait_laps"]]
        .rename(columns={"wait_laps": "wait_pred"})
        .set_index(GROUP_KEYS)
    )
    comp = real.join(pred)
    comp["wait_real"] = comp["wait_real"].astype(int)
    comp["wait_pred"] = comp["wait_pred"].astype(int)

    n_groups = len(comp)

    # --- Metricas -------------------------------------------------------------
    acc = (comp["wait_real"] == comp["wait_pred"]).mean()
    baseline_no_pit = (comp["wait_real"] == NO_PIT).mean()
    baseline_zero = (comp["wait_real"] == 0).mean()

    # Decision binaria: parar (0..5) vs no parar (6)
    is_pit_real = comp["wait_real"].between(0, 5)
    is_pit_pred = comp["wait_pred"].between(0, 5)
    binary_acc = (is_pit_real == is_pit_pred).mean()

    # Grupos con parada optima real: donde el recomendador debe aportar valor
    minority = comp[is_pit_real]
    n_min = len(minority)
    acc_min_exact = (minority["wait_real"] == minority["wait_pred"]).mean() if n_min else 0.0
    acc_min_binary = minority["wait_pred"].between(0, 5).mean() if n_min else 0.0

    # --- Interpretacion condicional ------------------------------------------
    if acc > baseline_no_pit:
        margin = (acc - baseline_no_pit) * 100
        interp1 = (
            f"La accuracy global ({acc:.4f}) supera al baseline trivial "
            f"'siempre NO_PIT' ({baseline_no_pit:.4f}) por {margin:.2f} puntos "
            "porcentuales."
        )
    else:
        interp1 = (
            f"La accuracy global ({acc:.4f}) **no supera** al baseline trivial "
            f"'siempre NO_PIT' ({baseline_no_pit:.4f}); en un target tan "
            "desbalanceado, la accuracy global sigue siendo poco informativa."
        )

    interp2 = (
        f"En los {n_min} grupos donde la decision optima real fue una parada "
        f"(offset 0-5), el modelo detecta correctamente la necesidad de parar "
        f"(decision binaria parar/no parar) en el {acc_min_binary:.2%} de los "
        f"casos y acierta el offset exacto en el {acc_min_exact:.2%}. La decision "
        "binaria es la metrica principal de utilidad del recomendador; el offset "
        "exacto es una exigencia mas estricta."
    )

    interp3 = (
        "Con NO_PIT como accion explicita, el candidato wait_laps=0 deja de recibir "
        "la mejor etiqueta por defecto en las vueltas sin ventana de parada real, "
        "de modo que 'quedarse fuera' se aprende como una decision propia y no como "
        "un artefacto del etiquetado."
    )

    # --- Construccion del reporte --------------------------------------------
    report = f"""# Auditoria de sesgo del ranking de pit stops (Capa 2)

Grupos de decision evaluados (carrera, piloto, vuelta): **{n_groups}**
Modelo: `models/ranking_layer2_model.pkl` (features: {len(feats)})
Dataset: `data/processed/recommendation/pit_decision_candidates_v1.parquet`
Acciones: `wait_laps` 0-5 (parar tras esperar w vueltas) + `wait_laps=6` (NO_PIT / STAY_OUT)

## Distribucion de la mejor accion real (ground truth)

{dist_table(comp["wait_real"], n_groups)}

## Distribucion de la mejor accion predicha

{dist_table(comp["wait_pred"], n_groups)}

## Metricas

| Metrica | Valor |
|---|---|
| Accuracy global (accion exacta) | {acc:.4f} |
| Baseline "siempre NO_PIT (6)" | {baseline_no_pit:.4f} |
| Baseline "siempre parar ya (0)" (referencia historica) | {baseline_zero:.4f} |
| Accuracy de decision binaria (parar vs no parar) | {binary_acc:.4f} |
| Grupos con parada optima real (optimo != 6) | {n_min} |
| Accuracy binaria en esos grupos (detecta que hay que parar) | {acc_min_binary:.4f} |
| Accuracy exacta en esos grupos (offset correcto) | {acc_min_exact:.4f} |

## Interpretacion

{interp1}

{interp2}

{interp3}

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
"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    # --- Resumen por consola --------------------------------------------------
    print(f"Grupos evaluados: {n_groups}")
    print(f"Accuracy global (accion exacta):        {acc:.4f}")
    print(f"Baseline 'siempre NO_PIT':              {baseline_no_pit:.4f}")
    print(f"Accuracy decision binaria:              {binary_acc:.4f}")
    print(f"Grupos con parada optima real:          {n_min}")
    print(f"  - accuracy binaria (detecta parar):   {acc_min_binary:.4f}")
    print(f"  - accuracy exacta (offset correcto):  {acc_min_exact:.4f}")
    print(f"\nReporte escrito en: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()