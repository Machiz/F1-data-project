import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "recommendation" / "pit_decision_candidates_v1.parquet"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("--- Entrenando Modelo de Ranking Capa 2 (Random Forest Point-wise) ---")
    df = pd.read_parquet(DATA_PATH)
    
    features = [
        "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
        "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
        "is_top10", "laps_remaining", "race_pct_complete", 
        "gap_ahead", "gap_behind", "wait_laps", "predicted_cost_of_staying",
        "pit_gap_ahead", "pit_gap_behind", "delta_time_loss",
        "compound_SOFT", "compound_MEDIUM", "compound_HARD"
    ]
    
    # Imputación
    for col in features:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        
    X = df[features]
    y = df["success_score_label"]
    
    # Calcular pesos de muestra para mitigar el desbalance de clases extremo
    # Definimos como clase positiva (1) los candidatos que tienen una etiqueta superior al penalizador neutro (-2.0)
    is_positive = (y > -2.0).astype(int)
    n_pos = (is_positive == 1).sum()
    n_neg = (is_positive == 0).sum()
    
    if n_pos > 0:
        weight_ratio = n_neg / n_pos
        sample_weights = np.where(is_positive == 1, weight_ratio, 1.0)
        print(f"Desbalance detectado: Negativos={n_neg}, Positivos={n_pos} | Ratio de peso aplicado={weight_ratio:.4f}")
    else:
        sample_weights = np.ones(len(y))
        print("No se detectaron clases positivas, usando pesos uniformes.")
    
    # Entrenar RandomForestRegressor (seleccionado como mejor modelo de ranking)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y, sample_weight=sample_weights)
    
    # Guardar modelo
    output_path = MODELS_DIR / "ranking_layer2_model.pkl"
    joblib.dump(model, output_path)
    print(f"Capa 2 completada. Modelo guardado en: {output_path}")

if __name__ == "__main__":
    main()
