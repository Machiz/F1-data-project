import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "recommendation" / "pit_decision_candidates_v1.parquet"
FEATURES_DIR = PROJECT_DIR / "data" / "features"

def main():
    print("--- Entrenando Modelo de Ranking Capa 2 (Random Forest Point-wise) ---")
    df = pd.read_parquet(DATA_PATH)
    
    features = [
        "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
        "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
        "is_top10", "laps_remaining", "race_pct_complete", 
        "gap_ahead", "gap_behind", "wait_laps", "predicted_cost_of_staying"
    ]
    
    # Imputación
    for col in features:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        
    X = df[features]
    y = df["success_score_label"]
    
    # Entrenar RandomForestRegressor (seleccionado como mejor modelo de ranking)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)
    
    # Guardar modelo
    output_path = FEATURES_DIR / "ranking_layer2_model.pkl"
    joblib.dump(model, output_path)
    print(f"Capa 2 completada. Modelo guardado en: {output_path}")

if __name__ == "__main__":
    main()
