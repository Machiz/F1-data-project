import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "recommendation" / "pit_decision_candidates_v1.parquet"
FEATURES_DIR = PROJECT_DIR / "data" / "features"

def main():
    print("--- Integrando Modelo de Regresión Capa 1 a los Candidatos (Puente) ---")
    df = pd.read_parquet(DATA_PATH)
    
    # Cargar modelo entrenado
    model_path = FEATURES_DIR / "regression_layer1_model.pkl"
    if not model_path.exists():
        print(f"Error: No se encuentra el modelo {model_path.name}. Ejecute primero train_regression_layer1.py")
        return
        
    model = joblib.load(model_path)
    
    features = [
        "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
        "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
        "is_top10", "laps_remaining", "race_pct_complete", 
        "gap_ahead", "gap_behind", "wait_laps"
    ]
    
    # Imputación sobre df general
    for col in features:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        
    # Predecir ritmo esperado de permanencia
    df["predicted_future_pace"] = model.predict(df[features])
    
    # predicted_cost_of_staying = wait_laps * (predicted_future_pace - current_lap_duration)
    df["predicted_cost_of_staying"] = df["wait_laps"] * (df["predicted_future_pace"] - df["lap_duration"])
    
    # Sobrescribir dataset
    df.to_parquet(DATA_PATH, index=False)
    print("Dataset de candidatos actualizado con predicted_cost_of_staying exitosamente.")

if __name__ == "__main__":
    main()
