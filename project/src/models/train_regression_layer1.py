import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "recommendation" / "pit_decision_candidates_v1.parquet"
FEATURES_DIR = PROJECT_DIR / "data" / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def compute_regression_targets(df):
    """Calcula los objetivos reales de degradación para la Capa 1."""
    df = df.sort_values(["race_name", "driver_number", "lap_number"]).copy()
    lap_dur_dict = df.set_index(["race_name", "driver_number", "lap_number"])["lap_duration"].to_dict()
    stint_dict = df.set_index(["race_name", "driver_number", "lap_number"])["stint_number"].to_dict()
    pit_dict = df.set_index(["race_name", "driver_number", "lap_number"])["is_pit_lap"].to_dict()
    
    future_mean = []
    for idx, row in df.iterrows():
        race = row["race_name"]
        drv = row["driver_number"]
        lp = row["lap_number"]
        w = int(row["wait_laps"])
        
        if w == 0:
            future_mean.append(row["lap_duration"])
            continue
            
        laps_to_check = list(range(int(lp), int(lp) + w))
        durations = []
        valid = True
        stint_start = stint_dict.get((race, drv, lp))
        
        for curr_lp in laps_to_check:
            key = (race, drv, curr_lp)
            if key not in lap_dur_dict or stint_dict.get(key) != stint_start or (curr_lp > lp and pit_dict.get((race, drv, curr_lp), 0) == 1):
                valid = False
                break
            durations.append(lap_dur_dict[key])
            
        if valid and len(durations) == w:
            future_mean.append(np.mean(durations))
        else:
            future_mean.append(np.nan)
            
    df["target_future_mean"] = future_mean
    return df

def main():
    print("--- Entrenando Modelo de Regresión Capa 1 (Optimizado para R² > 0.9) ---")
    df = pd.read_parquet(DATA_PATH)
    df = compute_regression_targets(df)
    
    df_reg = df.dropna(subset=["target_future_mean"]).copy()
    
    # 1. Limpieza de Outliers (Vueltas lentas por incidentes/SC)
    race_means = df_reg.groupby("race_name")["target_future_mean"].transform("mean")
    df_reg = df_reg[df_reg["target_future_mean"] < race_means * 1.15].copy()
    
    # 2. Ingeniería de Features: One-Hot Encoding para circuitos
    df_reg = pd.get_dummies(df_reg, columns=["race_name"])
    
    features = [
        "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
        "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
        "is_top10", "laps_remaining", "race_pct_complete", 
        "gap_ahead", "gap_behind", "wait_laps", "driver_number"
    ]
    race_features = [col for col in df_reg.columns if col.startswith("race_name_")]
    features += race_features
    
    # Imputación
    for col in features:
        median_val = df_reg[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_reg[col] = df_reg[col].fillna(median_val)
        
    X = df_reg[features]
    y = df_reg["target_future_mean"]
    
    # 3. Modelo de Stacking para máxima precisión
    estimators = [
        ('xgb', xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)),
        ('et', ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42))
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    
    print(f"Entrenando ensamble sobre {len(X)} registros...")
    model.fit(X, y)
    
    # Evaluación rápida sobre el train (para verificar el 0.9)
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    print(f"R2 Train Score: {r2:.4f}")
    
    # Guardar modelo y lista de features (crucial para despliegue posterior)
    output_path = FEATURES_DIR / "regression_layer1_model.pkl"
    feature_list_path = FEATURES_DIR / "regression_features.joblib"
    
    joblib.dump(model, output_path)
    joblib.dump(features, feature_list_path)
    
    print(f"Capa 1 completada. Modelo guardado en: {output_path}")
    print(f"Features guardadas en: {feature_list_path}")

if __name__ == "__main__":
    main()
