import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "recommendation" / "pit_decision_candidates_v1.parquet"
MODELS_DIR = PROJECT_DIR / "models"

def main():
    print("--- Integrando Modelo de Regresión Capa 1 a los Candidatos (Puente) ---")
    df = pd.read_parquet(DATA_PATH)
    
    # Cargar modelo entrenado
    model_path = MODELS_DIR / "regression_layer1_model.pkl"
    if not model_path.exists():
        print(f"Error: No se encuentra el modelo {model_path.name}. Ejecute primero train_regression_layer1.py")
        return
        
    model = joblib.load(model_path)
    
    # Cargar la lista de features entrenadas (que incluyen dummy variables de race_name y driver_number)
    feature_list_path = MODELS_DIR / "regression_features.joblib"
    if not feature_list_path.exists():
        print(f"Error: No se encuentra la lista de features {feature_list_path.name}. Ejecute primero train_regression_layer1.py")
        return
        
    features = joblib.load(feature_list_path)
    
    # Preprocesamiento: Generar variables dummy para race_name para alinearse con el modelo entrenado
    df_processed = pd.get_dummies(df, columns=["race_name"])
    
    # Asegurar que todas las columnas esperadas estén presentes en el df procesado (rellenar con 0 si faltan)
    for col in features:
        if col not in df_processed.columns:
            df_processed[col] = 0
            
    # Imputación de nulos sobre las columnas correspondientes
    for col in features:
        median_val = df_processed[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_processed[col] = df_processed[col].fillna(median_val)
        
    # Predecir ritmo esperado de permanencia
    df["predicted_future_pace"] = model.predict(df_processed[features])
    
    # predicted_cost_of_staying = wait_laps * (predicted_future_pace - current_lap_duration)
    df["predicted_cost_of_staying"] = df["wait_laps"] * (df["predicted_future_pace"] - df["lap_duration"])
    
    # Sobrescribir dataset
    df.to_parquet(DATA_PATH, index=False)
    print("Dataset de candidatos actualizado con predicted_cost_of_staying exitosamente.")

if __name__ == "__main__":
    main()
