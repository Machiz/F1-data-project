import os
import glob
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths config
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "recommendation" / "pit_decision_candidates_v1.parquet"
MODEL_PATH = PROJECT_DIR / "models" / "ranking_layer2_model.pkl"
RAW_DIR = PROJECT_DIR / "data" / "raw"

class ChatbotEngine:
    def __init__(self):
        self.df = None
        self.model = None
        self.driver_to_acronym = {}
        self.acronym_to_driver = {}
        self.features = [
            "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
            "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
            "is_top10", "laps_remaining", "race_pct_complete", 
            "gap_ahead", "gap_behind", "wait_laps", "predicted_cost_of_staying"
        ]
        
    def load_resources(self):
        """Loads Parquet candidates, trained model, and scans drivers metadata."""
        print("[INFO] Cargando base de datos de carrera...")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"No se encuentra el dataset en: {DATA_PATH}. Ejecuta el pipeline primero.")
        self.df = pd.read_parquet(DATA_PATH)
        
        print("[INFO] Cargando modelo de ranking Capa 2...")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encuentra el modelo en: {MODEL_PATH}. Entrena la Capa 2 primero.")
        self.model = joblib.load(MODEL_PATH)
        
        print("[INFO] Mapeando metadatos de pilotos...")
        self._build_driver_mappings()
        
    def _build_driver_mappings(self):
        """Scans drivers.csv files under data/raw/ to build bidirectional mappings."""
        # 1. Fallback hardcoded mappings for 2026 grid just in case
        fallback = {
            1: "VER", 44: "HAM", 4: "NOR", 55: "SAI", 63: "RUS", 
            16: "LEC", 81: "PIA", 11: "PER", 23: "ALB", 14: "ALO", 
            31: "OCO", 18: "STR", 27: "HUL", 30: "LAW", 10: "GAS", 
            77: "BOT", 22: "TSU", 2: "SAR", 20: "MAG", 24: "ZHO",
            87: "BEA", 39: "BOR", 5: "ANT", 43: "COL", 9: "LIN",
            12: "HAD", 98: "BOR"
        }
        
        for k, v in fallback.items():
            self.driver_to_acronym[k] = v
            self.acronym_to_driver[v] = k
            
        # 2. Dynamic scan from raw files
        driver_files = glob.glob(str(RAW_DIR / "*" / "drivers.csv"))
        for f_path in driver_files:
            try:
                drv_df = pd.read_csv(f_path)
                drv_df.columns = [c.lower() for c in drv_df.columns]
                
                # Check column variations
                num_col = "driver_number" if "driver_number" in drv_df.columns else "number"
                code_col = "name_acronym" if "name_acronym" in drv_df.columns else "code"
                
                if num_col in drv_df.columns and code_col in drv_df.columns:
                    for _, row in drv_df.iterrows():
                        num = int(row[num_col])
                        code = str(row[code_col]).upper().strip()
                        if num and code and code != "NAN":
                            self.driver_to_acronym[num] = code
                            self.acronym_to_driver[code] = num
            except Exception:
                pass # Fail silently and use other files or fallback

    def get_available_sessions(self):
        """Returns list of races, drivers, and lap ranges available in the dataset."""
        if self.df is None:
            return {}, [], []
        
        races = self.df["race_name"].unique().tolist()
        
        # Build list of active driver acronyms
        driver_nums = self.df["driver_number"].unique().tolist()
        drivers = sorted([self.driver_to_acronym.get(int(n), f"Nº{int(n)}") for n in driver_nums])
        
        return races, drivers

    def get_lap_range(self, race, driver_acronym):
        """Returns the min and max lap available for a specific driver and race."""
        driver_num = self.acronym_to_driver.get(driver_acronym.upper())
        if driver_num is None:
            return 0, 0
            
        race_df = self.df[(self.df["race_name"] == race) & (self.df["driver_number"] == driver_num)]
        if race_df.empty:
            return 0, 0
            
        return int(race_df["lap_number"].min()), int(race_df["lap_number"].max())

    def get_predictions(self, race, driver_acronym, lap):
        """Retrieves the 6 candidates, runs Layer 2 prediction, and returns ranked DataFrame."""
        driver_num = self.acronym_to_driver.get(driver_acronym.upper())
        if driver_num is None:
            return pd.DataFrame()
            
        # 1. Filter candidates for current state
        state_df = self.df[
            (self.df["race_name"] == race) & 
            (self.df["driver_number"] == driver_num) & 
            (self.df["lap_number"] == lap)
        ].copy()
        
        if state_df.empty:
            return pd.DataFrame()
            
        # 2. Impute missing feature values (precaution)
        X = state_df[self.features].copy()
        for col in self.features:
            median_val = self.df[col].median()
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            
        # 3. Model inference
        state_df["predicted_success_score"] = self.model.predict(X)
        
        # 4. Sort by predicted score
        state_df = state_df.sort_values(by="predicted_success_score", ascending=False)
        return state_df
