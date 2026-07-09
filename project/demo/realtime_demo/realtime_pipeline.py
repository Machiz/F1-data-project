import os
import glob
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR = PROJECT_DIR / "data" / "raw"

def linreg_slope(y):
    """Calcula la pendiente de regresión lineal para una serie de valores."""
    if len(y) < 2 or np.any(pd.isna(y)):
        return 0.0
    x = np.arange(len(y))
    n = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    denom = (n * sum_xx - sum_x * sum_x)
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom

class RealtimePipeline:
    def __init__(self, race_name="united_kingdom", driver_acronym="VER"):
        self.race_name = race_name
        self.driver_acronym = driver_acronym.upper()
        self.df_master = None
        self.model_reg = None
        self.model_rank = None
        self.features_reg = []
        self.driver_num = None
        self.driver_to_acronym = {}
        self.acronym_to_driver = {}
        
        # Features needed for Layer 2 RF model
        self.features_rank = [
            "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
            "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
            "is_top10", "laps_remaining", "race_pct_complete", 
            "gap_ahead", "gap_behind", "wait_laps", "predicted_cost_of_staying",
            "pit_gap_ahead", "pit_gap_behind", "delta_time_loss",
            "compound_SOFT", "compound_MEDIUM", "compound_HARD"
        ]
        
    def load_resources(self):
        """Loads models, feature configurations, drivers metadata, and master dataset."""
        # 1. Load models
        reg_model_path = MODEL_DIR / "regression_layer1_model.pkl"
        rank_model_path = MODEL_DIR / "ranking_layer2_model.pkl"
        reg_features_path = MODEL_DIR / "regression_features.joblib"
        
        if not reg_model_path.exists() or not rank_model_path.exists() or not reg_features_path.exists():
            raise FileNotFoundError("Modelos no encontrados. Asegúrate de haber entrenado los modelos primero.")
            
        self.model_reg = joblib.load(reg_model_path)
        self.model_rank = joblib.load(rank_model_path)
        self.features_reg = joblib.load(reg_features_path)
        
        # 2. Build driver mappings
        self._build_driver_mappings()
        self.driver_num = self.acronym_to_driver.get(self.driver_acronym)
        if self.driver_num is None:
            raise ValueError(f"Piloto {self.driver_acronym} no reconocido.")
            
        # 3. Load master dataset for the race
        master_files = glob.glob(str(DATA_PROCESSED_DIR / "master" / f"{self.race_name}_*master*.parquet"))
        if not master_files:
            raise FileNotFoundError(f"No se encontró el archivo master Parquet para {self.race_name} en data/processed/master/")
        
        # Read first matched master file
        self.df_master = pd.read_parquet(master_files[0])
        self._enrich_master_data()
        
    def _build_driver_mappings(self):
        """Scans drivers.csv files under data/raw/ to build bidirectional mappings."""
        fallback = {
            1: "VER", 44: "HAM", 4: "NOR", 55: "SAI", 63: "RUS", 
            16: "LEC", 81: "PIA", 11: "PER", 23: "ALB", 14: "ALO", 
            31: "OCO", 18: "STR", 27: "HUL", 30: "LAW", 10: "GAS", 
            77: "BOT", 22: "TSU", 2: "SAR", 20: "MAG", 24: "ZHO"
        }
        for k, v in fallback.items():
            self.driver_to_acronym[k] = v
            self.acronym_to_driver[v] = k
            
        driver_files = glob.glob(str(RAW_DIR / "*" / "drivers.csv"))
        for f_path in driver_files:
            try:
                drv_df = pd.read_csv(f_path)
                drv_df.columns = [c.lower() for c in drv_df.columns]
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
                pass

    def _enrich_master_data(self):
        """Enriches the master dataset with compound ordinals, lap stints deltas, and live traffic gaps."""
        # 1. Map compound strings to ordinals
        compound_map = {"SOFT": 1.0, "MEDIUM": 2.0, "HARD": 3.0}
        self.df_master["compound_ord"] = self.df_master["compound"].str.upper().map(compound_map).fillna(2.0)
        
        # Add one-hot compounds
        self.df_master["compound_SOFT"] = (self.df_master["compound_ord"] == 1.0).astype(float)
        self.df_master["compound_MEDIUM"] = (self.df_master["compound_ord"] == 2.0).astype(float)
        self.df_master["compound_HARD"] = (self.df_master["compound_ord"] == 3.0).astype(float)
        
        # 2. Compute lap_vs_best_stint dynamically
        self.df_master["best_lap_in_stint"] = self.df_master.groupby(["driver_number", "stint_number"])["lap_duration"].transform(
            lambda x: x.cummin()
        )
        self.df_master["lap_vs_best_stint"] = self.df_master["lap_duration"] - self.df_master["best_lap_in_stint"]
        
        # Compute delta_time_loss
        self.df_master = self.df_master.sort_values(["driver_number", "lap_number"])
        self.df_master["delta_time_loss"] = self.df_master.groupby(["driver_number", "stint_number"])["lap_vs_best_stint"].transform(
            lambda x: x.expanding().mean()
        )
        
        # 3. Load and merge intervals data for gap_ahead and gap_behind
        raw_folder = RAW_DIR / f"{self.race_name}_2026"
        intervals_file = raw_folder / "intervals.csv"
        laps_file = raw_folder / "laps.csv"
        
        if intervals_file.exists() and laps_file.exists():
            try:
                laps_raw = pd.read_csv(laps_file)
                laps_raw.columns = [c.lower() for c in laps_raw.columns]
                if "laptime" in laps_raw.columns: laps_raw = laps_raw.rename(columns={"laptime": "lap_duration"})
                if "lapnumber" in laps_raw.columns: laps_raw = laps_raw.rename(columns={"lapnumber": "lap_number"})
                if "drivernumber" in laps_raw.columns: laps_raw = laps_raw.rename(columns={"drivernumber": "driver_number"})
                
                laps_raw["date_start"] = pd.to_datetime(laps_raw["date_start"], format="ISO8601", errors="coerce")
                laps_raw = laps_raw.dropna(subset=["date_start"]).sort_values("date_start")
                
                intervals = pd.read_csv(intervals_file)
                intervals = intervals.dropna(subset=["date"])
                intervals["interval"] = pd.to_numeric(intervals["interval"], errors="coerce")
                intervals["gap_to_leader"] = pd.to_numeric(intervals["gap_to_leader"], errors="coerce")
                intervals["date"] = pd.to_datetime(intervals["date"], format="ISO8601", errors="coerce")
                intervals = intervals.sort_values("date")
                
                intervals_with_lap = pd.merge_asof(
                    intervals,
                    laps_raw[["driver_number", "lap_number", "date_start"]],
                    left_on="date",
                    right_on="date_start",
                    by="driver_number",
                    direction="backward"
                )
                
                lap_intervals = intervals_with_lap.groupby(["driver_number", "lap_number"]).agg(
                    gap_ahead=("interval", "last"),
                    gap_to_leader=("gap_to_leader", "last")
                ).reset_index()
                
                self.df_master = pd.merge(self.df_master, lap_intervals, on=["driver_number", "lap_number"], how="left")
            except Exception:
                pass
                
        # Fill NA traffic gaps and sort
        self.df_master["gap_ahead"] = self.df_master["gap_ahead"].fillna(30.0)
        self.df_master = self.df_master.sort_values(["lap_number", "position"])
        self.df_master["gap_behind"] = self.df_master.groupby("lap_number")["gap_ahead"].shift(-1).fillna(30.0)
        self.df_master = self.df_master.sort_values(["driver_number", "lap_number"])

        # 3.1 Calcular huecos de tráfico tras parada (pit_gap_ahead_lap, pit_gap_behind_lap)
        PIT_LOSS_DICT = {
            "australia": 15.5,
            "china": 39.0,
            "japan": 32.8,
            "united_states": 12.0,
            "united_kingdom": 20.0,
        }
        pit_loss = PIT_LOSS_DICT.get(self.race_name, 20.0)

        self.df_master["cum_duration"] = self.df_master.groupby("driver_number")["lap_duration"].cumsum()
        self.df_master["is_pit_lap"] = (self.df_master["pit_duration"] > 0).astype(float)
        
        cum_lookup = self.df_master.set_index(["driver_number", "lap_number"])["cum_duration"].to_dict()
        pit_lookup = self.df_master.set_index(["driver_number", "lap_number"])["is_pit_lap"].to_dict()
        drivers_list = sorted(self.df_master["driver_number"].unique())

        pit_gap_ahead_list = []
        pit_gap_behind_list = []

        for idx, row in self.df_master.iterrows():
            drv = row["driver_number"]
            lp = row["lap_number"]
            cum = row["cum_duration"]
            is_pit_real = pit_lookup.get((drv, lp), 0.0) == 1.0
            
            cum_post_pit = cum if is_pit_real else cum + pit_loss
            
            other_cums = []
            for other_drv in drivers_list:
                if other_drv != drv:
                    other_cum = cum_lookup.get((other_drv, lp))
                    if other_cum is not None:
                        other_cums.append(other_cum)
                        
            if not other_cums:
                pit_gap_ahead_list.append(30.0)
                pit_gap_behind_list.append(30.0)
                continue
                
            ahead_cums = [c for c in other_cums if c < cum_post_pit]
            gap_a = cum_post_pit - max(ahead_cums) if ahead_cums else 30.0
            
            behind_cums = [c for c in other_cums if c > cum_post_pit]
            gap_b = min(behind_cums) - cum_post_pit if behind_cums else 30.0
            
            pit_gap_ahead_list.append(gap_a)
            pit_gap_behind_list.append(gap_b)

        self.df_master["pit_gap_ahead_lap"] = pit_gap_ahead_list
        self.df_master["pit_gap_behind_lap"] = pit_gap_behind_list

    def get_total_laps(self):
        """Returns the maximum lap number in the race dataset."""
        if self.df_master is None:
            return 0
        return int(self.df_master["lap_number"].max())

    def get_leaderboard_at_lap(self, lap):
        """Returns the top 5 drivers at the end of a specific lap."""
        if self.df_master is None:
            return []
        lap_df = self.df_master[self.df_master["lap_number"] == lap].sort_values("position")
        leaders = []
        for _, row in lap_df.head(5).iterrows():
            d_num = int(row["driver_number"])
            acro = self.driver_to_acronym.get(d_num, f"Nº{d_num}")
            pos = int(row["position"])
            leaders.append(f"P{pos}: {acro}")
        return leaders

    def get_realtime_inference(self, lap):
        """Simulates live telemetry feature calculations and executes model predictions up to lap."""
        if self.df_master is None:
            return pd.DataFrame()
            
        # 1. Filter telemetry history up to current lap N (simulate live paddock environment)
        df_history = self.df_master[
            (self.df_master["driver_number"] == self.driver_num) & 
            (self.df_master["lap_number"] <= lap)
        ].sort_values("lap_number")
        
        if df_history.empty:
            return pd.DataFrame()
            
        current_state = df_history.iloc[-1].copy()
        
        # 2. Extract last 3 laps for rolling metrics calculation
        df_last3 = df_history.tail(3)
        lap_durations = df_last3["lap_duration"].values
        lap_deltas = df_last3["lap_vs_best_stint"].values
        
        lap_mean_3 = np.mean(lap_durations)
        lap_std_3 = np.std(lap_durations) if len(lap_durations) > 1 else 0.0
        lap_slope_3 = linreg_slope(lap_durations)
        deg_rate_3lap = linreg_slope(lap_deltas)
        
        # 3. Create candidates pool (wait_laps = 0 to 5)
        candidates = []
        total_laps = self.get_total_laps()
        
        # Lookups rápidos para el mapeo
        pit_gaps_lookup = self.df_master.set_index(["driver_number", "lap_number"])[["pit_gap_ahead_lap", "pit_gap_behind_lap"]].to_dict("index")
        delta_loss_lookup = self.df_master.set_index(["driver_number", "lap_number"])["delta_time_loss"].to_dict()

        for w in range(7):
            c_row = current_state.copy()
            c_row["wait_laps"] = w
            c_row["is_no_pit"] = int(w == 6)
            c_row["lap_mean_3"] = lap_mean_3
            c_row["lap_std_3"] = lap_std_3
            c_row["lap_slope_3"] = lap_slope_3
            c_row["deg_rate_3lap"] = deg_rate_3lap
            c_row["laps_remaining"] = total_laps - lap
            c_row["race_pct_complete"] = lap / total_laps
            c_row["is_top10"] = 1 if int(current_state["position"]) <= 10 else 0
            c_row["predicted_cost_of_staying"] = 0.0
            
            # Map target lap features
            target_lap = lap + w if w != 6 else lap
            gaps = pit_gaps_lookup.get((self.driver_num, target_lap))
            if gaps is not None:
                c_row["pit_gap_ahead"] = gaps["pit_gap_ahead_lap"]
                c_row["pit_gap_behind"] = gaps["pit_gap_behind_lap"]
            else:
                gaps_curr = pit_gaps_lookup.get((self.driver_num, lap), {"pit_gap_ahead_lap": 30.0, "pit_gap_behind_lap": 30.0})
                c_row["pit_gap_ahead"] = gaps_curr["pit_gap_ahead_lap"]
                c_row["pit_gap_behind"] = gaps_curr["pit_gap_behind_lap"]
                
            dtl = delta_loss_lookup.get((self.driver_num, target_lap))
            c_row["delta_time_loss"] = dtl if dtl is not None else current_state["delta_time_loss"]
            
            c_row["compound_SOFT"] = float(current_state["compound_ord"] == 1.0)
            c_row["compound_MEDIUM"] = float(current_state["compound_ord"] == 2.0)
            c_row["compound_HARD"] = float(current_state["compound_ord"] == 3.0)
            
            candidates.append(c_row)
            
        df_candidates = pd.DataFrame(candidates)
        
        # 4. Layer 1 Inference: Predict future pace and calculate bridge cost
        df_reg_input = df_candidates.copy()
        
        # One-hot encode race name
        df_reg_input["race_name"] = self.race_name
        df_reg_input = pd.get_dummies(df_reg_input, columns=["race_name"])
        
        # Add missing race dummies from feature list
        for col in self.features_reg:
            if col not in df_reg_input.columns:
                df_reg_input[col] = 0.0
                
        X_reg = df_reg_input[self.features_reg].copy()
        # Impute X_reg just in case
        for col in self.features_reg:
            X_reg[col] = X_reg[col].fillna(0.0)
            
        df_candidates["predicted_future_pace"] = self.model_reg.predict(X_reg)
        effective_wait = df_candidates["wait_laps"].clip(upper=5)
        df_candidates["predicted_cost_of_staying"] = effective_wait * (
            df_candidates["predicted_future_pace"] - current_state["lap_duration"]
        )
        
        # 5. Layer 2 Inference: Rank the options
        X_rank = df_candidates[self.features_rank].copy()
        for col in self.features_rank:
            X_rank[col] = X_rank[col].fillna(0.0)
            
        df_candidates["predicted_success_score"] = self.model_rank.predict(X_rank)
        
        # 6. Sort by success score
        df_candidates = df_candidates.sort_values(by="predicted_success_score", ascending=False)
        return df_candidates
