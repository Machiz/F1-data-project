import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

class F1PitEnv(gym.Env):
    """
    Gymnasium Environment for F1 Pit Stop Decision Strategy.
    Simulates a race for a single driver in a competitive field, predicting lap times
    using a compact Random Forest Regressor (trained on the fly) and calculating positions
    and gaps dynamically relative to the historical times of all other drivers in the same race.
    Handles gaps in telemetry and models track slowdowns (Safety Cars) dynamically.
    Optimized for high-speed RL training.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data_path=None, model_path=None, features_path=None):
        super().__init__()
        
        # Paths setup
        src_dir = Path(__file__).resolve().parent
        project_dir = src_dir.parent.parent
        
        if data_path is None:
            data_path = project_dir / "data" / "recommendation" / "pit_decision_candidates_v1.parquet"
        if features_path is None:
            features_path = project_dir / "data" / "features" / "regression_features.joblib"
            
        # Load data
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data not found at: {data_path}")
        self.df = pd.read_parquet(data_path)
        
        # Filter for wait_laps == 0 to get unique actual driver-race-lap records
        self.df = self.df[self.df["wait_laps"] == 0].copy()
        
        # One-hot encode race_name without dropping the original column
        for r_name in ["australia", "china", "japan", "united_states"]:
            self.df[f"race_name_{r_name}"] = (self.df["race_name"] == r_name).astype(float)
        
        # Clean and impute NaNs in columns
        group_cols = ["race_name", "driver_number"]
        self.df["compound_ord"] = self.df.groupby(group_cols)["compound_ord"].ffill().bfill().fillna(2.0)
        self.df["tyre_age"] = self.df.groupby(group_cols)["tyre_age"].ffill().bfill().fillna(0.0)
        self.df["stint_number"] = self.df.groupby(group_cols)["stint_number"].ffill().bfill().fillna(1.0)
        self.df["position"] = self.df.groupby(group_cols)["position"].ffill().bfill().fillna(10.0)
        self.df["lap_duration"] = self.df.groupby(group_cols)["lap_duration"].ffill().bfill().fillna(95.0)
        self.df["gap_ahead"] = self.df["gap_ahead"].fillna(30.0)
        self.df["gap_behind"] = self.df["gap_behind"].fillna(30.0)
        self.df["is_pit_lap"] = self.df["is_pit_lap"].fillna(0.0)
        
        if not Path(features_path).exists():
            raise FileNotFoundError(f"Feature names list not found at: {features_path}")
        self.features_list = joblib.load(features_path)
        
        # Identify one-hot race columns
        self.race_cols = [c for c in self.features_list if c.startswith("race_name_")]
        
        # Preprocess and sort data
        self.df = self.df.sort_values(["race_name", "driver_number", "lap_number"]).copy()
        
        # Train a compact RandomForestRegressor for closed-loop pace simulation.
        # The forest smooths the single-tree pace estimates so PPO has less room
        # to exploit unrealistically strong fresh-tyre jumps.
        # Using exact same feature names and order as Capa 1 stacking model
        self.fast_features = [
            "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
            "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
            "is_top10", "laps_remaining", "race_pct_complete", 
            "gap_ahead", "gap_behind", "wait_laps", "driver_number"
        ] + self.race_cols
        
        # Impute for training
        X_train = self.df[self.fast_features].fillna(0.0)
        y_train = self.df["lap_duration"].fillna(95.0)
        
        self.fast_model = RandomForestRegressor(
            n_estimators=40,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        self.fast_model.fit(X_train, y_train)
        
        # Compute cumulative lap durations for all drivers to simulate positions
        self.df["cum_duration"] = self.df.groupby(["race_name", "driver_number"])["lap_duration"].cumsum()
        
        # Create lookup dictionaries for fast access
        self.cum_time_lookup = self.df.set_index(["race_name", "driver_number", "lap_number"])["cum_duration"].to_dict()
        self.lap_duration_lookup = self.df.set_index(["race_name", "driver_number", "lap_number"])["lap_duration"].to_dict()
        
        # Filter driver-race pairs that have at least 20 laps for stability
        driver_lap_counts = self.df.groupby(["race_name", "driver_number"]).size().reset_index(name="counts")
        valid_pairs = driver_lap_counts[driver_lap_counts["counts"] >= 20]
        
        # Also filter out driver-race pairs with only 1 compound recorded (incomplete telemetry)
        compounds_per_driver = self.df.groupby(["race_name", "driver_number"])["compound_ord"].nunique().reset_index(name="n_compounds")
        multi_compound = compounds_per_driver[compounds_per_driver["n_compounds"] >= 2]
        valid_pairs = valid_pairs.merge(multi_compound[["race_name", "driver_number"]], on=["race_name", "driver_number"])
        
        self.race_driver_pairs = valid_pairs[["race_name", "driver_number"]].values.tolist()
        
        # Define action space:
        # 0 = Stay on track (no pit stop)
        # 1 = Pit for SOFT compound
        # 2 = Pit for MEDIUM compound
        # 3 = Pit for HARD compound
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (14 continuous variables, bounded)
        self.observation_space = spaces.Box(
            low=np.array([0.0]*14, dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Pit stop lane time loss per race
        self.pit_loss_dict = {
            "australia": 15.5,
            "china": 39.0,
            "japan": 32.8,
            "united_states": 12.0
        }
        
        # Median baseline lap durations per race under green-flag conditions
        self.race_medians = {
            "australia": 85.151,
            "china": 98.444,
            "japan": 96.011,
            "united_states": 94.629
        }
        
        # Initialize state variables
        self.current_race = None
        self.current_driver = None
        self.total_laps = 57
        self.valid_laps = []
        
        self.lap_number = 1
        self.tyre_age = 0
        self.compound_ord = 2
        self.stint_number = 1
        self.position = 10
        self.gap_ahead = 30.0
        self.gap_behind = 30.0
        
        self.lap_history = []
        self.lap_vs_best_stint_history = []
        self.best_lap_in_stint = 90.0
        self.agent_cum_time = 0.0
        self.used_compounds = set()
        self.compound_bonus_given = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Select race and driver
        if options and "race_name" in options and "driver_number" in options:
            self.current_race = options["race_name"]
            self.current_driver = options["driver_number"]
        else:
            pair_idx = self.np_random.integers(len(self.race_driver_pairs))
            self.current_race, self.current_driver = self.race_driver_pairs[pair_idx]
            
        # Get all laps for the selected driver
        driver_df = self.df[(self.df["race_name"] == self.current_race) & (self.df["driver_number"] == self.current_driver)].sort_values("lap_number")
        self.valid_laps = sorted(driver_df["lap_number"].unique())
        self.total_laps = int(driver_df["lap_number"].max())
        
        # Set starting lap number
        self.lap_number = int(driver_df["lap_number"].min())
        
        # Load initial telemetry values from the first lap
        first_lap_row = driver_df.iloc[0]
        self.tyre_age = int(first_lap_row["tyre_age"])
        self.compound_ord = int(first_lap_row["compound_ord"])
        self.stint_number = int(first_lap_row["stint_number"])
        self.position = int(first_lap_row["position"])
        self.gap_ahead = float(first_lap_row["gap_ahead"]) if "gap_ahead" in first_lap_row and not pd.isna(first_lap_row["gap_ahead"]) else 30.0
        self.gap_behind = float(first_lap_row["gap_behind"]) if "gap_behind" in first_lap_row and not pd.isna(first_lap_row["gap_behind"]) else 30.0
        
        # Cumulative agent time
        if self.lap_number > 1:
            self.agent_cum_time = self.cum_time_lookup.get((self.current_race, self.current_driver, self.lap_number - 1), 0.0)
        else:
            self.agent_cum_time = 0.0
            
        self.lap_history = []
        self.lap_vs_best_stint_history = []
        self.best_lap_in_stint = float(first_lap_row["lap_duration"])
        
        # Populate history windows with first lap duration to avoid division by zero
        first_dur = float(first_lap_row["lap_duration"])
        for _ in range(3):
            self.lap_history.append(first_dur)
            self.lap_vs_best_stint_history.append(0.0)
            
        self.used_compounds = {self.compound_ord}
        self.compound_bonus_given = False
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
        
    def _get_obs(self):
        norm_lap = float(self.lap_number) / float(self.total_laps) if self.total_laps > 0 else 0.0
        norm_age = float(self.tyre_age) / 50.0
        norm_compound = float(self.compound_ord) / 3.0
        norm_position = float(self.position) / 20.0
        norm_stint = float(self.stint_number) / 5.0
        norm_gap_ahead = min(self.gap_ahead, 30.0) / 30.0
        norm_gap_behind = min(self.gap_behind, 30.0) / 30.0
        
        lap_vs_best = self.lap_vs_best_stint_history[-1] if self.lap_vs_best_stint_history else 0.0
        
        lap_mean_3 = np.mean(self.lap_history[-3:]) if len(self.lap_history) >= 3 else 90.0
        norm_lap_mean_3 = lap_mean_3 / 100.0
        
        lap_std_3 = np.std(self.lap_history[-3:]) if len(self.lap_history) >= 3 else 0.0
        norm_lap_std_3 = min(lap_std_3 / 10.0, 1.0)
        
        laps_rem = float(self.total_laps - self.lap_number) / float(self.total_laps) if self.total_laps > 0 else 0.0
        
        used_s = 1.0 if 1 in self.used_compounds else 0.0
        used_m = 1.0 if 2 in self.used_compounds else 0.0
        used_h = 1.0 if 3 in self.used_compounds else 0.0
        
        obs_vec = np.array([
            norm_lap, norm_age, norm_compound, norm_position, norm_stint,
            norm_gap_ahead, norm_gap_behind, lap_vs_best, norm_lap_mean_3,
            norm_lap_std_3, laps_rem, used_s, used_m, used_h
        ], dtype=np.float32)
        
        return obs_vec
        
    def _get_info(self):
        return {
            "race_name": self.current_race,
            "driver_number": self.current_driver,
            "lap_number": self.lap_number,
            "tyre_age": self.tyre_age,
            "compound_ord": self.compound_ord,
            "position": self.position,
            "gap_ahead": self.gap_ahead,
            "gap_behind": self.gap_behind,
            "agent_cum_time": self.agent_cum_time,
            "stint_number": self.stint_number,
            "used_compounds": list(self.used_compounds)
        }
        
    def step(self, action):
        is_pit = action in [1, 2, 3]
        previous_compound = self.compound_ord
        previous_tyre_age = self.tyre_age
        
        # Advance lap number checking the actual valid laps sequence (handling telemetry gaps)
        try:
            curr_idx = self.valid_laps.index(self.lap_number)
            if curr_idx + 1 < len(self.valid_laps):
                next_lap = self.valid_laps[curr_idx + 1]
                laps_elapsed = next_lap - self.lap_number
            else:
                next_lap = self.lap_number + 1
                laps_elapsed = 1
        except ValueError:
            next_lap = self.lap_number + 1
            laps_elapsed = 1
            
        pit_time_loss = 0.0
        if is_pit:
            self.compound_ord = int(action)
            # Reset tyre age, but add elapsed laps if we jumped over a gap (except the pit lap itself)
            self.tyre_age = max(0, laps_elapsed - 1)
            self.stint_number += 1
            self.used_compounds.add(self.compound_ord)
            pit_time_loss = self.pit_loss_dict.get(self.current_race, 20.0)
        else:
            self.tyre_age += laps_elapsed
            
        # Build raw features list for high-speed DecisionTree prediction (avoiding DataFrame overhead)
        features_list = []
        features_list.append(float(self.tyre_age))
        features_list.append(float(self.compound_ord))
        
        if is_pit or len(self.lap_history) == 0:
            lap_vs_best = 0.0
        else:
            last_lap = self.lap_history[-1]
            lap_vs_best = (last_lap - self.best_lap_in_stint) / self.best_lap_in_stint if self.best_lap_in_stint > 0 else 0.0
            
        features_list.append(float(lap_vs_best))
        
        lap_history_3 = self.lap_history[-3:] if len(self.lap_history) >= 3 else [90.0, 90.0, 90.0]
        lap_vs_best_history_3 = self.lap_vs_best_stint_history[-3:] if len(self.lap_vs_best_stint_history) >= 3 else [0.0, 0.0, 0.0]
        
        features_list.append(float(np.mean(lap_history_3)))
        features_list.append(float(np.std(lap_history_3)))
        
        # Calculate slope
        def get_slope(y):
            if len(y) < 2:
                return 0.0
            x = np.arange(len(y))
            n = len(y)
            denom = n * np.sum(x**2) - np.sum(x)**2
            if denom == 0:
                return 0.0
            return (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / denom
            
        features_list.append(float(get_slope(lap_history_3)))
        features_list.append(float(get_slope(lap_vs_best_history_3)))
        
        features_list.append(float(self.position))
        features_list.append(1.0 if self.position <= 10 else 0.0)
        
        laps_rem = self.total_laps - self.lap_number
        features_list.append(float(laps_rem))
        features_list.append(float(self.lap_number) / float(self.total_laps) if self.total_laps > 0 else 0.0)
        
        features_list.append(float(self.gap_ahead))
        features_list.append(float(self.gap_behind))
        features_list.append(0.0) # wait_laps is always 0.0 in execution
        features_list.append(float(self.current_driver))
        
        # Add race one-hot columns
        for col in self.race_cols:
            race_suffix = col.replace("race_name_", "")
            features_list.append(1.0 if self.current_race == race_suffix else 0.0)
            
        # Predict pace (base lap duration without pit stop time loss)
        input_array = np.array([features_list], dtype=np.float32)
        pred_lap_duration = float(self.fast_model.predict(input_array)[0])
        
        # Clean any anomalies in prediction
        if pred_lap_duration < 40.0:
            pred_lap_duration = 90.0  # Fallback to average lap time
            
        # Base lap duration (includes pit stop if pitting)
        base_duration = pred_lap_duration + pit_time_loss
        
        # Calculate track condition (Safety Car) slowdown factor
        # Look up real average lap times of other drivers on the CURRENT lap
        lap_df = self.df[(self.df["race_name"] == self.current_race) & (self.df["lap_number"] == self.lap_number)]
        other_drivers = lap_df[lap_df["driver_number"] != self.current_driver]
        
        slowdown_factor = 1.0
        if not other_drivers.empty:
            other_mean_lap = other_drivers["lap_duration"].mean()
            median_pace = self.race_medians.get(self.current_race, 90.0)
            if other_mean_lap > 1.10 * median_pace:
                slowdown_factor = other_mean_lap / median_pace
                
        # Total duration for this lap (scaled by safety car or track conditions if slow)
        lap_duration = base_duration * slowdown_factor
        
        # If we jumped a gap, we multiply the lap time by the number of elapsed laps
        if laps_elapsed > 1:
            lap_duration += (laps_elapsed - 1) * pred_lap_duration * slowdown_factor
            
        # Update stint best lap time
        if not is_pit:
            if lap_duration < self.best_lap_in_stint:
                self.best_lap_in_stint = lap_duration
        else:
            self.best_lap_in_stint = pred_lap_duration
            
        self.lap_history.append(lap_duration)
        self.lap_vs_best_stint_history.append(lap_vs_best)
        
        # Update cumulative time
        self.agent_cum_time += lap_duration
        
        # Rank agent time relative to field (using current lap_number before advancing)
        self._update_position_and_gaps()
        
        # Update lap number to the next lap
        self.lap_number = next_lap
        
        # Termination conditions
        terminated = self.lap_number >= self.total_laps
        truncated = False
        
        # Calculate reward
        reward = self._calculate_reward(
            lap_duration,
            is_pit,
            terminated,
            previous_compound=previous_compound,
            previous_tyre_age=previous_tyre_age,
        )
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
        
    def _update_position_and_gaps(self):
        lap_df = self.df[(self.df["race_name"] == self.current_race) & (self.df["lap_number"] == self.lap_number)]
        
        if lap_df.empty:
            return
            
        other_drivers = lap_df[lap_df["driver_number"] != self.current_driver]
        if other_drivers.empty:
            return
            
        driver_times = []
        for _, row in other_drivers.iterrows():
            drv = row["driver_number"]
            cum_time = self.cum_time_lookup.get((self.current_race, drv, self.lap_number), None)
            if cum_time is not None:
                driver_times.append((drv, cum_time))
                
        driver_times.append((self.current_driver, self.agent_cum_time))
        driver_times.sort(key=lambda x: x[1])
        
        agent_rank = [i for i, x in enumerate(driver_times) if x[0] == self.current_driver]
        if not agent_rank:
            return
        agent_rank = agent_rank[0]
        
        self.position = agent_rank + 1
        
        if agent_rank > 0:
            driver_ahead_time = driver_times[agent_rank - 1][1]
            self.gap_ahead = max(0.0, self.agent_cum_time - driver_ahead_time)
        else:
            self.gap_ahead = 30.0
            
        if agent_rank < len(driver_times) - 1:
            driver_behind_time = driver_times[agent_rank + 1][1]
            self.gap_behind = max(0.0, driver_behind_time - self.agent_cum_time)
        else:
            self.gap_behind = 30.0
            
    def _calculate_reward(self, lap_duration, is_pit, terminated, previous_compound=None, previous_tyre_age=None):
        # Base: Relative pace advantage (median pace of race - agent lap duration)
        median_pace = self.race_medians.get(self.current_race, 90.0)
        reward = median_pace - lap_duration
        
        laps_remaining = max(0, self.total_laps - self.lap_number)
        race_progress = float(self.lap_number) / float(self.total_laps) if self.total_laps > 0 else 0.0
        completed_pits = max(0, self.stint_number - 1)
        
        # 1. Structural pit stop penalty to avoid pitting every lap. The penalty
        # grows after the second stop because the target behavior is a realistic
        # one-stop or two-stop strategy, not always-fresh tyres.
        if is_pit:
            extra_stop_penalty = max(0, completed_pits - 2) * 260.0
            reward -= 40.0 + extra_stop_penalty
            
            if previous_compound == self.compound_ord:
                reward -= 60.0
                
            min_useful_stint = {1: 16, 2: 18, 3: 22}.get(int(previous_compound or self.compound_ord), 16)
            if previous_tyre_age is not None and previous_tyre_age < min_useful_stint:
                reward -= (min_useful_stint - previous_tyre_age) * 7.0
                
            # Avoid front-loading the whole strategy without blocking legitimate
            # undercut windows. Keep this softer than the extra-stop penalty so
            # pace and track position can still decide the exact pit lap.
            if completed_pits == 1 and race_progress < 0.30:
                reward -= (0.30 - race_progress) * 240.0
            elif completed_pits == 2 and race_progress < 0.58:
                reward -= (0.58 - race_progress) * 280.0
                
            if laps_remaining <= 2:
                reward -= 80.0
            
        # 2. Small one-time bonus for satisfying the two-compound rule. This is
        # intentionally modest; the main incentive is paid at race end so the
        # agent learns the timing of the stop instead of rushing the rule.
        if len(self.used_compounds) >= 2 and not self.compound_bonus_given:
            reward += 35.0
            self.compound_bonus_given = True
        
        # If the agent has not changed compound and the race is running out, add
        # pressure gradually instead of a single cliff at the final lap.
        if len(self.used_compounds) < 2 and laps_remaining <= 12:
            reward -= (13 - laps_remaining) * 6.0
        
        # 3. Tyre age thresholds before critical drop-off:
        # SOFT = 22 laps, MEDIUM = 32 laps, HARD = 42 laps
        wear_limit = {1: 22, 2: 32, 3: 42}.get(self.compound_ord, 30)
        if self.tyre_age > wear_limit:
            excess = self.tyre_age - wear_limit
            reward -= 3.0 * excess  # Reduced penalty to encourage stretching stints
            
        if terminated:
            # Check F1 regulations: must use at least 2 distinct compounds during the race
            if len(self.used_compounds) < 2:
                reward -= 1500.0  # Huge disqualification penalty
            else:
                reward += 180.0   # Bonus for complying with regulations
                
            # Final strategy regularization: allow one-stop and two-stop races,
            # but penalize extra stops because the task is to find the optimal
            # pit lap/window rather than oscillating between fresh compounds.
            final_pits = max(0, self.stint_number - 1)
            if final_pits > 2:
                reward -= (final_pits - 2) * 500.0
            elif final_pits == 1:
                reward += 60.0
            elif final_pits == 2:
                reward += 45.0
                
            # Bonus for final position (lower rank is better, rank 1 is best)
            # Position 1 gives +200, position 20 gives +10
            pos_bonus = (21 - self.position) * 10.0
            reward += pos_bonus
            
        # Scale reward by 10.0 for stable PPO training (values are smaller now)
        return reward / 10.0
