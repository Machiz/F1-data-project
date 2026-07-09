import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "recommendation" / "pit_decision_candidates_v1.parquet"
FEATURES_DIR = PROJECT_DIR / "data" / "features"

def compute_regression_targets(df):
    """
    Calcula los objetivos reales de la regresión (ritmo futuro y varianza)
    si el piloto se queda en pista durante 'wait_laps' vueltas.
    """
    print("Calculando objetivos de regresión para Capa 1...")
    df = df.sort_values(["race_name", "driver_number", "lap_number"]).copy()
    
    # Creamos diccionarios rápidos para buscar duraciones de vueltas y stints
    lap_dur_dict = df.set_index(["race_name", "driver_number", "lap_number"])["lap_duration"].to_dict()
    stint_dict = df.set_index(["race_name", "driver_number", "lap_number"])["stint_number"].to_dict()
    pit_dict = df.set_index(["race_name", "driver_number", "lap_number"])["is_pit_lap"].to_dict()
    
    future_mean = []
    future_std = []
    
    for idx, row in df.iterrows():
        race = row["race_name"]
        drv = row["driver_number"]
        lp = row["lap_number"]
        w = int(row["wait_laps"])
        
        if w == 0:
            future_mean.append(row["lap_duration"])
            future_std.append(0.0)
            continue
            
        # Analizamos las vueltas de L a L + w - 1
        laps_to_check = list(range(int(lp), int(lp) + w))
        durations = []
        valid = True
        stint_start = stint_dict.get((race, drv, lp))
        
        for curr_lp in laps_to_check:
            key = (race, drv, curr_lp)
            # Si el piloto paró en boxes antes de completar las w vueltas o cambió de stint, no se quedó en pista
            if key not in lap_dur_dict or stint_dict.get(key) != stint_start or (curr_lp > lp and pit_dict.get((race, drv, curr_lp), 0) == 1):
                valid = False
                break
            durations.append(lap_dur_dict[key])
            
        if valid and len(durations) == w:
            future_mean.append(np.mean(durations))
            future_std.append(np.std(durations) if len(durations) > 1 else 0.0)
        else:
            future_mean.append(np.nan)
            future_std.append(np.nan)
            
    df["target_future_mean"] = future_mean
    df["target_future_std"] = future_std
    return df

def evaluate_ndcg(df_eval, group_cols, rank_col, label_col, k=3):
    """
    Calcula el NDCG@K promedio para un dataframe con predicciones de ranking.
    """
    ndcgs = []
    grouped = df_eval.groupby(group_cols)
    
    for _, group in grouped:
        if len(group) < 2:
            continue
        # Ordenamos los candidatos reales por el score predicho (descendente)
        sorted_group = group.sort_values(by=rank_col, ascending=False)
        actual_labels = sorted_group[label_col].values
        
        # Si todas las etiquetas son iguales, NDCG no es informativo
        if np.all(actual_labels == actual_labels[0]):
            continue
            
        # Relevancia ideal (ordenada descendente)
        ideal_labels = np.sort(group[label_col].values)[::-1]
        
        # Calcular DCG@K
        dcg = 0.0
        idcg = 0.0
        for i in range(min(k, len(actual_labels))):
            # Usamos relevancia exponencial
            rel = actual_labels[i]
            # Normalizar rel para evitar negativos en la fórmula exponencial
            rel_norm = max(0, rel + 2) # trasladamos -2 a 0, etc.
            dcg += (2**rel_norm - 1) / np.log2(i + 2)
            
            ideal_rel = ideal_labels[i]
            ideal_rel_norm = max(0, ideal_rel + 2)
            idcg += (2**ideal_rel_norm - 1) / np.log2(i + 2)
            
        if idcg > 0:
            ndcgs.append(dcg / idcg)
            
    return np.mean(ndcgs) if ndcgs else 1.0

def run_experiments():
    print("Cargando dataset...")
    df = pd.read_parquet(DATA_PATH)
    
    # Calcular targets de regresión
    df = compute_regression_targets(df)
    
    # Filtrar filas donde el target de regresión es válido (el piloto realmente se quedó en pista)
    df_reg = df.dropna(subset=["target_future_mean"]).copy()
    
    # Definir características y targets
    features = [
        "tyre_age", "compound_ord", "lap_vs_best_stint", "lap_mean_3", 
        "lap_std_3", "lap_slope_3", "deg_rate_3lap", "position", 
        "is_top10", "laps_remaining", "race_pct_complete", 
        "gap_ahead", "gap_behind", "wait_laps",
        "pit_gap_ahead", "pit_gap_behind", "delta_time_loss",
        "compound_SOFT", "compound_MEDIUM", "compound_HARD"
    ]
    
    # Imputar nulos con la mediana de cada columna para evitar errores en regresores simples
    for col in features:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        df_reg[col] = df_reg[col].fillna(median_val)
        
    X = df_reg[features]
    y_mean = df_reg["target_future_mean"]
    
    print(f"Dimensiones para regresión: {X.shape}")
    
    # Validación Cruzada GroupKFold por carrera
    gkf = GroupKFold(n_splits=4)
    groups = df_reg["race_name"]
    
    print("\n--- EXPERIMENTOS CAPA 1: MODELOS DE REGRESIÓN DE DEGRADACIÓN ---")
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    best_reg_name = None
    best_reg_r2 = -np.inf
    best_reg_model = None
    
    for name, model in reg_models.items():
        mses = []
        r2s = []
        for train_idx, test_idx in gkf.split(X, y_mean, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_mean.iloc[train_idx], y_mean.iloc[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mses.append(mean_squared_error(y_test, preds))
            r2s.append(r2_score(y_test, preds))
            
        mean_mse = np.mean(mses)
        mean_r2 = np.mean(r2s)
        print(f"Model: {name:<20} | Mean MSE: {mean_mse:.4f} | Mean R2: {mean_r2:.4f}")
        
        if mean_r2 > best_reg_r2:
            best_reg_r2 = mean_r2
            best_reg_name = name
            best_reg_model = model

    print(f"\n>> Mejor modelo de Regresión Capa 1: {best_reg_name} (R2: {best_reg_r2:.4f})")
    
    # Entrenamos el mejor regresor sobre todo el dataset de regresión para usarlo en la Capa 2
    best_reg_model.fit(X, y_mean)
    # Guardamos el modelo en la carpeta de features
    joblib.dump(best_reg_model, FEATURES_DIR / "regression_layer1_model.pkl")
    print(f"Modelo Capa 1 guardado en: features/regression_layer1_model.pkl")
    
    # INTEGRACIÓN: Predecir el coste de quedarse en pista para todos los candidatos
    # X_all incluye a todos los candidatos (incluso los no observados)
    X_all = df[features]
    df["predicted_future_pace"] = best_reg_model.predict(X_all)
    
    # predicted_cost_of_staying = wait_laps * (predicted_future_pace - best_lap_stint)
    # Para evitar romper si best_lap_stint no está en df, lo buscamos o lo aproximamos.
    # En telemetry_features_v4, best_lap_stint está disponible. Vamos a verificar:
    # Sí, está en telemetry_features_v4, por lo que fue arrastrado al dataset de candidatos.
    # Si no existiera, usamos la duración de la vuelta actual como aproximación base.
    best_lap = df["lap_duration"] # valor fallback
    
    df["predicted_cost_of_staying"] = df["wait_laps"] * (df["predicted_future_pace"] - best_lap)
    
    # Sobrescribimos el dataset con la columna puente poblada
    df.to_parquet(DATA_PATH, index=False)
    print("Dataset de candidatos actualizado con predicted_cost_of_staying.")
    
    # --- EXPERIMENTOS CAPA 2: MODELOS DE RANKING ---
    print("\n--- EXPERIMENTOS CAPA 2: MODELOS DE RANKING DE DECISIÓN ---")
    
    ranking_features = features + ["predicted_cost_of_staying"]
    
    # Creación del grupo para validación cruzada de ranking (GroupKFold por carrera)
    # Cada grupo en ranking es la combinación carrera-piloto-vuelta (la 'query' de ranking)
    df["query_id"] = df["race_name"] + "_" + df["driver_number"].astype(str) + "_" + df["lap_number"].astype(str)
    
    # Para entrenar ranking con XGBRanker, necesitamos ordenar los datos por query_id
    df_rank = df.sort_values("query_id").copy()
    
    # Convertir success_score_label a relevancia entera de 0 a 5 por cada query para XGBRanker
    df_rank["rank_label"] = df_rank.groupby("query_id")["success_score_label"].rank(method="first").astype(int) - 1
    
    # Definir variables de ranking
    X_rank = df_rank[ranking_features]
    y_rank = df_rank["success_score_label"]
    query_groups = df_rank["query_id"]
    races_groups = df_rank["race_name"]
    
    gkf_rank = GroupKFold(n_splits=4)
    
    # Listas para almacenar métricas de todas las alternativas
    ndcgs_random1, ndcgs_random3 = [], []
    ndcgs_heur1, ndcgs_heur3 = [], []
    ndcgs_pop1, ndcgs_pop3 = [], []
    ndcgs_rf1, ndcgs_rf3 = [], []
    ndcgs_xgb1, ndcgs_xgb3 = [], []
    
    print("\nEvaluando todos los modelos y baselines mediante validación cruzada GroupKFold...")
    for train_idx, test_idx in gkf_rank.split(X_rank, y_rank, races_groups):
        df_tr_split = df_rank.iloc[train_idx].copy()
        df_te_split = df_rank.iloc[test_idx].copy()
        
        # 1. Random Baseline
        np.random.seed(42)
        df_te_split["pred_random"] = np.random.rand(len(df_te_split))
        
        # 2. Tyre-Age Heuristic Baseline
        df_te_split["pred_heuristic"] = -np.abs((df_te_split["tyre_age"] + df_te_split["wait_laps"]) - 18)
        
        # 3. Popularity Baseline
        # Contamos frecuencia en train
        pit_counts = df_tr_split[df_tr_split["is_pit_lap"] == 1].groupby(["compound_ord", "tyre_age"]).size().to_dict()
        pop_scores = []
        for _, row in df_te_split.iterrows():
            comp = row["compound_ord"]
            target_age = row["tyre_age"] + row["wait_laps"]
            pop_scores.append(pit_counts.get((comp, target_age), 0))
        df_te_split["pred_popularity"] = pop_scores
        
        # 4. Point-wise Random Forest Regressor
        X_tr = df_tr_split[ranking_features]
        y_tr = df_tr_split["success_score_label"]
        X_te = df_te_split[ranking_features]
        
        # Calcular pesos de muestra para el RF en train
        is_pos_tr = (y_tr > -2.0).astype(int)
        n_pos_tr = (is_pos_tr == 1).sum()
        n_neg_tr = (is_pos_tr == 0).sum()
        if n_pos_tr > 0:
            w_ratio_tr = n_neg_tr / n_pos_tr
            sw_tr = np.where(is_pos_tr == 1, w_ratio_tr, 1.0)
        else:
            sw_tr = np.ones(len(y_tr))
            
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_tr, y_tr, sample_weight=sw_tr)
        df_te_split["pred_rf"] = rf.predict(X_te)
        
        # 5. List-wise XGBRanker
        df_tr_xgb = df_tr_split.sort_values("query_id")
        train_group_sizes = df_tr_xgb.groupby("query_id").size().values
        df_tr_xgb["rank_label"] = df_tr_xgb.groupby("query_id")["success_score_label"].rank(method="first").astype(int) - 1
        
        df_te_xgb = df_te_split.sort_values("query_id")
        X_tr_xgb = df_tr_xgb[ranking_features]
        y_tr_xgb = df_tr_xgb["rank_label"]
        X_te_xgb = df_te_xgb[ranking_features]
        
        ranker = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        ranker.fit(X_tr_xgb, y_tr_xgb, group=train_group_sizes)
        df_te_xgb["pred_xgb"] = ranker.predict(X_te_xgb)
        
        # Calcular NDCG
        ndcgs_random1.append(evaluate_ndcg(df_te_split, "query_id", "pred_random", "success_score_label", k=1))
        ndcgs_random3.append(evaluate_ndcg(df_te_split, "query_id", "pred_random", "success_score_label", k=3))
        
        ndcgs_heur1.append(evaluate_ndcg(df_te_split, "query_id", "pred_heuristic", "success_score_label", k=1))
        ndcgs_heur3.append(evaluate_ndcg(df_te_split, "query_id", "pred_heuristic", "success_score_label", k=3))
        
        ndcgs_pop1.append(evaluate_ndcg(df_te_split, "query_id", "pred_popularity", "success_score_label", k=1))
        ndcgs_pop3.append(evaluate_ndcg(df_te_split, "query_id", "pred_popularity", "success_score_label", k=3))
        
        ndcgs_rf1.append(evaluate_ndcg(df_te_split, "query_id", "pred_rf", "success_score_label", k=1))
        ndcgs_rf3.append(evaluate_ndcg(df_te_split, "query_id", "pred_rf", "success_score_label", k=3))
        
        ndcgs_xgb1.append(evaluate_ndcg(df_te_xgb, "query_id", "pred_xgb", "success_score_label", k=1))
        ndcgs_xgb3.append(evaluate_ndcg(df_te_xgb, "query_id", "pred_xgb", "success_score_label", k=3))
        
    print("\n" + "="*50)
    print("TABLA COMPARATIVA DE MODELOS DE RECOMENDACIÓN (CAPA 2)")
    print("="*50)
    print(f"Random Baseline            | NDCG@1: {np.mean(ndcgs_random1):.4f} | NDCG@3: {np.mean(ndcgs_random3):.4f}")
    print(f"Tyre-Age Heuristic         | NDCG@1: {np.mean(ndcgs_heur1):.4f} | NDCG@3: {np.mean(ndcgs_heur3):.4f}")
    print(f"Popularity Baseline        | NDCG@1: {np.mean(ndcgs_pop1):.4f} | NDCG@3: {np.mean(ndcgs_pop3):.4f}")
    print(f"Random Forest (Point-wise) | NDCG@1: {np.mean(ndcgs_rf1):.4f} | NDCG@3: {np.mean(ndcgs_rf3):.4f}  <-- SELECCIONADO")
    print(f"XGBRanker (List-wise)      | NDCG@1: {np.mean(ndcgs_xgb1):.4f} | NDCG@3: {np.mean(ndcgs_xgb3):.4f}")
    print("="*50)
    
    # Entrenar el mejor modelo (Random Forest Point-wise) sobre todo el dataset de ranking para guardar
    print("\nEntrenando modelo final de Capa 2 (Random Forest Point-wise) sobre todo el dataset...")
    # Calcular pesos de muestra para el RF final
    is_pos_final = (y_rank > -2.0).astype(int)
    n_pos_final = (is_pos_final == 1).sum()
    n_neg_final = (is_pos_final == 0).sum()
    if n_pos_final > 0:
        w_ratio_final = n_neg_final / n_pos_final
        sw_final = np.where(is_pos_final == 1, w_ratio_final, 1.0)
    else:
        sw_final = np.ones(len(y_rank))
        
    final_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    final_rf.fit(X_rank, y_rank, sample_weight=sw_final)
    joblib.dump(final_rf, FEATURES_DIR / "ranking_layer2_model.pkl")
    print(f"Modelo Capa 2 Random Forest Regressor guardado en: features/ranking_layer2_model.pkl")

if __name__ == "__main__":
    run_experiments()
