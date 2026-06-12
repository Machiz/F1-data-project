import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "recommendation" / "pit_decision_candidates_v1.parquet"

def evaluate_ndcg(df_eval, group_cols, rank_col, label_col, k=3):
    """
    Calcula el NDCG@K promedio para un dataframe con predicciones de ranking.
    """
    ndcgs = []
    grouped = df_eval.groupby(group_cols)
    
    for _, group in grouped:
        if len(group) < 2:
            continue
        actual_labels = group[label_col].values
        if np.all(actual_labels == actual_labels[0]):
            continue
            
        # Ordenamos los candidatos reales por el score predicho (descendente)
        sorted_group = group.sort_values(by=rank_col, ascending=False)
        sorted_labels = sorted_group[label_col].values
        
        # Relevancia ideal (ordenada descendente)
        ideal_labels = np.sort(actual_labels)[::-1]
        
        # Calcular DCG@K
        dcg = 0.0
        idcg = 0.0
        for i in range(min(k, len(sorted_labels))):
            rel = sorted_labels[i]
            # Normalizar rel para evitar negativos en la fórmula exponencial (+100)
            rel_norm = max(0, rel + 100)
            dcg += (2**rel_norm - 1) / np.log2(i + 2)
            
            ideal_rel = ideal_labels[i]
            ideal_rel_norm = max(0, ideal_rel + 100)
            idcg += (2**ideal_rel_norm - 1) / np.log2(i + 2)
            
        if idcg > 0:
            ndcgs.append(dcg / idcg)
            
    return np.mean(ndcgs) if ndcgs else 1.0

def get_popularity_scores(df_train, df_test):
    """
    Calcula las puntuaciones de popularidad empírica basadas en la frecuencia histórica
    de paradas en boxes para cada compuesto y edad del neumático.
    """
    # Contar paradas reales por compuesto y edad del neumático en el conjunto de entrenamiento
    pit_counts = df_train[df_train["is_pit_lap"] == 1].groupby(["compound_ord", "tyre_age"]).size().to_dict()
    
    # Asignar la puntuación a cada candidato basándose en la edad proyectada del neumático (tyre_age + wait_laps)
    scores = []
    for _, row in df_test.iterrows():
        comp = row["compound_ord"]
        target_age = row["tyre_age"] + row["wait_laps"]
        # Buscar frecuencia en el diccionario
        score = pit_counts.get((comp, target_age), 0)
        scores.append(score)
    return scores

def main():
    print("--- EVALUACIÓN DE SISTEMAS BASELINE PARA RECOMENDACIÓN DE PIT STOPS ---")
    if not DATA_PATH.exists():
        print(f"Error: No se encontró el dataset en {DATA_PATH}")
        return
        
    df = pd.read_parquet(DATA_PATH)
    df["query_id"] = df["race_name"] + "_" + df["driver_number"].astype(str) + "_" + df["lap_number"].astype(str)
    
    # Imputación de nulos
    for col in ["tyre_age", "compound_ord", "wait_laps"]:
        df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)
        
    gkf = GroupKFold(n_splits=4)
    races_groups = df["race_name"]
    
    ndcgs_random_1, ndcgs_random_3 = [], []
    ndcgs_heuristic_1, ndcgs_heuristic_3 = [], []
    ndcgs_popularity_1, ndcgs_popularity_3 = [], []
    
    for train_idx, test_idx in gkf.split(df, df["success_score_label"], races_groups):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        
        # 1. Random Baseline
        np.random.seed(42)
        df_test["score_random"] = np.random.rand(len(df_test))
        
        # 2. Tyre-Age Heuristic Baseline (Pitar cuando la edad proyectada del neumático está cerca de la media histórica de 18 vueltas)
        df_test["score_heuristic"] = -np.abs((df_test["tyre_age"] + df_test["wait_laps"]) - 18)
        
        # 3. Popularity Baseline
        df_test["score_popularity"] = get_popularity_scores(df_train, df_test)
        
        # Evaluación
        ndcgs_random_1.append(evaluate_ndcg(df_test, "query_id", "score_random", "success_score_label", k=1))
        ndcgs_random_3.append(evaluate_ndcg(df_test, "query_id", "score_random", "success_score_label", k=3))
        
        ndcgs_heuristic_1.append(evaluate_ndcg(df_test, "query_id", "score_heuristic", "success_score_label", k=1))
        ndcgs_heuristic_3.append(evaluate_ndcg(df_test, "query_id", "score_heuristic", "success_score_label", k=3))
        
        ndcgs_popularity_1.append(evaluate_ndcg(df_test, "query_id", "score_popularity", "success_score_label", k=1))
        ndcgs_popularity_3.append(evaluate_ndcg(df_test, "query_id", "score_popularity", "success_score_label", k=3))
        
    print("\nResultados Consolidados de Baselines:")
    print(f"1. Random Baseline            | NDCG@1: {np.mean(ndcgs_random_1):.4f} | NDCG@3: {np.mean(ndcgs_random_3):.4f}")
    print(f"2. Tyre-Age Heuristic (18L)   | NDCG@1: {np.mean(ndcgs_heuristic_1):.4f} | NDCG@3: {np.mean(ndcgs_heuristic_3):.4f}")
    print(f"3. Popularity (Empirical)     | NDCG@1: {np.mean(ndcgs_popularity_1):.4f} | NDCG@3: {np.mean(ndcgs_popularity_3):.4f}")

if __name__ == "__main__":
    main()
