import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
import traceback

warnings.filterwarnings('ignore')

# Configuración de rutas
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
FEATURES_DIR = PROJECT_DIR / "data" / "processed" / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def linreg_slope(y):
    """Calcula la pendiente de regresión lineal para una serie de valores."""
    if len(y) < 2 or np.any(pd.isna(y)):
        return 0.0
    x = np.arange(len(y))
    n = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x**2)
    sum_xy = np.sum(x*y)
    denom = (n * sum_xx - sum_x**2)
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom

def process_race_recommender(race_folder: Path, df_tel_race: pd.DataFrame) -> pd.DataFrame:
    race_folder_name = race_folder.name
    # Mapeo del nombre del folder al race_name en parquet (ej: australia_2026 -> australia)
    race_name = race_folder_name.replace("_2026", "")
    print(f"    -> Procesando carrera: {race_folder_name} (Clave en telemetria: {race_name})")
    
    # 1. Cargar archivos raw
    laps = pd.read_csv(race_folder / "laps.csv")
    
    # Limpiar nulos de llaves de tiempo y convertir fechas
    laps = laps.dropna(subset=["date_start"])
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601")
    laps = laps.sort_values("date_start")
    
    # Cargar intervalos si existen
    intervals_file = race_folder / "intervals.csv"
    if intervals_file.exists():
        intervals = pd.read_csv(intervals_file)
        intervals = intervals.dropna(subset=["date"])
        intervals["interval"] = pd.to_numeric(intervals["interval"], errors="coerce")
        intervals["gap_to_leader"] = pd.to_numeric(intervals["gap_to_leader"], errors="coerce")
        intervals["date"] = pd.to_datetime(intervals["date"], format="ISO8601")
        intervals = intervals.sort_values("date")
        
        # 2. Alinear los intervalos con las vueltas para gap_ahead (gap al auto de adelante)
        intervals_with_lap = pd.merge_asof(
            intervals,
            laps[["driver_number", "lap_number", "date_start"]],
            left_on="date",
            right_on="date_start",
            by="driver_number",
            direction="backward"
        )
        
        lap_intervals = intervals_with_lap.groupby(["driver_number", "lap_number"]).agg(
            gap_ahead=("interval", "last"),
            gap_to_leader=("gap_to_leader", "last")
        ).reset_index()
    else:
        print(f"    [WARN] intervals.csv no encontrado para {race_folder_name}. Se usaran valores de trafico por defecto.")
        lap_intervals = pd.DataFrame(columns=["driver_number", "lap_number", "gap_ahead", "gap_to_leader"])
    
    # 3. Obtener el número máximo de vueltas de la carrera
    total_laps = df_tel_race["lap_number"].max()
    
    # 4. Construir base piloto-vuelta desde Capa A (telemetry_features_v4)
    base_df = df_tel_race[[
        "race_name", "driver_number", "lap_number", "lap_duration", 
        "tyre_age", "compound_ord", "lap_vs_best_stint", "position", 
        "stint_number", "is_pit_lap"
    ]].copy()
    
    # Integrar gaps con la base
    base_df = pd.merge(base_df, lap_intervals, on=["driver_number", "lap_number"], how="left")
    
    # Calcular gap_behind a partir del gap_ahead del auto de atrás (posición + 1)
    base_df = base_df.sort_values(["lap_number", "position"])
    base_df["gap_behind"] = base_df.groupby("lap_number")["gap_ahead"].shift(-1)
    
    # Imputar nulos de tráfico con un valor por defecto seguro (ej: 30.0 segundos = pista limpia)
    base_df["gap_ahead"] = base_df["gap_ahead"].fillna(30.0)
    base_df["gap_behind"] = base_df["gap_behind"].fillna(30.0)
    
    # 5. Calcular métricas de ritmo y degradación recientes (ventana móvil de 3 vueltas)
    base_df = base_df.sort_values(["driver_number", "lap_number"])
    
    base_df["lap_mean_3"] = base_df.groupby("driver_number")["lap_duration"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    base_df["lap_std_3"] = base_df.groupby("driver_number")["lap_duration"].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0.0)
    
    # Pendiente de ritmo de las últimas 3 vueltas
    base_df["lap_slope_3"] = base_df.groupby("driver_number")["lap_duration"].transform(
        lambda x: x.rolling(3, min_periods=2).apply(linreg_slope, raw=True)
    ).fillna(0.0)
    
    # Pendiente de degradación de las últimas 3 vueltas (deg_rate_3lap)
    base_df["deg_rate_3lap"] = base_df.groupby("driver_number")["lap_vs_best_stint"].transform(
        lambda x: x.rolling(3, min_periods=2).apply(linreg_slope, raw=True)
    ).fillna(0.0)
    
    # 6. Calcular contexto de carrera
    base_df["laps_remaining"] = total_laps - base_df["lap_number"]
    base_df["race_pct_complete"] = base_df["lap_number"] / total_laps
    base_df["is_top10"] = (base_df["position"] <= 10).astype(int)
    
    # 7. Identificar paradas en boxes reales y calcular etiqueta proxy de éxito
    pit_stops = base_df[base_df["is_pit_lap"] == 1][["driver_number", "lap_number", "position", "lap_duration"]].copy()
    
    pit_metrics = []
    for _, row in pit_stops.iterrows():
        drv = row["driver_number"]
        lp = row["lap_number"]
        pos_pit = row["position"]
        
        # Ritmo de referencia antes del pit: media de las últimas 3 vueltas antes del pit
        pace_before_df = base_df[(base_df["driver_number"] == drv) & (base_df["lap_number"] >= lp-3) & (base_df["lap_number"] < lp)]
        pace_before = pace_before_df["lap_duration"].mean() if not pace_before_df.empty else np.nan
        
        # Desempeño post-pit: ventana de las siguientes 5 vueltas (lp+1 a lp+5)
        after_df = base_df[(base_df["driver_number"] == drv) & (base_df["lap_number"] > lp) & (base_df["lap_number"] <= lp+5)]
        
        if not after_df.empty:
            pos_after = after_df["position"].iloc[-1]
            pos_gain = pos_pit - pos_after # Ganancia de posición (+ es mejor)
            
            pace_after = after_df["lap_duration"].mean()
            pace_improvement = pace_before - pace_after if not pd.isna(pace_before) else 0.0
            
            # Score de éxito continuo:
            # Ponderación de ganancia de posiciones y mejora de ritmo (en segundos)
            success_score = pos_gain + (pace_improvement / 2.0)
        else:
            pos_gain = 0.0
            pace_improvement = 0.0
            success_score = 0.0
            
        pit_metrics.append({
            "driver_number": drv,
            "pit_lap": lp,
            "pos_gain": pos_gain,
            "pace_improvement": pace_improvement,
            "success_score": success_score
        })
        
    df_pit_metrics = pd.DataFrame(pit_metrics)
    
    # 8. Expandir en 6 candidatos (wait_laps = 0 a 5)
    expanded_rows = []
    for _, row in base_df.iterrows():
        drv = row["driver_number"]
        lp = row["lap_number"]
        
        for w in range(6):
            candidate_row = row.copy()
            candidate_row["candidate"] = w
            candidate_row["wait_laps"] = w
            
            # El predicted_cost_of_staying lo inicializamos en 0.0. 
            # Será actualizado después con las predicciones del modelo de regresión (Capa 1).
            candidate_row["predicted_cost_of_staying"] = 0.0
            
            # Asignación de la etiqueta de éxito
            label = 0.0
            if not df_pit_metrics.empty:
                # Buscamos si el piloto paró en la vuelta real correspondiente (lp + w)
                match = df_pit_metrics[(df_pit_metrics["driver_number"] == drv) & (df_pit_metrics["pit_lap"] == lp + w)]
                if not match.empty:
                    label = match["success_score"].values[0]
                else:
                    # Penalización por no parar en la ventana de parada real
                    # Esto ayuda al modelo a aprender que las vueltas lejanas a la parada real son subóptimas
                    label = -2.0 if w > 0 else 0.0
            
            candidate_row["success_score_label"] = label
            expanded_rows.append(candidate_row)
            
    print(f"      * Filas expandidas generadas: {len(expanded_rows)}")
    return pd.DataFrame(expanded_rows)

def main():
    print("\n" + "="*70)
    print("Iniciando F1 Recommender Pipeline - Capa C (Recomendacion)")
    print("="*70)
    
    # Validar archivos de entrada
    telemetry_path = FEATURES_DIR / "telemetry_features_v4.parquet"
    if not telemetry_path.exists():
        print(f"Error: No se encuentra {telemetry_path.name} en {FEATURES_DIR}")
        return
        
    print(f"    [OK] Cargando Capa A: {telemetry_path.name}")
    df_tel = pd.read_parquet(telemetry_path)
    
    # Identificar carpetas de carrera con laps.csv
    race_folders = [f for f in RAW_DIR.iterdir() if f.is_dir() and (f / "laps.csv").exists()]
    if not race_folders:
        print(f"No se encontraron carpetas de carreras con laps.csv en {RAW_DIR}")
        return
        
    all_expanded_data = []
    
    for folder in race_folders:
        race_name = folder.name.replace("_2026", "")
        df_tel_race = df_tel[df_tel["race_name"] == race_name]
        
        if df_tel_race.empty:
            print(f"    Omitiendo {folder.name} porque no hay datos coincidentes en telemetry_features_v4.")
            continue
            
        try:
            df_expanded = process_race_recommender(folder, df_tel_race)
            all_expanded_data.append(df_expanded)
        except Exception as e:
            print(f"    Error procesando {folder.name}: {e}")
            print(traceback.format_exc())
            
    if all_expanded_data:
        df_final = pd.concat(all_expanded_data, ignore_index=True)
        
        # Guardar en parquet
        recommendation_dir = PROJECT_DIR / "data" / "processed" / "recommendation"
        recommendation_dir.mkdir(parents=True, exist_ok=True)
        output_path = recommendation_dir / "pit_decision_candidates_v1.parquet"
        df_final.to_parquet(output_path, index=False)
        print("\n" + "="*70)
        print("Pipeline completado exitosamente.")
        print(f"   * Dataset final guardado en: {output_path}")
        print(f"   * Dimensiones del dataset: {df_final.shape}")
        print(f"   * Columnas generadas: {list(df_final.columns)}")
        print("="*70)
    else:
        print("No se generaron datos para exportar.")

if __name__ == "__main__":
    main()
