import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import traceback

# Ignorar warnings menores de pandas para mantener limpia la consola
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURACIÓN DE RUTAS 
# =====================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
EVENTS_DIR = PROJECT_DIR / "data" / "events"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# FASE 1: PREPROCESAMIENTO GLOBAL
# =====================================================================

def _parse_laptime(series: pd.Series) -> pd.Series:
    """Convierte tiempos de vuelta '1:23.456' a segundos continuos."""
    def _convert(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        val = str(val).strip()
        if ":" in val:
            parts = val.split(":")
            try: return float(parts[0]) * 60 + float(parts[1])
            except ValueError: return np.nan
        try: return float(val)
        except ValueError: return np.nan
    return series.apply(_convert)

def preprocess_race(race_folder: Path) -> pd.DataFrame:
    print(f"\n  [1/2] Limpiando e integrando datos de: {race_folder.name}")
    
    # 1. Cargar Laps
    print("    -> Cargando archivo principal de vueltas (laps.csv)...")
    master = pd.read_csv(race_folder / "laps.csv", low_memory=False)
    
    # Estandarizar nombres (pasando todo a minúsculas preventivamente)
    master.columns = [c.lower() for c in master.columns]
    if "laptime" in master.columns: master = master.rename(columns={"laptime": "lap_duration"})
    if "lapnumber" in master.columns: master = master.rename(columns={"lapnumber": "lap_number"})
    if "drivernumber" in master.columns: master = master.rename(columns={"drivernumber": "driver_number"})
    
    master["lap_duration"] = _parse_laptime(master["lap_duration"])
    master = master.dropna(subset=["lap_duration"]).copy()
    master["lap_number"] = pd.to_numeric(master["lap_number"], errors="coerce").fillna(0).astype(int)
    master["driver_number"] = pd.to_numeric(master["driver_number"], errors="coerce").fillna(0).astype(int)
    
    print(f"    -> Vueltas válidas procesadas: {len(master)}")

    # =================================================================
    # SOLUCIÓN DEL ERROR: Reconstrucción matemática de la posición
    # =================================================================
    if "position" not in master.columns:
        print("    ⚠️ Columna 'position' no encontrada. Reconstruyendo posiciones matemáticas a partir de los tiempos acumulados...")
        master = master.sort_values(["driver_number", "lap_number"])
        
        # Calcular el tiempo total transcurrido para cada piloto
        master["cumulative_time"] = master.groupby("driver_number")["lap_duration"].cumsum()
        
        # El que tiene el menor tiempo acumulado en la vuelta X, es la posición 1
        master["position"] = master.groupby("lap_number")["cumulative_time"].rank(method="first").astype(int)
        
        # Limpiar columnas auxiliares
        master = master.drop(columns=["cumulative_time"])
        print("    -> Posiciones reconstruidas exitosamente.")
    else:
        # Si la columna ya existía, simplemente nos aseguramos de que sea int
        master["position"] = pd.to_numeric(master["position"], errors="coerce").fillna(0).astype(int)
        print("    -> Columna 'position' encontrada y validada.")

    # 2. Cargar Stints
    stints_file = race_folder / "stints.csv"
    if stints_file.exists() and stints_file.stat().st_size > 10:
        print("    -> Integrando datos de neumáticos (stints.csv)...")
        stints = pd.read_csv(stints_file)
        stints.columns = [c.lower() for c in stints.columns]
        
        stint_rows = []
        for _, row in stints.iterrows():
            start = row.get("lap_start", np.nan)
            end = row.get("lap_end", np.nan)
            if pd.isna(start) or pd.isna(end): continue
            
            for lap in range(int(start), int(end) + 1):
                stint_rows.append({
                    "driver_number": int(row.get("driver_number", 0)),
                    "lap_number": lap,
                    "compound": row.get("compound", "UNKNOWN"),
                    "stint_number": row.get("stint", row.get("stint_number", 1)),
                    "tyre_age": lap - int(start) + row.get("tyrelife", 0)
                })
        
        if stint_rows:
            stints_exp = pd.DataFrame(stint_rows)
            master = master.merge(stints_exp, on=["driver_number", "lap_number"], how="left")
            master["compound"] = master["compound"].fillna("UNKNOWN")
            master["tyre_age"] = master["tyre_age"].fillna(1)
            master["stint_number"] = master["stint_number"].fillna(1)
            print(f"      * {len(stints_exp)} mapeos de neumáticos aplicados.")
    else:
        print("    -> ⚠️ Omitiendo stints.csv (Archivo vacío o no encontrado).")

    # 3. Cargar Pits
    pit_file = race_folder / "pit.csv"
    if pit_file.exists() and pit_file.stat().st_size > 10:
        print("    -> Integrando paradas en boxes (pit.csv)...")
        pit = pd.read_csv(pit_file)
        pit.columns = [c.lower() for c in pit.columns]
        if "lap_number" in pit.columns and "driver_number" in pit.columns:
            pit_agg = pit.groupby(["driver_number", "lap_number"]).agg(
                pit_duration=("pit_duration", "sum") if "pit_duration" in pit.columns else ("stopduration", "sum")
            ).reset_index()
            master = master.merge(pit_agg, on=["driver_number", "lap_number"], how="left")
            print(f"      * {len(pit_agg)} paradas en boxes detectadas.")
    else:
        print("    -> ⚠️ Omitiendo pit.csv (Archivo vacío o no encontrado).")

    if "pit_duration" not in master.columns:
        master["pit_duration"] = 0.0
    master["pit_duration"] = master["pit_duration"].fillna(0.0)
    master["is_pit_lap"] = (master["pit_duration"] > 0).astype(int)

    print("    ✅ Integración completada con éxito.")
    return master.sort_values(["lap_number", "position"]).reset_index(drop=True)

# =====================================================================
# FASE 2: EXTRACCIÓN DE EVENTOS
# =====================================================================

def extract_events(master: pd.DataFrame, race_name: str) -> pd.DataFrame:
    print(f"  [2/2] Extrayendo Grafos de Eventos (Overtakes & Undercuts)...")
    events = []
    prev_positions = {}
    pit_windows = {}
    max_lap = master["lap_number"].max()
    
    overtakes_count = 0
    undercuts_count = 0
    
    for lap in range(1, max_lap + 1):
        lap_data = master[master["lap_number"] == lap]
        if lap_data.empty: continue
            
        current_positions = dict(zip(lap_data["driver_number"], lap_data["position"]))
        pitting_drivers = lap_data[lap_data["is_pit_lap"] == 1]["driver_number"].tolist()
        
        # 1. EVALUAR OVERTAKES
        if prev_positions:
            for driver, pos in current_positions.items():
                if driver not in prev_positions: continue
                prev_pos = prev_positions[driver]
                
                if pos < prev_pos and driver not in pitting_drivers:
                    for other_driver, other_prev_pos in prev_positions.items():
                        if other_driver == driver: continue
                        other_curr_pos = current_positions.get(other_driver)
                        
                        if other_curr_pos is not None:
                            if prev_pos > other_prev_pos and pos <= other_curr_pos:
                                if other_driver not in pitting_drivers:
                                    events.append({
                                        "race_id": race_name,
                                        "lap_number": lap,
                                        "event_type": "On_Track_Overtake",
                                        "initiator_driver": driver,
                                        "target_driver": other_driver,
                                        "initiator_compound": lap_data[lap_data['driver_number'] == driver]['compound'].values[0] if 'compound' in lap_data else 'UNK',
                                        "initiator_pos_change": f"P{prev_pos} -> P{pos}"
                                    })
                                    overtakes_count += 1

        # 2. ABRIR VENTANAS UNDERCUT
        for driver in pitting_drivers:
            if driver in prev_positions:
                pit_windows[driver] = {
                    "start_lap": lap,
                    "pos_before": prev_positions[driver],
                    "active": True
                }
                
        # 3. CERRAR VENTANAS (+3 vueltas después del pit)
        drivers_to_close = []
        for p_driver, window in pit_windows.items():
            if window["active"] and lap == window["start_lap"] + 3:
                if p_driver in current_positions:
                    pos_after = current_positions[p_driver]
                    subtype = "Undercut_Success" if pos_after < window["pos_before"] else "Undercut_Fail"
                    events.append({
                        "race_id": race_name,
                        "lap_number": window["start_lap"],
                        "event_type": "Pit_Strategy",
                        "initiator_driver": p_driver,
                        "target_driver": 0, 
                        "initiator_compound": "Pitted",
                        "initiator_pos_change": subtype
                    })
                    undercuts_count += 1
                drivers_to_close.append(p_driver)
                
        for d in drivers_to_close: del pit_windows[d]
        prev_positions = current_positions.copy()

    print(f"    -> Detectados: {overtakes_count} Adelantamientos en pista y {undercuts_count} eventos estratégicos (Undercuts).")
    return pd.DataFrame(events)

# =====================================================================
# ORQUESTADOR
# =====================================================================

def main():
    print("\n" + "="*70)
    print("🏎️  Iniciando F1 Data Pipeline - Generación de Grafos")
    print("="*70)

    if not RAW_DIR.exists():
        print(f"❌ Error: El directorio crudo no existe: {RAW_DIR}")
        return

    race_folders = [f for f in RAW_DIR.iterdir() if f.is_dir() and list(f.glob("laps.csv"))]
    
    if not race_folders:
        print(f"❌ No se encontraron carpetas de carreras con 'laps.csv' en {RAW_DIR}")
        return

    for folder in race_folders:
        race_name = folder.name
        try:
            # 1. Master Table 
            master_df = preprocess_race(folder)
            master_path = PROCESSED_DIR / f"{race_name}_master.parquet"
            master_df.to_parquet(master_path, index=False)
            print(f"    ✓ [OK] Dataset Master exportado a '{master_path.name}'")

            # 2. Events Table 
            events_df = extract_events(master_df, race_name)
            if not events_df.empty:
                events_path = EVENTS_DIR / f"{race_name}_events.parquet"
                events_df.to_parquet(events_path, index=False)
                print(f"    ✓ [OK] Dataset de Eventos exportado a '{events_path.name}'\n")
            else:
                print("    ⚠️ No se detectaron interacciones.\n")
                
        except Exception as e:
            print(f"  ❌ Error procesando {race_name}: {e}")
            print("Detalles del error:")
            print(traceback.format_exc())

if __name__ == "__main__":
    main()