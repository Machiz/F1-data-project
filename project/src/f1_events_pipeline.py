import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================================
# FASE 1: PREPROCESAMIENTO (Limpieza y Consolidación)
# =====================================================================

def _parse_laptime(series: pd.Series) -> pd.Series:
    """Convierte tiempos en formato '1:23.456' a segundos (float)."""
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
    """Carga los CSVs globales de la carrera, los limpia y los une en un Master DataFrame."""
    print(f"  [1/2] Preprocesando datos de: {race_folder.name}")
    
    # 1. Cargar Vueltas (Base)
    laps_file = race_folder / "laps.csv"
    if not laps_file.exists():
        raise FileNotFoundError(f"No se encontró {laps_file}")
    
    master = pd.read_csv(laps_file, low_memory=False)
    
    # Estandarizar nombres (adaptado de la API de OpenF1)
    if "LapTime" in master.columns: master = master.rename(columns={"LapTime": "lap_duration"})
    if "LapNumber" in master.columns: master = master.rename(columns={"LapNumber": "lap_number"})
    if "Position" in master.columns: master = master.rename(columns={"Position": "position"})
    
    master["lap_duration"] = _parse_laptime(master["lap_duration"])
    master = master.dropna(subset=["lap_duration", "position"]).copy()
    master["lap_number"] = pd.to_numeric(master["lap_number"], errors="coerce").astype(int)
    master["position"] = pd.to_numeric(master["position"], errors="coerce").astype(int)
    
    # 2. Cargar y unir Pit Stops
    pit_file = race_folder / "pit.csv"
    if pit_file.exists() and pit_file.stat().st_size > 10:
        pit = pd.read_csv(pit_file)
        if "lap_number" in pit.columns and "driver_number" in pit.columns:
            pit_agg = pit.groupby(["driver_number", "lap_number"]).agg(
                pit_duration=("pit_duration", "sum"),
            ).reset_index()
            master = master.merge(pit_agg, on=["driver_number", "lap_number"], how="left")
    
    # Rellenar vueltas sin pit stops con 0.0
    if "pit_duration" not in master.columns:
        master["pit_duration"] = 0.0
    master["pit_duration"] = master["pit_duration"].fillna(0.0)
    master["is_pit_lap"] = (master["pit_duration"] > 0).astype(int)

    # Ordenar cronológicamente
    master = master.sort_values(["lap_number", "position"]).reset_index(drop=True)
    return master


# =====================================================================
# FASE 2: EXTRACCIÓN DE EVENTOS (Undercuts & Overtakes)
# =====================================================================

def extract_events(master: pd.DataFrame, race_name: str) -> pd.DataFrame:
    """
    Analiza la tabla maestra vuelta a vuelta para encontrar interacciones 
    entre todos los pilotos de la parrilla.
    """
    print(f"  [2/2] Extrayendo eventos estratégicos y adelantamientos...")
    events = []
    
    # Variables de estado para rastrear posiciones en la vuelta anterior
    prev_positions = {} # {driver_number: position}
    pit_windows = {}    # {driver_number: {start_lap, position_before_pit}}
    
    max_lap = master["lap_number"].max()
    
    for lap in range(1, max_lap + 1):
        lap_data = master[master["lap_number"] == lap]
        if lap_data.empty: continue
            
        current_positions = dict(zip(lap_data["driver_number"], lap_data["position"]))
        pitting_drivers = lap_data[lap_data["is_pit_lap"] == 1]["driver_number"].tolist()
        
        # 1. EVALUAR ADELANTAMIENTOS EN PISTA (OVERTAKES)
        if prev_positions:
            for driver, pos in current_positions.items():
                if driver not in prev_positions: continue
                
                prev_pos = prev_positions[driver]
                
                # Si el piloto mejoró su posición (número menor) y no entró a pits
                if pos < prev_pos and driver not in pitting_drivers:
                    # Buscar a quién superó
                    for other_driver, other_prev_pos in prev_positions.items():
                        if other_driver == driver: continue
                        other_curr_pos = current_positions.get(other_driver)
                        
                        if other_curr_pos is None: continue
                        
                        # Si 'driver' estaba detrás de 'other_driver' y ahora está delante
                        if prev_pos > other_prev_pos and pos <= other_curr_pos:
                            # Ignoramos si el objetivo estaba en pits (eso es estrategia, no pista pura)
                            if other_driver not in pitting_drivers:
                                events.append({
                                    "race_id": race_name,
                                    "lap_number": lap,
                                    "event_type": "On_Track_Overtake",
                                    "initiator_driver": driver,
                                    "target_driver": other_driver,
                                    "initiator_pos_change": f"P{prev_pos} -> P{pos}",
                                    "target_pos_change": f"P{other_prev_pos} -> P{other_curr_pos}"
                                })

        # 2. ABRIR VENTANAS DE ESTRATEGIA (PITS / UNDERCUT)
        for driver in pitting_drivers:
            if driver in prev_positions:
                pit_windows[driver] = {
                    "start_lap": lap,
                    "pos_before": prev_positions[driver],
                    "active": True
                }
                
        # 3. CERRAR VENTANAS DE ESTRATEGIA (Evaluación post-pit)
        # Evaluamos 3 vueltas después de que el piloto entró a pits para ver cómo se asentó
        drivers_to_close = []
        for p_driver, window in pit_windows.items():
            if window["active"] and lap == window["start_lap"] + 3:
                if p_driver in current_positions:
                    pos_after = current_positions[p_driver]
                    
                    # Determinar si ganó posiciones a largo plazo por el undercut
                    event_subtype = "Undercut_Success" if pos_after < window["pos_before"] else "Routine_Stop_or_Overcut"
                    
                    events.append({
                        "race_id": race_name,
                        "lap_number": window["start_lap"],
                        "event_type": "Pit_Strategy_Phase",
                        "initiator_driver": p_driver,
                        "target_driver": "Grid", # Afecta a su entorno general
                        "initiator_pos_change": f"P{window['pos_before']} -> P{pos_after} (+3 laps)",
                        "target_pos_change": event_subtype
                    })
                drivers_to_close.append(p_driver)
                
        for d in drivers_to_close:
            del pit_windows[d]

        prev_positions = current_positions.copy()

    return pd.DataFrame(events)

# =====================================================================
# ORQUESTADOR PRINCIPAL
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="F1 Event Extraction Pipeline")
    parser.add_argument("--input", type=Path, default=Path("./project/data/raw"), help="Directorio con carpetas de carreras")
    parser.add_argument("--output_master", type=Path, default=Path("./project/data/processed"), help="Salida para Parquets maestros")
    parser.add_argument("--output_events", type=Path, default=Path("./project/data/events"), help="Salida para Grafo de Eventos")
    args = parser.parse_args()

    args.output_master.mkdir(parents=True, exist_ok=True)
    args.output_events.mkdir(parents=True, exist_ok=True)

    race_folders = [f for f in args.input.iterdir() if f.is_dir() and list(f.glob("laps.csv"))]
    
    if not race_folders:
        print("❌ No se encontraron carpetas con 'laps.csv' en el directorio de entrada.")
        return

    print(f"🏎️ Iniciando Pipeline de Eventos. Carreras detectadas: {len(race_folders)}\n" + "="*50)

    for folder in race_folders:
        race_name = folder.name
        try:
            # 1. Crear Master Table
            master_df = preprocess_race(folder)
            
            # Guardar el Master (Útil para clustering/ranking)
            master_path = args.output_master / f"{race_name}_master.parquet"
            master_df.to_parquet(master_path, index=False)
            print(f"    ✓ Master Table guardada ({len(master_df)} filas)")

            # 2. Extraer Eventos
            events_df = extract_events(master_df, race_name)
            
            # Guardar Eventos (Útil para Grafos)
            if not events_df.empty:
                events_path = args.output_events / f"{race_name}_events.parquet"
                events_df.to_parquet(events_path, index=False)
                print(f"    ✓ Dataset de Eventos generado ({len(events_df)} interacciones detectadas)\n")
            else:
                print("    ⚠️ No se detectaron interacciones en esta carrera.\n")
                
        except Exception as e:
            print(f"  ❌ Error procesando {race_name}: {e}\n")

if __name__ == "__main__":
    main()