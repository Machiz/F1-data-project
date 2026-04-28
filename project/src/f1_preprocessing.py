"""
F1 Preprocessing Pipeline
=========================
Lee 51 CSVs de 3 carreras, los limpia, integra y exporta
un parquet por carrera listo para EDA.

Estructura esperada de archivos:
  <race_folder>/
    laps_driver_16.csv, laps_driver_44.csv
    car_data_driver_16.csv, car_data_driver_44.csv
    stints_driver_16.csv, stints_driver_44.csv
    pit_driver_16.csv, pit_driver_44.csv
    intervals_driver_16.csv, intervals_driver_44.csv
    location_driver_16.csv, location_driver_44.csv
    weather.csv, race_control.csv, sessions.csv
    meetings.csv, drivers.csv

Uso:
    python f1_preprocessing.py --input ./data --output ./parquet
"""

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. INGESTA  — leer y clasificar los CSVs
# ─────────────────────────────────────────────

DRIVER_TYPES = ["laps", "car_data", "stints", "pit", "intervals", "location"]
GLOBAL_TYPES = ["weather", "race_control", "sessions", "meetings", "drivers"]
DRIVER_RE    = re.compile(r"^(?P<tipo>[a-z_]+)_driver_(?P<num>\d+)$")


def load_race_folder(folder: Path) -> dict:
    """
    Carga todos los CSVs de una carpeta de carrera.
    Retorna:
        {
          "driver": {tipo: {num: df}},   # archivos por piloto
          "global": {tipo: df}           # archivos globales
        }
    """
    print(f"Cargando archivos de la carpeta: {folder}")
    data = {"driver": {t: {} for t in DRIVER_TYPES},
            "global": {}}

    for csv_path in sorted(folder.glob("*.csv")):
        print(f"  Leyendo: {csv_path.name}")
        stem = csv_path.stem
        m = DRIVER_RE.match(stem)

        if m:
            tipo = m.group("tipo")
            num  = int(m.group("num"))
            if tipo in DRIVER_TYPES:
                df = pd.read_csv(csv_path, low_memory=False)
                df["driver_number"] = num
                data["driver"][tipo][num] = df

        elif stem in GLOBAL_TYPES:
            data["global"][stem] = pd.read_csv(csv_path, low_memory=False)

    return data


# ─────────────────────────────────────────────
# 2. LIMPIEZA  — por tipo de DataFrame
# ─────────────────────────────────────────────

def _parse_laptime(series: pd.Series) -> pd.Series:
    """Convierte '1:23.456' → segundos float. Acepta también valores ya numéricos."""
    def _convert(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if ":" in val:
            parts = val.split(":")
            try:
                return float(parts[0]) * 60 + float(parts[1])
            except ValueError:
                return np.nan
        try:
            return float(val)
        except ValueError:
            return np.nan

    return series.apply(_convert)


def _iqr_filter(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    """Elimina outliers extremos usando IQR × factor."""
    if col not in df.columns:
        return df
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[df[col].between(Q1 - factor * IQR, Q3 + factor * IQR)]


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Renombrar columnas a estándar
    rename_dict = {
        "LapTime": "lap_duration",
        "Sector1Time": "duration_sector_1",
        "Sector2Time": "duration_sector_2",
        "Sector3Time": "duration_sector_3",
        "PitInTime": "pit_in_time",
        "PitOutTime": "pit_out_time",
        "LapNumber": "lap_number",
        "Position": "position",
        "DateStart": "date_start",
        "I1Speed": "i1_speed",
        "I2Speed": "i2_speed",
        "IsPitOutLap": "is_pit_out_lap",
        "StSpeed": "st_speed"
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    # Tiempos → segundos
    for col in ["lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3", "pit_in_time", "pit_out_time"]:
        if col in df.columns:
            df[col] = _parse_laptime(df[col])

    # Tipos básicos
    for col in ["lap_number", "position", "driver_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Eliminar vueltas sin tiempo registrado
    if "lap_duration" in df.columns:
        df = df.dropna(subset=["lap_duration"])

    return df.reset_index(drop=True)


def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Renombrar columnas a estándar
    rename_dict = {
        "RPM": "rpm",
        "Speed": "speed",
        "nGear": "n_gear",
        "Throttle": "throttle",
        "Brake": "brake",
        "DRS": "drs",
        "Date": "date"
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    # Timestamp
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Numéricos (telemetría continua)
    num_cols = ["rpm", "speed", "n_gear", "throttle", "brake", "drs"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rellenar huecos de telemetría con valor anterior
    df = df.sort_values(["driver_number", "date"]).reset_index(drop=True)
    df[num_cols] = df.groupby("driver_number")[num_cols].ffill()

    return df


def clean_stints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Renombrar columnas a estándar
    rename_dict = {
        "Stint": "stint_number",
        "LapStart": "lap_start",
        "LapEnd": "lap_end",
        "TyreLife": "tyre_age_at_start",
        "Compound": "compound"
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    for col in ["stint_number", "lap_start", "lap_end", "tyre_age_at_start", "driver_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "compound" in df.columns:
        df["compound"] = df["compound"].str.upper().str.strip()

    return df.reset_index(drop=True)


def clean_pit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Renombrar columnas a estándar
    rename_dict = {
        "PitDuration": "pit_duration",
        "LaneDuration": "lane_duration",
        "StopDuration": "stop_duration",
        "Date": "date"
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    if "pit_duration" in df.columns:
        df["pit_duration"] = _parse_laptime(df["pit_duration"])

    for col in ["lap_number", "driver_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df.reset_index(drop=True)


def clean_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Renombrar columnas a estándar
    rename_dict = {
        "Gap": "gap_to_leader",
        "Interval": "interval",
        "Date": "date"
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    for col in ["gap_to_leader", "interval"]:
        if col in df.columns:
            df[col] = _parse_laptime(df[col])

    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Renombrar columnas a estándar
    rename_dict = {
        "AirTemp": "air_temperature",
        "TrackTemp": "track_temperature",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "WindSpeed": "wind_speed",
        "Date": "date"
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    for col in ["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


CLEANERS = {
    "laps":        clean_laps,
    "car_data":    clean_car_data,
    "stints":      clean_stints,
    "pit":         clean_pit,
    "intervals":   clean_intervals,
    "weather":     clean_weather,
    "race_control": lambda df: df,
    "sessions":    lambda df: df,
    "meetings":    lambda df: df,
    "drivers":     lambda df: df,
}


def clean_all(data: dict) -> dict:
    """Aplica los cleaners a todos los DataFrames."""
    cleaned = {"driver": {}, "global": {}}

    for tipo, drivers in data["driver"].items():
        cleaned["driver"][tipo] = {}
        cleaner = CLEANERS.get(tipo, lambda df: df)
        for num, df in drivers.items():
            print(f"Limpiando tabla: {tipo} (piloto {num})")
            cleaned["driver"][tipo][num] = cleaner(df)

    for tipo, df in data["global"].items():
        print(f"Limpiando tabla global: {tipo}")
        cleaner = CLEANERS.get(tipo, lambda df: df)
        cleaned["global"][tipo] = cleaner(df)

    return cleaned


# ─────────────────────────────────────────────
# 3. INTEGRACIÓN  — merge por carrera
# ─────────────────────────────────────────────

def _concat_drivers(tipo_dict: dict) -> pd.DataFrame:
    """Une los DataFrames de ambos pilotos en uno solo."""
    frames = list(tipo_dict.values())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_master(cleaned: dict) -> pd.DataFrame:
    """
    Construye el DataFrame maestro por vuelta para una carrera.
    Eje principal: (driver_number, lap_number)
    """
    print("Integrando DataFrames por carrera...")
    # ── Base: vueltas ──────────────────────────────────────
    laps = _concat_drivers(cleaned["driver"]["laps"])
    if laps.empty:
        if laps.empty:
            # Mensaje de error más específico: qué drivers y cuántas filas tiene cada uno
            laps_dict = cleaned["driver"].get("laps", {})
            counts = {k: int(v.shape[0]) for k, v in laps_dict.items()}
            drivers = list(counts.keys())
            raise ValueError(
                f"No hay datos de 'laps'. Drivers encontrados en 'laps': {drivers}; "
                f"filas por driver: {counts}; tipos de tablas de driver disponibles: {list(cleaned['driver'].keys())}"
            )

    master = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)

    # ── Stints (por stint, luego por vuelta) ───────────────
    stints = _concat_drivers(cleaned["driver"]["stints"])
    if not stints.empty and "stint_number" in stints.columns and "lap_start" in stints.columns:
        # Expandir stint a nivel de vuelta
        stint_rows = []
        for _, row in stints.iterrows():
            lap_start = row.get("lap_start", np.nan)
            lap_end   = row.get("lap_end",   np.nan)
            if pd.isna(lap_start) or pd.isna(lap_end):
                continue
            for lap in range(int(lap_start), int(lap_end) + 1):
                stint_rows.append({
                    "driver_number": row["driver_number"],
                    "lap_number":    lap,
                    "stint_number":  row.get("stint_number"),
                    "compound":      row.get("compound"),
                    "tyre_age_at_start": row.get("tyre_age_at_start"),
                    # Assuming you don't have FreshTyre anymore based on your dict, remove it if so
                })
        if stint_rows:
            stints_exp = pd.DataFrame(stint_rows)
            master = master.merge(stints_exp, on=["driver_number", "lap_number"], how="left")

    # ── Pit stops ──────────────────────────────────────────
    pit = _concat_drivers(cleaned["driver"]["pit"])
    if not pit.empty and "lap_number" in pit.columns:
        pit_agg = pit.groupby(["driver_number", "lap_number"]).agg(
            pit_duration=("pit_duration", "sum"),
            PitStop=("lap_number", "count"),
        ).reset_index()
        master = master.merge(pit_agg, on=["driver_number", "lap_number"], how="left")
        master["PitStop"] = master["PitStop"].fillna(0).astype(int)
        
        # AGREGA ESTA LÍNEA PARA RELLENAR LOS NULOS:
        master["pit_duration"] = master["pit_duration"].fillna(0.0)

    # ── Intervalos (gap_to_leader al final de la vuelta) ───
    intervals = _concat_drivers(cleaned["driver"]["intervals"])
    if not intervals.empty and "lap_number" in intervals.columns:
        int_agg = intervals.groupby(["driver_number", "lap_number"]).last().reset_index()
        int_cols = ["driver_number", "lap_number"] + [
            c for c in ["gap_to_leader", "interval"] if c in int_agg.columns
        ]
        master = master.merge(int_agg[int_cols], on=["driver_number", "lap_number"], how="left")

    # ── Weather (merge por timestamp más cercano) ──────────
    weather = cleaned["global"].get("weather", pd.DataFrame())
    if not weather.empty and "date" in weather.columns and "date_start" in master.columns:
        master["date_start"] = pd.to_datetime(master["date_start"], errors="coerce", utc=True)
        weather = weather.sort_values("date").reset_index(drop=True)
        master = pd.merge_asof(
            master.sort_values("date_start"),
            weather.rename(columns={"date": "date_start"}),
            on="date_start",
            direction="nearest",
        )

    return master


# ─────────────────────────────────────────────
# 4. ENRIQUECIMIENTO  — features derivadas
# ─────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    print("Enriqueciendo DataFrame maestro con features derivadas...")
    df = df.copy().sort_values(["driver_number", "lap_number"]).reset_index(drop=True)

    # stint_lap: cuántas vueltas lleva el neumático actual
    if "Stint" in df.columns:
        df["stint_lap"] = df.groupby(["driver_number", "Stint"]).cumcount() + 1

    # fuel_load_est: decae linealmente desde vuelta 1 (aprox. 1.8 kg/vuelta F1)
    if "lap_number" in df.columns:
        max_lap = df["lap_number"].max()
        FUEL_LOAD_KG = 105  # kg al inicio
        BURN_RATE    = FUEL_LOAD_KG / max(max_lap, 1)
        df["fuel_load_est"] = (max_lap - df["lap_number"] + 1) * BURN_RATE

    # gap_to_leader: diferencia de tiempo acumulada vs el líder por vuelta
    if "lap_duration" in df.columns:
        lap_min = df.groupby("lap_number")["lap_duration"].min().rename("leader_laptime")
        df = df.merge(lap_min, on="lap_number", how="left")
        df["delta_to_leader"] = df["lap_duration"] - df["leader_laptime"]
        df = df.drop(columns=["leader_laptime"])

    # drs_active: flag si el piloto va significativamente más rápido que su propia media
    if "lap_duration" in df.columns:
        driver_mean = df.groupby("driver_number")["lap_duration"].transform("median")
        df["pace_ratio"] = df["lap_duration"] / driver_mean
        df["is_fast_lap"] = df["pace_ratio"] < 0.98  # < 2 % del ritmo mediano

    return df


# ─────────────────────────────────────────────
# 5. EXPORTACIÓN  — un parquet por carrera
# ─────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, out_dir: Path, race_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{race_name}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  ✓ Guardado: {out_path}  ({len(df):,} filas × {len(df.columns)} cols)")
    return out_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def process_race(folder: Path, out_dir: Path) -> pd.DataFrame:
    race_name = folder.name
    print(f"\n── Carrera: {race_name} ──")

    print("  [1/4] Cargando CSVs...")
    raw = load_race_folder(folder)

    print("  [2/4] Limpiando...")
    cleaned = clean_all(raw)

    print("  [3/4] Integrando...")
    master = build_master(cleaned)

    print("  [4/4] Enriqueciendo features...")
    master = enrich(master)

    save_parquet(master, out_dir, race_name)
    return master


def main():
    parser = argparse.ArgumentParser(description="F1 CSV → Parquet pipeline")
    parser.add_argument("--input",  "-i", type=Path, default=Path("./data"),
                        help="Carpeta raíz con subcarpetas por carrera")
    parser.add_argument("--output", "-o", type=Path, default=Path("./parquet"),
                        help="Carpeta de salida para los parquets")
    args = parser.parse_args()

    # Detectar carpetas de carrera (contienen al menos un CSV)
    race_folders = sorted([
        f for f in args.input.iterdir()
        if f.is_dir() and list(f.glob("*.csv"))
    ])

    # Si no hay subcarpetas, asumir que args.input ES la carpeta de carrera
    if not race_folders:
        race_folders = [args.input]

    print(f"\nF1 Preprocessing Pipeline")
    print(f"Carreras encontradas: {len(race_folders)}")

    summaries = []
    for folder in race_folders:
        try:
            df = process_race(folder, args.output)
            summaries.append({
                "carrera": folder.name,
                "vueltas": len(df),
                "pilotos": df["driver_number"].nunique() if "driver_number" in df.columns else "?",
                "columnas": len(df.columns),
            })
        except Exception as e:
            print(f"  ✗ Error en {folder.name}: {e}")

    print("\n── Resumen ──────────────────────────────")
    for s in summaries:
        print(f"  {s['carrera']}: {s['vueltas']} filas | "
              f"{s['pilotos']} pilotos | {s['columnas']} cols")
    print(f"\nParquets en: {args.output.resolve()}")


if __name__ == "__main__":
    main()