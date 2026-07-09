"""
Asistente tactico de estrategia de pit stops en tiempo real (simulado).

Frontend Streamlit que reutiliza RealtimePipeline sin modificar el backend.
Ubicacion prevista en el repo: project/demo/realtime_demo/app_streamlit.py
Ejecucion:  streamlit run demo/realtime_demo/app_streamlit.py

Banner de tres vias:
  - wait_laps == 0  -> BOX (parar ahora)
  - wait_laps 1..5  -> STAY (ventana optima en k vueltas)
  - wait_laps == 6  -> NO_PIT / STAY OUT (no parar en la ventana)
"""

import sys
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- Resolucion de rutas e import del pipeline ---------------------------------
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent.parent
MASTER_DIR = PROJECT_DIR / "data" / "processed" / "master"

# Asegura que el modulo del pipeline sea importable al lanzar con streamlit
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from realtime_pipeline import RealtimePipeline  # noqa: E402

# Carreras presentes en el dataset de entrenamiento de las dos capas.
# Cualquier otra (p. ej. united_kingdom) es una carrera NO vista -> se advierte.
TRAINING_RACES = {"australia", "china", "japan", "united_states"}

# Colores de compuesto reales de F1
COMPOUND_COLORS = {"SOFT": "#e10600", "MEDIUM": "#ffd12e", "HARD": "#f0f0f0"}

# --- Configuracion de pagina y estilo -----------------------------------------
st.set_page_config(
    page_title="F1 Pit-Wall | Asistente Tactico",
    page_icon="\U0001F3CE",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; }
      .pitwall-title {
          font-size: 1.9rem; font-weight: 800; letter-spacing: .5px;
          color: #f5f5f5; margin-bottom: .1rem;
      }
      .pitwall-sub { color: #9aa0a6; font-size: .9rem; margin-bottom: 1rem; }
      .banner-box {
          background: linear-gradient(90deg, #e10600, #7a0300);
          color: #fff; font-weight: 800; font-size: 1.5rem;
          padding: 18px 22px; border-radius: 10px; text-align: center;
          letter-spacing: 1px; box-shadow: 0 0 18px rgba(225,6,0,.45);
      }
      .banner-stay {
          background: linear-gradient(90deg, #1db954, #0b6b30);
          color: #fff; font-weight: 800; font-size: 1.5rem;
          padding: 18px 22px; border-radius: 10px; text-align: center;
          letter-spacing: 1px; box-shadow: 0 0 18px rgba(29,185,84,.35);
      }
      .tyre-badge {
          display: inline-block; padding: 4px 14px; border-radius: 14px;
          font-weight: 700; color: #111; font-size: .85rem;
      }
      .metric-card {
          background: #161b22; border: 1px solid #262c36; border-radius: 10px;
          padding: 12px 16px; text-align: center;
      }
      .metric-card .lbl { color: #8b949e; font-size: .72rem; text-transform: uppercase; letter-spacing: .5px; }
      .metric-card .val { color: #f5f5f5; font-size: 1.35rem; font-weight: 700; }
      .subnote { color: #6e7681; font-size: .78rem; line-height: 1.5; }
      .subnote code { color: #9aa0a6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Utilidades cacheadas ------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_races():
    """Devuelve las carreras con archivo master disponible."""
    files = glob.glob(str(MASTER_DIR / "*_*master*.parquet"))
    races = set()
    for f in files:
        stem = Path(f).stem
        # nombre = <race>_<...>master...  -> tomamos el prefijo hasta el primer _<year/algo>
        races.add(stem.split("_")[0] if "_" not in stem else stem.split("master")[0].split("_")[0])
    # Reconstruccion robusta: prioriza coincidencia directa con nombres conocidos
    known = TRAINING_RACES | {"united_kingdom"}
    detected = {r for r in known if glob.glob(str(MASTER_DIR / f"{r}_*master*.parquet"))}
    return sorted(detected) if detected else sorted(races)


@st.cache_data(show_spinner=False)
def available_drivers(race_name):
    """Lista (numero, acronimo) de pilotos presentes en el master de la carrera."""
    files = glob.glob(str(MASTER_DIR / f"{race_name}_*master*.parquet"))
    if not files:
        return []
    df = pd.read_parquet(files[0], columns=["driver_number"])
    nums = sorted(int(n) for n in df["driver_number"].dropna().unique())

    tmp = RealtimePipeline(race_name=race_name, driver_acronym="VER")
    tmp._build_driver_mappings()
    out = []
    for n in nums:
        acro = tmp.driver_to_acronym.get(n)
        if acro:  # solo pilotos con acronimo reconocible por el pipeline
            out.append((n, acro))
    return out


@st.cache_resource(show_spinner="Cargando modelos y datos de carrera...")
def get_pipeline(race_name, driver_acronym):
    p = RealtimePipeline(race_name=race_name, driver_acronym=driver_acronym)
    p.load_resources()
    return p


# --- Barra lateral -------------------------------------------------------------
st.sidebar.markdown("### Configuracion de sesion")

races = list_races()
default_race = next((r for r in races if r in TRAINING_RACES), races[0] if races else None)
if not races:
    st.error(f"No se encontraron archivos master en {MASTER_DIR}")
    st.stop()

race = st.sidebar.selectbox("Gran Premio", races, index=races.index(default_race))

drivers = available_drivers(race)
if not drivers:
    st.error(f"No hay pilotos reconocibles en el master de {race}.")
    st.stop()

driver_labels = [a for _, a in drivers]
default_driver_idx = driver_labels.index("VER") if "VER" in driver_labels else 0
driver_acro = st.sidebar.selectbox("Piloto", driver_labels, index=default_driver_idx)

autoplay = st.sidebar.toggle("Reproduccion automatica", value=False)
delay = st.sidebar.slider("Velocidad (seg/vuelta)", 0.2, 3.0, 1.0, 0.1)

# --- Carga del pipeline --------------------------------------------------------
try:
    pipe = get_pipeline(race, driver_acro)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo inicializar el pipeline: {type(e).__name__}: {e}")
    st.stop()

total_laps = pipe.get_total_laps()
if total_laps <= 0:
    st.error("El master no contiene vueltas para esta carrera.")
    st.stop()

# Estado de vuelta persistente por (carrera, piloto)
state_key = f"lap_{race}_{driver_acro}"
if state_key not in st.session_state:
    st.session_state[state_key] = min(10, total_laps)

lap = st.sidebar.slider("Vuelta actual", 1, total_laps, st.session_state[state_key])
st.session_state[state_key] = lap

# --- Cabecera ------------------------------------------------------------------
st.markdown('<div class="pitwall-title">\U0001F3CE\uFE0F Muro de boxes — Asistente tactico</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="pitwall-sub">{race.replace("_", " ").title()} · Piloto {driver_acro} · '
    f"Vuelta {lap} / {total_laps}</div>",
    unsafe_allow_html=True,
)

if race not in TRAINING_RACES:
    st.warning(
        f"**{race.replace('_', ' ').title()}** no formo parte de los datos de entrenamiento "
        "(australia, china, japan, united_states). Las recomendaciones son extrapolaciones "
        "sobre una carrera no vista y deben leerse con cautela."
    )

# --- Tablero de posiciones -----------------------------------------------------
leaders = pipe.get_leaderboard_at_lap(lap)
if leaders:
    st.caption("Clasificacion (top 5): " + "  |  ".join(leaders))

# --- Inferencia ----------------------------------------------------------------
cands = pipe.get_realtime_inference(lap)
if cands.empty:
    st.info(f"No hay telemetria disponible para {driver_acro} hasta la vuelta {lap}.")
    st.stop()

best = cands.iloc[0]
wait_best = int(best["wait_laps"])

# --- Telemetria actual (tarjetas) ---------------------------------------------
cur = pipe.df_master[
    (pipe.df_master["driver_number"] == pipe.driver_num)
    & (pipe.df_master["lap_number"] == lap)
]
if not cur.empty:
    cur = cur.iloc[0]
    compound = str(cur.get("compound", "MEDIUM")).upper()
    tyre_color = COMPOUND_COLORS.get(compound, "#cccccc")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="lbl">Neumatico</div>'
            f'<div class="val"><span class="tyre-badge" style="background:{tyre_color}">{compound}</span></div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="lbl">Edad neumatico</div>'
            f'<div class="val">{int(cur.get("tyre_age", 0))} v</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="lbl">Posicion</div>'
            f'<div class="val">P{int(cur.get("position", 0))}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="lbl">Ultima vuelta</div>'
            f'<div class="val">{cur.get("lap_duration", float("nan")):.2f}s</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# --- Banner de recomendacion (3 vias) -----------------------------------------
if wait_best == 6:
    st.markdown(
        '<div class="banner-stay">[STAY OUT] MANTENER EN PISTA — NO PARAR EN LA VENTANA</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "El ranker no identifica una ventana de parada favorable en las proximas "
        "5 vueltas; la accion optima es permanecer fuera."
    )
elif wait_best == 0:
    st.markdown(
        '<div class="banner-box">[BOX] PARAR AHORA</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Costo fisico proyectado (Capa 1): +{best['predicted_cost_of_staying']:.2f}s si no paramos."
    )
else:
    st.markdown(
        f'<div class="banner-stay">[STAY] MANTENER POSICION — VENTANA OPTIMA EN {wait_best} '
        f"VUELTA{'S' if wait_best > 1 else ''}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Esperar {wait_best} vuelta(s) maximiza el score "
        f"(costo acumulado proyectado: {best['predicted_cost_of_staying']:.2f}s)."
    )

# --- Tabla contrafactual + graficos -------------------------------------------
def action_label(w):
    w = int(w)
    if w == 6:
        return "NO PARAR"
    if w == 0:
        return "Parar ahora"
    return f"Esperar {w}"


left, right = st.columns([1.15, 1])

with left:
    st.subheader("Prioridades contrafacticas (ranker point-wise)")
    tabla = cands[["wait_laps", "predicted_success_score", "predicted_cost_of_staying"]].copy()
    tabla.insert(0, "Accion", tabla["wait_laps"].apply(action_label))
    tabla = tabla.rename(
        columns={
            "wait_laps": "Codigo",
            "predicted_success_score": "Score de exito",
            "predicted_cost_of_staying": "Costo acumulado (s)",
        }
    ).reset_index(drop=True)
    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score de exito": st.column_config.NumberColumn(format="%.4f"),
            "Costo acumulado (s)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    chart_df = cands[["wait_laps", "predicted_success_score"]].copy()
    chart_df["wait_laps"] = chart_df["wait_laps"].apply(action_label)
    st.bar_chart(chart_df.set_index("wait_laps")["predicted_success_score"], height=220)

with right:
    st.subheader("Ritmo del stint (historial en vivo)")
    hist = pipe.df_master[
        (pipe.df_master["driver_number"] == pipe.driver_num)
        & (pipe.df_master["lap_number"] <= lap)
    ][["lap_number", "lap_duration"]].set_index("lap_number")
    st.line_chart(hist, height=300)
    st.caption("Duracion de vuelta acumulada hasta la vuelta actual (sin datos futuros).")

# --- Nota de transparencia (auditoria de sesgo) -------------------------------
st.divider()
st.markdown(
    '<p class="subnote">Prototipo funcional de asistente tactico en tiempo real simulado. '
    "El target del ranker incorpora una accion explicita NO_PIT (codigo 6) para evitar que "
    "wait_laps=0 gane por defecto en vueltas sin ventana de parada real. La auditoria del "
    "sistema (ver <code>reports/ranking_system/ranking_bias_audit.md</code>) reporta el "
    "desempeno por clase; las recomendaciones no constituyen estrategia optima garantizada.</p>",
    unsafe_allow_html=True,
)

# --- Reproduccion automatica ---------------------------------------------------
if autoplay and lap < total_laps:
    time.sleep(delay)
    st.session_state[state_key] = lap + 1
    st.rerun()
