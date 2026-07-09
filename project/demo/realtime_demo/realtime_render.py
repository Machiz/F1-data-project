import os

def map_compound(ord_val):
    mapping = {1: "SOFT", 2: "MEDIUM", 3: "HARD"}
    return mapping.get(int(ord_val), "UNKNOWN")

def render_dashboard(candidates_df, race, driver, lap, leaders, total_laps, speed_multiplier, is_paused):
    """Clears console and renders a colored F1 Pit Wall monitor layout."""
    # 1. Clear terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    if candidates_df.empty:
        print("[ERROR] No hay datos disponibles para renderizar la vuelta actual.")
        return
        
    best_row = candidates_df.iloc[0]
    rec_laps = int(best_row["wait_laps"])
    tyre_age = int(best_row["tyre_age"])
    compound = map_compound(best_row["compound_ord"])
    cost_of_staying = best_row["predicted_cost_of_staying"]
    position = int(best_row["position"])
    gap_ahead = best_row["gap_ahead"]
    gap_behind = best_row["gap_behind"]
    
    # 2. Build progress bar
    pct = int((lap / total_laps) * 15) if total_laps > 0 else 0
    progress_bar = "[" + "█" * pct + "░" * (15 - pct) + "]"
    
    # 3. Header
    speed_lbl = "2x (Acelerado)" if speed_multiplier == 2.0 else "1x (Normal)"
    status_lbl = "\033[1;31mPAUSADO\033[0m" if is_paused else "\033[1;32mCORRIENDO\033[0m"
    
    print(f"\033[1;36m{'='*78}\033[0m")
    print(f"\033[1;33m::: MONITOR TACTICO DE BOXES | GP: {race.upper()} | MODO: SIMULACION EN VIVO :::\033[0m")
    print(f"   Velocidad: {speed_lbl} | Estado: {status_lbl} | Control: [Espacio] Velocidad | [P] Pausa")
    print(f"\033[1;36m{'='*78}\033[0m\n")
    
    # 4. Timeline and Leaderboard
    print(f"[TIME] Carrera: {progress_bar} Vuelta {lap}/{total_laps}")
    print(f"[RANK] Top 5:  " + " -> ".join(leaders))
    print("")
    
    # 5. Telemetry Card
    print("\033[1;34m[TELEMETRIA] TELEMETRIA EN VIVO (Piloto Principal):\033[0m")
    print(f"  * Piloto: \033[1;37m{driver}\033[0m | Posicion: \033[1;37mP{position}\033[0m | Compuesto: \033[1;37m{compound}\033[0m ({tyre_age} vueltas)")
    print(f"  * Intervalo Delante: \033[1;37m{gap_ahead:.1f}s\033[0m (DRS {'ACTIVO' if gap_ahead < 1.0 else 'INACTIVO'})")
    print(f"  * Intervalo Detras:  \033[1;37m{gap_behind:.1f}s\033[0m")
    print("")
    
    # 6. Strategic Advice Card
    print("\033[1;35m[IA RECOMMENDATION] RECOMENDACION ESTRATEGICA DE IA:\033[0m")
    if rec_laps == 0:
        print("  \033[1;37;41m   [BOX] BOX THIS LAP -- BOX THIS LAP -- PARAR EN BOXES AHORA   \033[0m")
        print("  * Sustento Tactico: El Ranker predice que la ventana de pits optimo se cierra. Parar ahora maximiza la ventaja.")
    else:
        print(f"  \033[1;30;43m   [STAY OUT] STAY OUT -- ESPERAR {rec_laps} VUELTAS -- CONTINUAR EN PISTA   \033[0m")
        print(f"  * Sustento Tactico: Evitar paradas. Se proyecta que estirar el stint es {predicted_score_delta(candidates_df):.2f} ptos mas eficiente.")
        
    print(f"  * Costo Fisico Proyectado (Capa 1 Stacking): \033[1;37m+{cost_of_staying:.2f}s\033[0m perdidos por degradacion si no paramos.")
    print("")
    
    # 7. Priorities Table
    print("\033[1;37m=== Tabla de Prioridades Contrafacticas (Point-wise Ranker): ===\033[0m")
    print(f"  {'Espera (Wait Laps)':<20} | {'Score de Exito Predicho':<25} | {'Costo Acumulado (Capa 1)':<25}")
    print(f"  {'-'*20}-+-{'-'*25}-+-{'-'*25}")
    
    for _, row in candidates_df.iterrows():
        w = int(row["wait_laps"])
        score = row["predicted_success_score"]
        cost = row["predicted_cost_of_staying"]
        
        star = " *" if w == rec_laps else ""
        color = "\033[1;32m" if w == rec_laps else "\033[0m"
        print(f"  {color}{w:<20} | {score:<25.4f} | {cost:<25.2f}s{star}\033[0m")
        
    print(f"\033[1;36m{'='*78}\033[0m")
    print("\033[1;30mComandos: [Espacio] Alternar Vel. | [P] Pausar | [Enter] Siguiente Vuelta | [Q] Salir\033[0m")

def predicted_score_delta(df):
    """Calcula el delta de score entre la opción óptima de esperar y la opción de parar ya (w=0)."""
    if len(df) < 2:
        return 0.0
    best_score = df.iloc[0]["predicted_success_score"]
    row_0 = df[df["wait_laps"] == 0]
    score_0 = row_0.iloc[0]["predicted_success_score"] if not row_0.empty else 0.0
    return abs(best_score - score_0)
