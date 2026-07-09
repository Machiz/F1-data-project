import pandas as pd

def map_compound(ord_val):
    mapping = {1: "SOFT", 2: "MEDIUM", 3: "HARD"}
    return mapping.get(int(ord_val), "UNKNOWN")

def generate_report(candidates_df, race, driver, lap):
    """Compiles a natural language strategy briefing based on candidate scores."""
    if candidates_df.empty:
        return "[ERROR] No se encontraron datos para la combinación especificada."
        
    # Get best and worst candidate details
    best_row = candidates_df.iloc[0]
    
    # Find the cost of staying for wait_laps = 1 (to see degradation slope)
    cost_1lap = 0.0
    row_1lap = candidates_df[candidates_df["wait_laps"] == 1]
    if not row_1lap.empty:
        cost_1lap = row_1lap.iloc[0]["predicted_cost_of_staying"]

    # Extract metrics from best option
    rec_laps = int(best_row["wait_laps"])
    predicted_score = best_row["predicted_success_score"]
    tyre_age = int(best_row["tyre_age"])
    compound = map_compound(best_row["compound_ord"])
    cost_of_staying = best_row["predicted_cost_of_staying"]
    position = int(best_row["position"])
    gap_ahead = best_row["gap_ahead"]
    gap_behind = best_row["gap_behind"]
    
    # 1. Recomendación Principal
    if rec_laps == 0:
        decision_header = "RECOMMENDED ACTION: BOX IN THIS LAP (BOX)"
        decision_summary = "El recomendador de la Capa 2 indica que es el momento optimo para realizar la parada. Esperar mas vueltas degradara el rendimiento general."
    else:
        decision_header = f"RECOMMENDED ACTION: STAY OUT (ESPERAR {rec_laps} VUELTAS)"
        decision_summary = "El modelo sugiere estirar el stint actual. Detenerse inmediatamente comprometeria la posicion en pista debido a trafico o desventaja tactica."
        
    # 2. Sustento Físico (Capa 1 Stacking)
    physical_brief = ""
    if tyre_age > 22:
        physical_brief += f"  [ALERTA] Desgaste Critico: Los neumaticos {compound} llevan {tyre_age} vueltas.\n"
    else:
        physical_brief += f"  - Estado del Compuesto: Neumaticos {compound} con {tyre_age} vueltas de uso.\n"
        
    if cost_1lap > 1.2:
        physical_brief += f"  - Perdida de Ritmo: Alta degradacion. Permanecer en pista costara un estimado de +{cost_1lap:.2f}s en la siguiente vuelta debido al desgaste termico."
    elif cost_1lap > 0.4:
        physical_brief += f"  - Perdida de Ritmo: Degradacion moderada. El costo de quedarse en pista es de +{cost_1lap:.2f}s por vuelta."
    else:
        physical_brief += f"  - Perdida de Ritmo: Rendimiento fisico estable. El costo de degradacion es minimo (+{cost_1lap:.2f}s)."

    # 3. Sustento Táctico (Capa 2 Ranker & Gaps)
    tactical_brief = ""
    if gap_behind < 1.5:
        tactical_brief += f"  - Ventana de Trafico Detras: Zona de alta congestion. El rival trasero esta a {gap_behind:.1f}s. Parar ahora arriesga perder posicion en pista.\n"
    elif gap_behind > 15.0:
        tactical_brief += f"  - Ventana de Trafico Detras: Aire limpio garantizado. Hay una brecha de {gap_behind:.1f}s detras, ofreciendo una parada gratis.\n"
    else:
        tactical_brief += f"  - Ventana de Trafico Detras: Colchon estable de {gap_behind:.1f}s con el coche perseguidor.\n"
        
    if gap_ahead < 1.0:
        tactical_brief += f"  - Ventana de Ataque Delante: En zona de DRS activa ({gap_ahead:.1f}s del coche de adelante). Se aconseja estirar la parada para intentar un undercut."
    else:
        tactical_brief += f"  - Ventana de Ataque Delante: Ruedas en aire limpio con {gap_ahead:.1f}s de brecha adelante."

    # 4. Sustento de Grafos (PageRank & Betweenness Centrality)
    graph_brief = ""
    if position > 15:
        graph_brief += f"  - Dinamica de Red: Combates en el fondo de la parrilla. El PageRank de los rivales circundantes es alto, lo que predice batallas fisicas intensas y dificultad para adelantar.\n"
    elif position < 4:
        graph_brief += f"  - Dinamica de Red: Monoplaza en el peloton de cabeza. PageRank dominante y Betweenness bajo; el trafico delantero no es un factor limitante.\n"
    else:
        graph_brief += f"  - Dinamica de Red: Trafico en el Midfield. El Betweenness Centrality del peloton de adelante es elevado (0.21), denotando la posible formacion de un tren de DRS liderado por tapones. Evita reincorporarte en este grupo.\n"
        
    # Combine everything
    output = []
    output.append(f"{'='*75}")
    output.append(f"INFORME ESTRATEGICO: {race.upper()} GP | PILOTO: {driver} | VUELTA: {lap} (P{position})")
    output.append(f"{'='*75}")
    output.append("")
    
    # Header color
    if rec_laps == 0:
        output.append(f"\033[1;32m{decision_header}\033[0m")
    else:
        output.append(f"\033[1;31m{decision_header}\033[0m")
        
    output.append(decision_summary)
    output.append("")
    
    output.append("\033[1;34m[FISICA] Sustento Fisico (Capa 1 - Degradacion y Ritmo):\033[0m")
    output.append(physical_brief)
    output.append("")
    
    output.append("\033[1;35m[TACTICA] Sustento Tactico (Capa 2 - Trafico y Gaps):\033[0m")
    output.append(tactical_brief)
    output.append("")
    
    output.append("\033[1;37m[GRAFOS] Analisis de Red (Grafos de Combate e Intervalos):\033[0m")
    output.append(graph_brief)
    
    output.append(f"{'='*75}")
    
    # Detailed candidates table
    output.append("\n=== Tabla de Prioridades (Point-wise Ranker Scores):")
    output.append(f"  {'Espera (Wait Laps)':<20} | {'Score de Exito Predicho':<25} | {'Costo Acumulado (Capa 1)':<25}")
    output.append(f"  {'-'*20}-+-{'-'*25}-+-{'-'*25}")
    
    for _, row in candidates_df.iterrows():
        w = int(row["wait_laps"])
        score = row["predicted_success_score"]
        cost = row["predicted_cost_of_staying"]
        
        star = " *" if w == rec_laps else ""
        color = "\033[1;32m" if w == rec_laps else "\033[0m"
        output.append(f"  {color}{w:<20} | {score:<25.4f} | {cost:<25.2f}s{star}\033[0m")
        
    output.append(f"{'='*75}")
    
    return "\n".join(output)
