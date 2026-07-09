import sys
from chatbot_engine import ChatbotEngine
from template_generator import generate_report

def print_splash():
    print(r"""
\033[1;31m  ______   __    __       _______   ______    ______   _______   _______   _______  
 /      | /  |  /  |     /       | /      |  /      | /       | /       | /       | 
/$$$$$$/  $$ |  $$ |     $$$$$$$/  $$$$$$/  /$$$$$$/  $$$$$$$/  $$$$$$$/  $$$$$$$/  
$$ |  __  $$ |__$$ |        $$ |     $$ |   $$ |__    $$ |__    $$ |__    $$ |__    
$$ | /  | $$    $$ |        $$ |     $$ |   $$    |   $$    |   $$    |   $$    |   
$$ | $$ | $$$$$$$$ |        $$ |     $$ |   $$$$$$/   $$$$$$/   $$$$$$/   $$$$$$/   
$$ \_$$ | $$ |  $$ |        $$ |    _$$ |_  $$ |_____ $$ |_____ $$ |_____ $$ |_____ 
$$$$$$/   $$/   $$/         $$/     $$$$$$/ $$$$$$$$/ $$$$$$$$/ $$$$$$$$/ $$$$$$$$/ 
                                                                                    
            ::: MOTOR DE RECOMENDACIÓN TÁCTICA F1 (Muro de Boxes) :::\033[0m
""")
    print("\033[1;36mPara consultar una decisión estratégica, usa el formato:\033[0m")
    print("  \033[1;37m[carrera] [piloto] [vuelta]\033[0m  -> (Ejemplo: \033[1;32munited_states VER 39\033[0m)")
    print("\n\033[1;33mComandos adicionales:\033[0m")
    print("  \033[1;37mlist\033[0m  -> Muestra las carreras y pilotos disponibles")
    print("  \033[1;37mhelp\033[0m  -> Muestra las instrucciones de uso")
    print("  \033[1;37mexit\033[0m  -> Cierra el asistente de boxes\n")

def start_chatbot_loop():
    engine = ChatbotEngine()
    try:
        engine.load_resources()
    except Exception as e:
        print(f"\033[1;31m[ERROR] Error al cargar recursos: {e}\033[0m")
        return

    races, drivers = engine.get_available_sessions()
    print_splash()

    while True:
        try:
            user_input = input("\033[1;32m[PitWall-IA] > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[BOXES] Cerrando canal de boxes. ¡Buen GP!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["exit", "quit", "q"]:
            print("[BOXES] Cerrando canal de boxes. ¡Buen GP!")
            break
        elif cmd == "help":
            print_splash()
            continue
        elif cmd == "list":
            print("\n::: \033[1;33mCarreras Disponibles:\033[0m")
            print("  " + ", ".join(races))
            print("\n::: \033[1;33mPilotos Disponibles:\033[0m")
            # Group print for drivers
            for i in range(0, len(drivers), 10):
                print("  " + ", ".join(drivers[i:i+10]))
            print("")
            continue

        # Parse query
        parts = user_input.split()
        if len(parts) < 3:
            print("\033[1;31m[ERROR] Entrada inválida. Formato: [carrera] [piloto] [vuelta] (Ej: united_states VER 39)\033[0m")
            continue

        race_in = parts[0].lower()
        driver_in = parts[1].upper()
        lap_str = parts[2]

        # Validations
        if race_in not in races:
            # Check for partial matches
            matched_races = [r for r in races if race_in in r]
            if len(matched_races) == 1:
                race_in = matched_races[0]
            else:
                print(f"\033[1;31m[ERROR] Carrera '{race_in}' no reconocida. Carreras en dataset: {', '.join(races)}\033[0m")
                continue

        if driver_in not in drivers:
            print(f"\033[1;31m[ERROR] Piloto '{driver_in}' no reconocido. Usa 'list' para ver los pilotos activos.\033[0m")
            continue

        try:
            lap_in = int(lap_str)
        except ValueError:
            print(f"\033[1;31m[ERROR] El número de vuelta '{lap_str}' debe ser un entero.\033[0m")
            continue

        # Validate lap range
        min_lap, max_lap = engine.get_lap_range(race_in, driver_in)
        if min_lap == 0 and max_lap == 0:
            print(f"\033[1;31m[ERROR] No hay datos para {driver_in} en la carrera {race_in}.\033[0m")
            continue

        if lap_in < min_lap or lap_in > max_lap:
            print(f"\033[1;31m[ERROR] Vuelta fuera de rango. Rango disponible para {driver_in} en {race_in}: {min_lap} - {max_lap}\033[0m")
            continue

        # Run strategy assistant
        print(f"\n[RUN] Procesando telemetría y ejecutando modelos para {driver_in} (Vuelta {lap_in})...")
        candidates = engine.get_predictions(race_in, driver_in, lap_in)
        
        if candidates.empty:
            print("\033[1;31m[ERROR] Error de correspondencia: no hay candidatos disponibles para esta vuelta.\033[0m")
            continue

        report = generate_report(candidates, race_in, driver_in, lap_in)
        print(report)
        print("")
