import sys
from realtime_simulation import RealtimeSimulation

def main():
    print("="*60)
    print("::: INICIALIZANDO SIMULACION DE CARRERA EN TIEMPO REAL :::")
    print("="*60)
    
    # Prompt for driver acronym
    try:
        driver = input("Ingrese la sigla del piloto principal (ej: VER, HAM, NOR) [Defecto: VER]: ").strip().upper()
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Cancelado.")
        return
        
    if not driver:
        driver = "VER"
        
    print(f"\n[INFO] Cargando GP de Silverstone (United Kingdom) para el piloto {driver}...")
    sim = RealtimeSimulation(race_name="united_kingdom", driver_acronym=driver)
    sim.start()

if __name__ == "__main__":
    main()
