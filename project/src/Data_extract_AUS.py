import requests
import pandas as pd
import os

class OpenF1StrategyExtractor:
    def __init__(self, base_url="https://api.openf1.org/v1"):
        self.base_url = base_url
        self.output_dir = "data/raw/australia_2026_H2H"
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_endpoint(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                
                # Nomenclatura dinámica: Si filtramos por piloto, añadir su número al nombre del CSV
                driver_num = params.get('driver_number', '')
                suffix = f"_driver_{driver_num}" if driver_num else ""
                file_name = f"{endpoint}{suffix}.csv"
                
                file_path = os.path.join(self.output_dir, file_name)
                df.to_csv(file_path, index=False)
                print(f"✅ Guardado: {file_name} ({len(df)} filas)")
                return df
            else:
                print(f"⚠️ Sin datos para {endpoint} con esos parámetros.")
                return None
                
        except Exception as e:
            print(f"❌ Error al extraer {endpoint}: {e}")
            return None

def main():
    extractor = OpenF1StrategyExtractor()
    
    print("--- 1. Inicializando Sistema de Estrategia: Australia 2026 ---")
    meetings = extractor.fetch_endpoint("meetings", params={"year": 2026, "country_name": "Australia"})
    
    if meetings is not None and not meetings.empty:
        m_key = meetings.iloc[0]['meeting_key']
        
        sessions = extractor.fetch_endpoint("sessions", params={"meeting_key": m_key, "session_name": "Race"})
        
        if sessions is not None and not sessions.empty:
            s_key = sessions.iloc[0]['session_key']
            print(f"\n[OK] Sesión de Carrera Encontrada (Key: {s_key})")
            
            # ---------------------------------------------------------
            # BLOQUE A: Contexto de Pista (Aplica a todos)
            # ---------------------------------------------------------
            print("\n--- 2. Descargando Contexto de Pista ---")
            context_endpoints = ["weather", "race_control", "drivers"]
            for ep in context_endpoints:
                extractor.fetch_endpoint(ep, params={"session_key": s_key})

            # ---------------------------------------------------------
            # BLOQUE B: Extracción Táctica 16 vs 44
            # ---------------------------------------------------------
            print("\n--- 3. Descargando Telemetría H2H (16 vs 44) ---")
            target_drivers = [16, 44]
            
            # Endpoints críticos para analizar el duelo directo
            h2h_endpoints = ["car_data", "location", "intervals", "stints", "pit", "laps"]
            
            for driver in target_drivers:
                print(f"\n>> Procesando métricas para el Coche {driver}...")
                for ep in h2h_endpoints:
                    extractor.fetch_endpoint(ep, params={"session_key": s_key, "driver_number": driver})
                    
        else:
            print("Error: No se encontró la sesión de carrera (Race).")
    else:
        print("Error: No se encontró el evento de Australia 2026.")

if __name__ == "__main__":
    main()