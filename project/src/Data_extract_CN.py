import requests
import pandas as pd
import os

class OpenF1ChinaEngineerExtractor:
    def __init__(self, base_url="https://api.openf1.org/v1"):
        self.base_url = base_url
        self.output_dir = "data/raw/china_2026_perf"
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_endpoint(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                
                # Nomenclatura para análisis de rendimiento individual
                driver_num = params.get('driver_number', '')
                suffix = f"_car_{driver_num}" if driver_num else ""
                file_name = f"{endpoint}{suffix}.csv"
                
                file_path = os.path.join(self.output_dir, file_name)
                df.to_csv(file_path, index=False)
                print(f"✅ [DATA INGESTED]: {file_name} - {len(df)} registros.")
                return df
            return None
        except Exception as e:
            print(f"❌ [API ERROR] en {endpoint}: {e}")
            return None

def main():
    extractor = OpenF1ChinaEngineerExtractor()
    
    print("--- Diagnostic System: Gran Premio de China 2026 ---")
    
    # 1. Localizar el evento en Shanghái
    meetings = extractor.fetch_endpoint("meetings", params={"year": 2026, "country_name": "China"})
    
    if meetings is not None and not meetings.empty:
        m_key = meetings.iloc[0]['meeting_key']
        
        # 2. Localizar la sesión de Carrera
        sessions = extractor.fetch_endpoint("sessions", params={"meeting_key": m_key, "session_name": "Race"})
        
        if sessions is not None and not sessions.empty:
            s_key = sessions.iloc[0]['session_key']
            print(f"\n[ENGINEERING MODE] Sesión de Carrera: {s_key}")

            # 3. Extracción de Contexto (Clima y Control de Carrera para filtrar nulos)
            print("\n--- Extrayendo Contexto Ambiental y Normativo ---")
            for ep in ["weather", "race_control", "drivers"]:
                extractor.fetch_endpoint(ep, params={"session_key": s_key})

            # 4. Extracción de Rendimiento Específico (Coches 16 y 44)
            print("\n--- Extrayendo Métricas de Rendimiento (Car 16 & 44) ---")
            target_cars = [16, 44]
            # Endpoints para medir eficiencia, degradación y ritmo
            perf_endpoints = ["car_data", "location", "intervals", "stints", "pit", "laps"]
            
            for car in target_cars:
                print(f"\n>> Analizando sensores del monoplaza #{car}...")
                for ep in perf_endpoints:
                    # Filtramos por session y driver para optimizar volumen
                    extractor.fetch_endpoint(ep, params={"session_key": s_key, "driver_number": car})
                    
        else:
            print("Error: Sesión de carrera no encontrada.")
    else:
        print("Error: GP de China 2026 no registrado en la API.")

if __name__ == "__main__":
    main()