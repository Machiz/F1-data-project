import requests
import pandas as pd
import os
import argparse
import time 

class OpenF1DataExtractor:
    def __init__(self, year=2026, base_url="https://api.openf1.org/v1"):
        self.base_url = base_url
        self.year = year
        self.base_output_dir = "data/raw"

    def fetch_endpoint(self, endpoint, params=None, max_retries=6, base_delay=5):
        """Descarga datos con manejo avanzado de Rate Limiting estricto."""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=20)
                
                # Manejo del Error 429: Demasiadas peticiones
                if response.status_code == 429:
                    # Fórmula ajustada: 5s, 10s, 20s, 40s, 80s...
                    wait_time = base_delay * (2 ** attempt) 
                    print(f"    ⚠️ Límite de API (429) en {endpoint}. Castigo activo. Esperando {wait_time}s...")
                    time.sleep(wait_time)
                    continue 
                
                # Manejo del Error 404: Dato no encontrado
                if response.status_code == 404:
                    return pd.DataFrame()
                
                response.raise_for_status()
                data = response.json()
                if data:
                    return pd.DataFrame(data)
                return pd.DataFrame()
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Intento {attempt+1}/{max_retries} falló en {endpoint}: {e}")
                time.sleep(base_delay)
                
        print(f"❌ Se omitió {endpoint} tras {max_retries} reintentos fallidos. El servidor no responde.")
        return pd.DataFrame()

    def extract_race(self, country_name):
        """Orquesta la descarga completa de una carrera."""
        print(f"\n{'='*60}\n🏁 Iniciando extracción para: {country_name} {self.year}\n{'='*60}")
        
        # 1. Obtener Meeting y Session
        meetings = self.fetch_endpoint("meetings", params={"year": self.year, "country_name": country_name})
        if meetings.empty:
            print(f"⚠️ No se encontró el evento para {country_name}.")
            return

        m_key = meetings.iloc[0]['meeting_key']
        sessions = self.fetch_endpoint("sessions", params={"meeting_key": m_key, "session_name": "Race"})
        
        if sessions.empty:
            print(f"⚠️ No se encontró la sesión de carrera para {country_name}.")
            return

        s_key = sessions.iloc[0]['session_key']
        print(f"[OK] Sesión de Carrera Encontrada (Key: {s_key})")

        # Crear directorio estructurado: data/raw/australia_2026/
        folder_name = f"{country_name.lower().replace(' ', '_')}_{self.year}"
        output_dir = os.path.join(self.base_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # 2. Descargar Datos Globales y obtener listado de pilotos
        global_endpoints = ["weather", "race_control", "drivers", "sessions", "meetings"]
        print("\n--- 1. Descargando Datos Globales ---")
        
        drivers_list = []
        for ep in global_endpoints:
            if ep == "meetings":
                df = meetings
            elif ep == "sessions":
                df = sessions
            else:
                df = self.fetch_endpoint(ep, params={"session_key": s_key})
            
            if not df.empty:
                df.to_csv(os.path.join(output_dir, f"{ep}.csv"), index=False)
                print(f"  ✅ {ep}.csv guardado ({len(df)} filas)")
                
                # Rescatar los números de los 20 pilotos para usarlos después
                if ep == "drivers" and "driver_number" in df.columns:
                    drivers_list = df["driver_number"].unique().tolist()
        
        if not drivers_list:
            print("⚠️ No se pudo obtener la lista de pilotos. Deteniendo extracción.")
            return

        # 3. Descargar Datos por Entidad (Agrupando a TODOS los pilotos)
        print(f"\n--- 2. Descargando Entidades para los {len(drivers_list)} Pilotos ---")
        entity_endpoints = ["laps", "stints", "pit", "intervals", "car_data", "location"]

        for ep in entity_endpoints:
            print(f"⏳ Extrayendo {ep}...")
            all_drivers_data = []
            
            for driver in drivers_list:
                # ESTRANGULAMIENTO: Pausa obligatoria entre cada piloto para no enfurecer a la API
                time.sleep(1.5) 
                
                df_driver = self.fetch_endpoint(ep, params={"session_key": s_key, "driver_number": driver})
                
                if not df_driver.empty and not df_driver.isna().all().all():
                    all_drivers_data.append(df_driver)
            
            if all_drivers_data:
                # CONCATENACIÓN CRÍTICA: Unimos a los 20 pilotos en un solo DataFrame
                df_final = pd.concat(all_drivers_data, ignore_index=True)
                file_path = os.path.join(output_dir, f"{ep}.csv")
                df_final.to_csv(file_path, index=False)
                print(f"  ✅ Guardado: {ep}.csv -> Datos unificados de todos los pilotos ({len(df_final):,} filas)")
            else:
                print(f"  ⚠️ Sin datos en pista para {ep}.")

def main():
    parser = argparse.ArgumentParser(description="F1 Data Pipeline - Ingesta Universal")
    parser.add_argument("--year", type=int, default=2026, help="Año de la temporada a extraer")
    parser.add_argument("--races", nargs="+", default=["Australia", "China", "Japan"], 
                        help="Países separados por espacio (ej: --races Australia Monza Mexico)")
    
    args = parser.parse_args()

    extractor = OpenF1DataExtractor(year=args.year)
    
    for race in args.races:
        extractor.extract_race(race)
        
    print("\n🎉 Pipeline de Ingesta Completado Exitosamente.")

if __name__ == "__main__":
    main()