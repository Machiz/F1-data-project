import requests
import pandas as pd
import os
import argparse
import time 

class OpenF1DataExtractor:
    def __init__(self, base_url="https://api.openf1.org/v1"):
        self.base_url = base_url
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

    def extract_session_data(self, meeting_key, session_key):
        """Orquesta la descarga completa usando meeting_key y session_key."""
        print(f"\n{'='*60}\n🏁 Iniciando extracción para: Meeting [{meeting_key}] | Session [{session_key}]\n{'='*60}")
        
        # 1. Validar Meeting y Session, y obtener metadatos para nombrar la carpeta
        meetings = self.fetch_endpoint("meetings", params={"meeting_key": meeting_key})
        if meetings.empty:
            print(f"⚠️ No se encontró información para el meeting_key: {meeting_key}.")
            return

        sessions = self.fetch_endpoint("sessions", params={"session_key": session_key})
        if sessions.empty:
            print(f"⚠️ No se encontró información para el session_key: {session_key}.")
            return

        # RESOLUCIÓN CRÍTICA: Convertimos "latest" en la key numérica real para mantener consistencia
        # durante toda la ejecución, por si una nueva sesión inicia mientras el script corre.
        real_meeting_key = meetings.iloc[0]['meeting_key']
        real_session_key = sessions.iloc[0]['session_key']

        # Extraer metadatos para un nombre de carpeta amigable
        country_name = meetings.iloc[0].get('country_name', f'meeting_{real_meeting_key}')
        year = meetings.iloc[0].get('year', 'UnknownYear')
        session_name = sessions.iloc[0].get('session_name', f'session_{real_session_key}')

        print(f"[OK] Sesión Encontrada: {country_name} - {session_name} ({year}) | Keys reales: M:{real_meeting_key} S:{real_session_key}")

        # Crear directorio estructurado: data/raw/australia_race_2026/
        c_name_clean = str(country_name).lower().replace(' ', '_')
        s_name_clean = str(session_name).lower().replace(' ', '_')
        folder_name = f"{c_name_clean}_{s_name_clean}_{year}"
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
                # Usamos la key numérica real de aquí en adelante
                df = self.fetch_endpoint(ep, params={"session_key": real_session_key})
            
            if not df.empty:
                df.to_csv(os.path.join(output_dir, f"{ep}.csv"), index=False)
                print(f"  ✅ {ep}.csv guardado ({len(df)} filas)")
                
                # Rescatar los números de los pilotos para usarlos después
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
                # ESTRANGULAMIENTO: Pausa obligatoria entre cada piloto
                time.sleep(1.5) 
                
                df_driver = self.fetch_endpoint(ep, params={"session_key": real_session_key, "driver_number": driver})
                
                if not df_driver.empty and not df_driver.isna().all().all():
                    all_drivers_data.append(df_driver)
            
            if all_drivers_data:
                # CONCATENACIÓN CRÍTICA: Unimos a todos los pilotos en un solo DataFrame
                df_final = pd.concat(all_drivers_data, ignore_index=True)
                file_path = os.path.join(output_dir, f"{ep}.csv")
                df_final.to_csv(file_path, index=False)
                print(f"  ✅ Guardado: {ep}.csv -> Datos unificados de todos los pilotos ({len(df_final):,} filas)")
            else:
                print(f"  ⚠️ Sin datos en pista para {ep}.")

def main():
    parser = argparse.ArgumentParser(description="F1 Data Pipeline - Ingesta por Keys")
    # type=str y default="latest"
    parser.add_argument("--meeting-key", type=str, default="latest", help="ID único del evento (ej: 1219) o 'latest'")
    parser.add_argument("--session-key", type=str, default="latest", help="ID único de la sesión (ej: 9158) o 'latest'")
    
    args = parser.parse_args()

    extractor = OpenF1DataExtractor()
    extractor.extract_session_data(args.meeting_key, args.session_key)
        
    print("\n🎉 Pipeline de Ingesta Completado Exitosamente.")

if __name__ == "__main__":
    main()