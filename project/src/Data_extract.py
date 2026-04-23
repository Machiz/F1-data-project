import requests
import pandas as pd
import os

class OpenF1Extractor:
    def __init__(self, base_url="https://api.openf1.org/v1"):
        self.base_url = base_url
        self.output_dir = "data/raw"
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_data(self, endpoint, params=None):
        """Extrae datos de un endpoint específico y los devuelve como DataFrame."""
        url = f"{self.base_url}/{endpoint}"
        try:
            print(f"Extrayendo datos de: {endpoint}...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"Advertencia: No se encontraron datos para {endpoint}.")
                return None
            
            df = pd.DataFrame(data)
            # Guardar automáticamente como CSV para el entregable
            file_path = os.path.join(self.output_dir, f"{endpoint}.csv")
            df.to_csv(file_path, index=False)
            print(f"Éxito: {endpoint} guardado en {file_path} ({len(df)} filas).")
            return df
        except Exception as e:
            print(f"Error al extraer {endpoint}: {e}")
            return None

def main():
    extractor = OpenF1Extractor()

    # Ejemplo: Filtrar por una sesión específica para no saturar la memoria
    # Puedes obtener el session_key de la tabla 'sessions'
    # 9159 es el session_key para el GP de Bélgica 2023 (ejemplo)
    target_params = {"session_key": 9159}

    endpoints = [
        "car_data",
        "drivers",
        "intervals",
        "laps",
        "location",
        "overtakes",
        "pit",
        "meetings",
        "race_control",
        "sessions",
        "session_results",
        "starting_grid",
        "stints",
        "weather"
    ]

    for endpoint in endpoints:
        # Algunos endpoints como 'sessions' o 'meetings' funcionan mejor sin filtros específicos de sesión
        if endpoint in ["sessions", "meetings"]:
            extractor.fetch_data(endpoint, params={"year": 2023})
        else:
            extractor.fetch_data(endpoint, params=target_params)

if __name__ == "__main__":
    main()