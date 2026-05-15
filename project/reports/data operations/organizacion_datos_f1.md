# Documentación de Arquitectura y Pipeline de Datos F1

Este documento detalla la estructura organizacional, el flujo de transformación técnica y la semántica de los datos finales para el proyecto de inteligencia estratégica en Fórmula 1.

## 1. Organización de Datos Crudos (Capa de Catálogo)

La arquitectura ha sido diseñada para superar el modelo de archivos fragmentados por piloto, adoptando un esquema de **Entidades Consolidadas por Carrera**. Esto permite una visión holística de la competencia y habilita análisis de tráfico e interacciones complejas.

### Estructura de Directorios (Raw Data)

Cada carrera descargada desde la API OpenF1 se almacena en una subcarpeta bajo el nombre del evento y el año:

```text
project/data/raw/
└── [nombre_carrera]_[año]/
    ├── laps.csv          # Tiempos de vuelta y posiciones de los 22 pilotos.
    ├── pit.csv           # Registros de paradas en boxes para toda la parrilla.
    ├── stints.csv        # Historial de compuestos de neumáticos por piloto.
    ├── car_data.csv      # Telemetría de alta frecuencia (RPM, velocidad, etc.).
    ├── weather.csv       # Condiciones climáticas globales.
    └── drivers.csv       # Metadatos identificativos de la sesión.
```

## 2. El Pipeline de Transformación (`f1_events_pipeline.py`)

El pipeline actúa como el puente entre los datos crudos de sensores y las capas de inteligencia (Machine Learning y Grafos).

### Flujo de Trabajo Técnico:

1. **Carga e Ingesta:** El script detecta automáticamente las carpetas de carrera y lee los CSVs unificados.

2. **Limpieza y Normalización:** Se estandarizan los nombres a `snake_case`. Los tiempos se convierten a segundos flotantes y se eliminan registros incompletos.

3. **Ingeniería de Características (Feature Engineering):**

   * **Reconstrucción de Posición:** Si los datos de posición son inconsistentes o nulos en el origen, se calcula el tiempo acumulado de carrera y se asigna un ranking matemático exacto por vuelta.

   * **Expansión de Neumáticos:** Se cruza la duración de los stints con las vueltas para inyectar el compuesto actual y calcular la edad del neumático (`tyre_age`) de forma continua.

4. **Extracción de Interacciones:** Se escanea la tabla maestra buscando cruces de posición (Adelantamientos) y detonadores de estrategia (Entradas a Pits) para generar el dataset de eventos.

## 3. Artefactos de Salida: Significado y Diccionario

El pipeline unifica y comprime la información, generando dos archivos Parquet optimizados (Snappy) con granularidades específicas.

### A. Master Parquet (`data/processed/[carrera]_master.parquet`)

**Granularidad:** 1 fila = 1 vuelta de 1 piloto. Es el dataset cronológico base.

| **Columna** | **Origen CSV** | **Descripción** | 
| :--- | :--- | :--- |
| `meeting_key` | `laps.csv` | Identificador único del evento (Gran Premio). | 
| `session_key` | `laps.csv` | Identificador único de la sesión (ej. Carrera). | 
| `driver_number` | `laps.csv` | Identificador único del piloto. | 
| `lap_number` | `laps.csv` | Número de la vuelta actual. | 
| `date_start` | `laps.csv` | Marca de tiempo (timestamp) exacta de inicio de la vuelta. | 
| `duration_sector_1` | `laps.csv` | Tiempo empleado en recorrer el sector 1 (en segundos). | 
| `duration_sector_2` | `laps.csv` | Tiempo empleado en recorrer el sector 2 (en segundos). | 
| `duration_sector_3` | `laps.csv` | Tiempo empleado en recorrer el sector 3 (en segundos). | 
| `i1_speed` | `laps.csv` | Velocidad registrada en la primera trampa intermedia (km/h). | 
| `i2_speed` | `laps.csv` | Velocidad registrada en la segunda trampa intermedia (km/h). | 
| `st_speed` | `laps.csv` | Velocidad máxima registrada en la trampa de velocidad principal (speed trap) (km/h). | 
| `is_pit_out_lap` | `laps.csv` | Flag booleano que indica si es una vuelta de salida desde boxes. | 
| `lap_duration` | `laps.csv` | Tiempo de vuelta normalizado en segundos flotantes. | 
| `segments_sector_1` | `laps.csv` | Array de valores categóricos representando los mini-sectores del sector 1. | 
| `segments_sector_2` | `laps.csv` | Array de valores categóricos representando los mini-sectores del sector 2. | 
| `segments_sector_3` | `laps.csv` | Array de valores categóricos representando los mini-sectores del sector 3. | 
| `position` | `laps.csv` | Posición en pista (limpiada o recalculada matemáticamente por tiempo acumulado). | 
| `compound` | `stints.csv` | Compuesto de neumático usado (SOFT, MEDIUM, HARD, UNKNOWN). | 
| `stint_number` | `stints.csv` | Número de stint actual del piloto en la carrera (secuencia de paradas). | 
| `tyre_age` | `stints.csv` | Variable calculada: Vueltas acumuladas con el set de neumáticos actual. | 
| `pit_duration` | `pit.csv` | Segundos totales gastados en la calle de boxes durante esa vuelta. | 
| `is_pit_lap` | `pit.csv` | Flag binario (1 si paró en boxes en esa vuelta, 0 si no). | 

* **Significado Analítico:** Es el "Mapa de Estado" de la carrera. Proporciona la materia prima para experimentos de **Clustering** (ej. agrupar perfiles de degradación de neumáticos) y algoritmos de **Ranking/Recomendación** (ej. predecir posiciones finales). Las nuevas métricas de sectores (duration y segments) abren posibilidades a algoritmos de predicción de ritmo más finos.

### B. Events Parquet (`data/events/[carrera]_events.parquet`)

**Granularidad:** 1 fila = 1 interacción estratégica (El tiempo continuo desaparece).

*Nota sobre el origen:* Este archivo no se descarga directamente de la API, sino que se genera algorítmicamente escaneando secuencialmente el dataset Master. A continuación, se detalla el archivo CSV original del cual proviene la lógica para extraer cada característica del evento:

| **Columna** | **Origen CSV (Vía Master)** | **Descripción** | 
| :--- | :--- | :--- |
| `race_id` | Metadata del Directorio | Identificador del evento (Ej: "australia_2026"). | 
| `lap_number` | `laps.csv` | Vuelta exacta en la que se detonó la acción. | 
| `event_type` | `laps.csv` + `pit.csv` | Categoría calculada evaluando cambios de posición físicos (`On_Track_Overtake`) o flags de entradas a boxes (`Pit_Strategy`). | 
| `initiator_driver` | `laps.csv` | **NODO ORIGEN**: Piloto que ataca, adelanta o inicia la estrategia de pit (derivado del `driver_number`). | 
| `target_driver` | `laps.csv` | **NODO DESTINO**: Piloto que defiende la posición (0 si es estrategia general contra la parrilla). | 
| `initiator_compound` | `stints.csv` | Compuesto de neumático del atacante al momento del evento, extraído cruzando la vuelta con el historial de stints. | 
| `initiator_pos_change` | `laps.csv` | Resultado del evento calculado comparando la columna `position` entre la vuelta actual y vueltas previas/futuras (Ej: "P10 -> P7" o "Undercut_Success"). | 

* **Significado Analítico:** Es la "Capa de Red". Define las conexiones (aristas) entre pilotos (nodos) para el **Análisis de Grafos**, permitiendo modelar redes de agresión e influencia estratégica en la pista.

## 4. Relevancia para el Proyecto Final

Esta arquitectura de datos garantiza el cumplimiento estricto de los estándares técnicos exigidos para aprobar el curso:

* **Dataset No Trivial:** Al unificar los datos de los 22 pilotos, se generan bases de datos de alta dimensionalidad con telemetría a nivel sectorial.

* **Feature Layer Implementada:** Incluye variables numéricas derivadas de alta complejidad como la degradación (`tyre_age`) y posiciones calculadas.

* **Graph Layer Preparada:** El archivo relacional de eventos habilita la construcción de grafos para la segunda mitad del semestre.