# Documentación de Organización y Procesamiento de Datos F1

Este documento detalla la arquitectura, el flujo de trabajo y las transformaciones aplicadas al proyecto de datos de Fórmula 1 para cumplir con los requisitos del curso [cite: semester_group_assignment_brief (1).md].

## 1. Organización del Proyecto
El proyecto sigue una estructura de directorios diseñada para la reproducibilidad y el escalamiento de datos [cite: machiz/f1-data-project/F1-data-project-883597425611c129f47d8fe9c3af9220b33876e9/README.md]:

* `project/src/`: Contiene los scripts de extracción y pipelines [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].
* `project/data/raw/`: Almacena los archivos CSV originales descargados de la API OpenF1 [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/data/raw/australia_2026/laps.csv].
* `project/data/processed/`: Contiene las tablas maestras integradas en formato Parquet [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].
* `project/data/events/`: Contiene los datasets de interacciones (adelantamientos y estrategias) para análisis de grafos [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].

## 2. Proceso de Extracción
La extracción evolucionó de un enfoque limitado a uno robusto y escalable:

* **Enfoque Inicial**: Se extrajeron datos específicos para los pilotos 16 y 44 en tres carreras (Australia, China, Japón) [cite: machiz/f1-data-project/F1-data-project-883597425611c129f47d8fe9c3af9220b33876e9/project/src/Data_extract_AUS.py].
* **Enfoque Final (`extract_f1_data.py`)**: Se implementó una extracción universal para los 22 pilotos de la parrilla [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/extract_f1_data.py].
* **Manejo de Errores**: Se incorporó *Exponential Backoff* y pausas de cortesía para mitigar los errores "429 Too Many Requests" de la API [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/extract_f1_data.py].

## 3. Pipeline de Procesamiento y Eventos (`f1_events_pipeline.py`)
Este script unifica el preprocesamiento y la generación de capas de interacción [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py]:

### Fase 1: Preprocesamiento Global (Granularidad: Vuelta-Piloto)
* **Limpieza de Tiempos**: Conversión de formatos de texto a segundos flotantes [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].
* **Reconstrucción de Posiciones**: Si la columna `position` falta, se calcula matemáticamente mediante el tiempo acumulado de los pilotos por vuelta [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].
* **Integración de Neumáticos**: Se expande la información de `stints.csv` para calcular la edad del neumático (`tyre_age`) en cada vuelta [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].

### Fase 2: Extracción de Eventos (Granularidad: Evento/Interacción)
* **Adelantamientos (Overtakes)**: Detección de cambios de posición en pista entre pilotos que no están en fase de pits [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].
* **Estrategia de Pits (Undercuts)**: Evaluación del éxito o fallo de una parada en boxes comparando la posición antes y 3 vueltas después del evento [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].

## 4. Cumplimiento de Estándares Técnicos
La organización de los datos permite realizar los experimentos obligatorios [cite: semester_group_assignment_brief (1).md]:
* **Clustering**: Posible sobre el `master.parquet` usando métricas de degradación y tiempos [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].
* **Ranking/Recomendación**: Modelado de posiciones finales a partir de la evolución de carrera [cite: semester_group_assignment_brief (1).md].
* **Análisis de Grafos**: Soportado por los datasets en la carpeta `events/` que actúan como aristas entre nodos (pilotos) [cite: machiz/f1-data-project/F1-data-project-ddbdb447f0582866e49972a73411cdcce44610d2/project/src/f1_events_pipeline.py].
