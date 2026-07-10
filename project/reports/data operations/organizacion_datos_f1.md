# F1 Data Architecture and Pipeline Documentation

This document details the organizational structure, the technical transformation flow, and the semantics of the final datasets for the Formula 1 strategic intelligence project.

## 1. Raw Data Organization (Catalog Layer)

The data architecture has been designed to overcome the model of fragmented files per driver, adopting a **Consolidated Entities per Race** schema. This allows a holistic view of the competition and enables traffic and complex interaction analysis.

### Directory Structure (Raw Data)

Each race downloaded from the OpenF1 API is stored in a subfolder named after the event and year:

```text
project/data/raw/
└── [race_name]_[year]/
    ├── laps.csv          # Lap times and positions of all 22 drivers.
    ├── pit.csv           # Pit stop records for the entire grid.
    ├── stints.csv        # Tire compound history per driver.
    ├── car_data.csv      # High-frequency telemetry (RPM, speed, etc.).
    ├── weather.csv       # Global weather conditions.
    └── drivers.csv       # Identifying metadata for the session.
```

## 2. The Transformation Pipeline (`f1_events_pipeline.py`)

The pipeline acts as the bridge between raw sensor data and the intelligence layers (Machine Learning and Graphs).

### Technical Workflow:

1. **Load and Ingestion:** The script automatically detects race folders and reads the unified CSVs.

2. **Cleaning and Normalization:** Headers are standardized to `snake_case`. Times are converted to float seconds, and incomplete records are removed.

3. **Feature Engineering:**

   * **Position Reconstruction:** If position data is inconsistent or null in the source, the cumulative race time is calculated, and an exact mathematical ranking is assigned per lap.

   * **Tire Expansion:** Stint durations are joined with laps to inject the current compound and calculate the tire age (`tyre_age`) continuously.

4. **Interaction Extraction:** The master table is scanned looking for position crosses (Overtakes) and strategy triggers (Pit Entry) to generate the events dataset.

## 3. Output Artifacts: Meaning and Dictionary

The pipeline unifies and compresses the information, generating two optimized Parquet files (Snappy compression) with specific granularities.

### A. Master Parquet (`data/processed/[race]_master.parquet`)

**Granularity:** 1 row = 1 lap of 1 driver. This is the base chronological dataset.

| **Column** | **Source CSV** | **Description** | 
| :--- | :--- | :--- |
| `meeting_key` | `laps.csv` | Unique identifier for the event (Grand Prix). | 
| `session_key` | `laps.csv` | Unique identifier for the session (e.g. Race). | 
| `driver_number` | `laps.csv` | Unique identifier for the driver. | 
| `lap_number` | `laps.csv` | Current lap number. | 
| `date_start` | `laps.csv` | Exact start timestamp of the lap. | 
| `duration_sector_1` | `laps.csv` | Time spent in Sector 1 (in seconds). | 
| `duration_sector_2` | `laps.csv` | Time spent in Sector 2 (in seconds). | 
| `duration_sector_3` | `laps.csv` | Time spent in Sector 3 (in seconds). | 
| `i1_speed` | `laps.csv` | Speed recorded at the first intermediate trap (km/h). | 
| `i2_speed` | `laps.csv` | Speed recorded at the second intermediate trap (km/h). | 
| `st_speed` | `laps.csv` | Maximum speed recorded at the main speed trap (km/h). | 
| `is_pit_out_lap` | `laps.csv` | Boolean flag indicating a pit exit lap. | 
| `lap_duration` | `laps.csv` | Normalized lap time in float seconds. | 
| `segments_sector_1` | `laps.csv` | Array of categorical values representing Sector 1 mini-sectors. | 
| `segments_sector_2` | `laps.csv` | Array of categorical values representing Sector 2 mini-sectors. | 
| `segments_sector_3` | `laps.csv` | Array of categorical values representing Sector 3 mini-sectors. | 
| `position` | `laps.csv` | Track position (cleaned or mathematically recalculated using cumulative time). | 
| `compound` | `stints.csv` | Tire compound used (SOFT, MEDIUM, HARD, UNKNOWN). | 
| `stint_number` | `stints.csv` | Current driver stint number in the race (sequence of stops). | 
| `tyre_age` | `stints.csv` | Calculated variable: Accumulated laps on the current set of tires. | 
| `pit_duration` | `pit.csv` | Total seconds spent in the pit lane during that lap. | 
| `is_pit_lap` | `pit.csv` | Binary flag (1 if pitting on that lap, 0 if not). | 

* **Analytical Meaning:** This is the "State Map" of the race. It provides raw materials for **Clustering** experiments (e.g. grouping tire degradation profiles) and **Ranking/Recommendation** algorithms (e.g. predicting final positions). The new sector metrics (duration and segments) open possibilities for finer pace prediction algorithms.

### B. Events Parquet (`data/events/[race]_events.parquet`)

**Granularity:** 1 row = 1 strategic interaction (continuous time is removed).

*Note on origin:* This file is not downloaded directly from the API, but is generated algorithmically by sequentially scanning the Master dataset. Below is the source CSV file from which the logic to extract each event feature originates:

| **Column** | **Source CSV (Via Master)** | **Description** | 
| :--- | :--- | :--- |
| `race_id` | Directory Metadata | Event identifier (e.g. "australia_2026"). | 
| `lap_number` | `laps.csv` | Exact lap on which the action was triggered. | 
| `event_type` | `laps.csv` + `pit.csv` | Calculated category evaluating physical position changes (`On_Track_Overtake`) or pit entry flags (`Pit_Strategy`). | 
| `initiator_driver` | `laps.csv` | **ORIGIN NODE**: Driver attacking, overtaking, or initiating the pit strategy (derived from `driver_number`). | 
| `target_driver` | `laps.csv` | **DESTINATION NODE**: Driver defending the position (0 if general strategy against the grid). | 
| `initiator_compound` | `stints.csv` | Tire compound of the attacker at the moment of the event, extracted by crossing the lap with stint history. | 
| `initiator_pos_change` | `laps.csv` | Event result calculated by comparing the `position` column between current and surrounding laps (e.g. "P10 -> P7" or "Undercut_Success"). | 

* **Analytical Meaning:** This is the "Network Layer". It defines connections (edges) between drivers (nodes) for **Graph Analysis**, allowing us to model aggression networks and strategic influence on track.

## 4. Final Project Relevance

This data architecture guarantees strict compliance with the technical standards required to pass the course:

* **Non-Trivial Dataset:** By unifying the data of all 22 drivers, high-dimensional databases are generated with sector-level telemetry.

* **Implemented Feature Layer:** Includes highly complex derived numerical variables such as tire degradation (`tyre_age`) and calculated positions.

* **Prepared Graph Layer:** The relational event file enables graph construction for the second half of the semester.