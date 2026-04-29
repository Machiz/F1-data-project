# F1 Data Architecture & Pipeline Documentation

This document provides a comprehensive overview of the data architecture, preprocessing pipeline, and output artifacts for the Formula 1 analytics project. The system is designed to transform raw telemetry and timing data into structured datasets suitable for machine learning and graph analysis.

## 1. Raw Data Organization (Catalog Layer)

The initial architecture, which separated data by individual drivers (e.g., `laps_driver_16.csv`), was refactored into a consolidated, entity-based model. This approach is highly scalable and necessary for analyzing multi-driver interactions (like traffic and overtaking).

### Directory Structure

The raw data is organized by race and entity, containing data for all 20-22 drivers in single files:

```text
project/
└── data/
    └── raw/
        └── [race_name]_[year]/
            ├── laps.csv          # All lap times and positions for the entire grid
            ├── pit.csv           # All pit stop durations and timestamps
            ├── stints.csv        # Tyre compound history for all drivers
            ├── weather.csv       # Global weather metrics (temperature, humidity)
            └── drivers.csv       # Driver metadata
```

### Rationale
* **Scalability:** Processing a single `laps.csv` file with 1,000 rows is significantly faster and less error-prone than iterating through 22 separate files.
* **Global Context:** To calculate relative positions (e.g., who is directly ahead of whom), all drivers must exist within the same tabular structure.

---

## 2. The Preprocessing Pipeline (`f1_events_pipeline.py`)

The pipeline script is the core engine that transforms raw sensor and timing data into meaningful sporting context. 

### A. Cleaning and Normalization
* **Standardization:** Column names are converted to `snake_case` (e.g., `LapTime` becomes `lap_duration`).
* **Type Casting:** String-based lap times (e.g., "1:25.300") are parsed into continuous float seconds (`85.3`) to enable mathematical operations.

### B. Mathematical Position Reconstruction (Feature Engineering)
If the raw `laps.csv` lacks explicit `position` data, the pipeline reconstructs it using a Big Data approach:
1.  **Cumulative Time:** Calculates the total elapsed race time for each driver at any given lap.
2.  **Lap-by-Lap Ranking:** Ranks drivers based on their cumulative time. The driver with the lowest cumulative time at Lap $X$ is assigned `position = 1`.

### C. Tyre Integration (Stint Expansion)
Raw stint data defines ranges (e.g., "Stint 1: Laps 1 to 15"). The pipeline **expands** this range so that every individual lap row knows:
* The exact tyre `compound` being used (Soft, Medium, Hard).
* The current `tyre_age` (how many laps that specific set of tyres has completed).

---

## 3. Parquet Output Artifacts and Their Significance

The pipeline outputs data in `.parquet` format. Parquet provides columnar storage and Snappy compression, drastically reducing file size and load times compared to CSVs. The system generates two distinct types of datasets, fulfilling different analytical requirements.

### A. The Master Parquet (`data/processed/[race]_master.parquet`)

* **Granularity:** 1 Row = 1 Lap for 1 Driver.
* **Description:** This is the "Single Source of Truth." It contains the complete state of the car and driver for every lap of the race.
* **Analytical Use Case:** This dense, chronological dataset is the mandatory foundation for **Clustering** models (e.g., grouping drivers by tyre degradation profiles) and **Ranking** algorithms (e.g., predicting final race positions based on early-race pace).

### B. The Events Parquet (`data/events/[race]_events.parquet`)

* **Granularity:** 1 Row = 1 Strategic Event or Interaction.
* **Description:** This dataset removes continuous time. A row only exists if a specific action occurred between entities.
    * **`On_Track_Overtake`:** Records physical passes on the track, detailing the `initiator` (attacker), the `target` (defender), the lap, and the tyre compounds involved.
    * **`Pit_Strategy`:** Evaluates Undercut attempts. It records a driver entering the pits and evaluates if the strategy was successful (gained positions) 3 laps later.
* **Analytical Use Case:** This relational dataset (Node $\rightarrow$ Edge $\rightarrow$ Node) is specifically engineered for the **Graph Layer** of the project. It allows for the mapping of interaction networks, visualizing driver aggressiveness and strategic battles.