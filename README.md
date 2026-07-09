# F1 Race Intelligence & Strategic Optimization 🏎️📊

This repository contains the development of the project for the **Big Data** course at **Universidad Peruana de Ciencias Aplicadas (UPC)**. The main objective is to build a strategic decision engine based on high-frequency telemetry and advanced analytics to optimize race performance.

## 🏁 Project Vision
In modern Formula 1, decision-making is driven by data. This project utilizes the **OpenF1 API** to extract, process, and analyze real-time data streams (Speed, RPM, DRS, Intervals) to predict optimal **Pit Windows** and evaluate the success probability of tactical maneuvers such as the *undercut*.

**Role:** Chief Strategy Engineer / Head of Race Intelligence

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Key Libraries:** - `Requests`: Data ingestion from the API.
  - `Pandas`: Dataset structuring and cleaning.
  - `Scikit-learn`: Predictive modeling and dimensionality reduction (PCA).
  - `NetworkX`: Graph analysis for race intervals.
- **Data Source:** [OpenF1 API](https://openf1.org/) (Open Source).

---

## 📂 Repository Structure

All development and pipelines are located in the `project/` directory:

```text
F1-data-project/
├── project/
│   ├── data/                 # Raw data and generated Parquet files
│   ├── src/                  # Production pipeline and training scripts
│   │   ├── data_extraction/  # Ingestion scripts (OpenF1 API)
│   │   ├── features/         # Preprocessing and event extraction
│   │   ├── graphs/           # Graph construction (DRS & Overtakes)
│   │   └── models/           # Layer 1 (Stacking) and Layer 2 (Ranker) training
│   ├── notebooks/            # Jupyter Notebooks (EDA, PCA, Clustering, t-SNE)
│   ├── reports/              # Core project documentation and findings
│   ├── requirements.txt      # Project environment dependencies
│   └── runbook.md            # Reproduction guide
└── README.md
```

## 🚀 Reproduction and Execution

The project features a fully automated and reproducible data-to-model pipeline. To install dependencies, extract telemetry data, process features, train the Stacking and Ranking models, and analyze graphs:

👉 **Please refer to the detailed [project/runbook.md](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/runbook.md) for step-by-step instructions.**

