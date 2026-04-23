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
```text
F1-data-project/
├── data/
│   ├── raw/                # Raw CSV files extracted from the API.
│   │   ├── australia_2026/
│   │   ├── china_2026/
│   │   └── japan_2026/
│   └── processed/          # Cleaned and transformed datasets for models.
├── src/
│   ├── Data_extract.py     # Ingestion engineering script (E-L).
│   ├── preprocessing.py    # Cleaning and time-series alignment.
│   └── models/             # PCA, Clustering, and Graph scripts (Coming Soon).
├── notebooks/              # Exploratory Data Analysis (EDA).
└── README.md
