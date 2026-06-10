import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import os
import requests

# Base paths
BASE_DIR = Path('../../data')
FEAT_DIR = BASE_DIR / 'features'
REPORT_DIR = Path('../../reports/clustering models')
FIGURES_DIR = REPORT_DIR / 'figures'

print("Starting experiment...")

# 1. Load telemetry features
df = pd.read_parquet(FEAT_DIR / 'telemetry_features_v4.parquet')
print(f"Original shape: {df.shape}")

# Try to get team_name and name_acronym as in PCA_v4
session_keys = df['session_key'].dropna().unique().tolist()
drivers_list = []
for sk in session_keys:
    url = f'https://api.openf1.org/v1/drivers?session_key={sk}'
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            drivers_list.extend(r.json())
    except Exception as e:
        print(f"Error fetching drivers for session {sk}: {e}")

if drivers_list:
    df_drivers = pd.DataFrame(drivers_list)
    df_drivers = df_drivers[['session_key', 'driver_number', 'team_name', 'name_acronym']].drop_duplicates(subset=['session_key', 'driver_number'])
    df_drivers['session_key'] = df_drivers['session_key'].astype(float)
    df_drivers['driver_number'] = df_drivers['driver_number'].astype(float)
    df = df.merge(df_drivers, on=['session_key', 'driver_number'], how='left')
    print("Successfully joined drivers metadata from OpenF1 API.")
else:
    print("Could not fetch drivers from OpenF1, proceeding with existing columns.")

# 2. Filter pit laps (like in PCA_v4)
df_clean = df[
    (df['is_pit_lap'] != 1) &
    (df['is_pit_out_lap'] != 1)
].reset_index(drop=True)
print(f"Shape after filtering pit laps: {df_clean.shape}")

# 3. Outlier filtering GLOBALLY on lap_duration (ignoring race_name!)
global_median = df_clean['lap_duration'].median()
global_std = df_clean['lap_duration'].std()
df_clean = df_clean[
    df_clean['lap_duration'] <= global_median + 2 * global_std
].reset_index(drop=True)
print(f"Shape after global outlier filtering: {df_clean.shape}")

# Define identifiers and exclude columns
ID_COLS = ['race_name', 'driver_number', 'lap_number', 'team_name']
EXCLUDE = ['is_pit_out_lap', 'is_pit_lap', 'stint_number']

feature_cols = [c for c in df_clean.columns if c not in (ID_COLS + EXCLUDE)]

X_raw = df_clean[feature_cols]

# Null diagnostic
null_pct = X_raw.isnull().mean().sort_values(ascending=False)
cols_ok = null_pct[null_pct <= 0.4].index.tolist()
X_numeric = X_raw[cols_ok].select_dtypes(include='number')
print(f"Numeric features for PCA: {X_numeric.shape[1]}")
print(f"Features: {X_numeric.columns.tolist()}")

# Imputation and scaling
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X_numeric)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
print(f"Scaled matrix shape: {X_scaled.shape}")

# 4. Perform PCA (6 components)
pca = PCA(n_components=6, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_ind = pca.explained_variance_ratio_
var_cum = np.cumsum(var_ind)

print("\nPCA Results:")
for i, (ind, cum) in enumerate(zip(var_ind, var_cum)):
    print(f"  PC{i+1}: individual {ind:.1%} | cumulative {cum:.1%}")

# Save PCA scores for reference
df_pca = df_clean[ID_COLS + EXCLUDE].copy()
for i in range(6):
    df_pca[f'PC{i+1}'] = X_pca[:, i]

# 5. Parameter sweep for DBSCAN on the new PCA space
print("\nSweeping DBSCAN parameters...")
best_eps = None
best_min_samples = None
best_silhouette = -1
best_noise_pct = 100
best_n_clusters = 0

results = []
eps_range = np.linspace(0.8, 2.0, 13)
min_samples_range = range(5, 26, 5)

for eps in eps_range:
    for ms in min_samples_range:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_pca)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).sum() / len(labels) * 100
        
        if n_clusters > 1:
            # Silhouette on signal
            mask = labels != -1
            if mask.sum() > 10:
                sil = silhouette_score(X_pca[mask], labels[mask])
                db_score = davies_bouldin_score(X_pca[mask], labels[mask])
            else:
                sil = -1
                db_score = 99
        else:
            sil = -1
            db_score = 99
            
        results.append({
            'eps': eps,
            'min_samples': ms,
            'n_clusters': n_clusters,
            'noise_pct': noise_pct,
            'silhouette': sil,
            'davies_bouldin': db_score
        })
        
        # Criteria: noise between 5% and 15%, n_clusters between 3 and 6, maximize silhouette
        if 5.0 <= noise_pct <= 15.0 and 3 <= n_clusters <= 6:
            if sil > best_silhouette:
                best_silhouette = sil
                best_eps = eps
                best_min_samples = ms
                best_noise_pct = noise_pct
                best_n_clusters = n_clusters

print(f"Optimal parameters found: eps={best_eps:.2f}, min_samples={best_min_samples}, "
      f"clusters={best_n_clusters}, noise={best_noise_pct:.1f}%, silhouette={best_silhouette:.4f}")

# Fallback in case search is empty
if best_eps is None:
    print("Could not find configuration matching criteria. Using fallback: eps=1.2, min_samples=15")
    best_eps = 1.2
    best_min_samples = 15

# Run optimal DBSCAN
db_opt = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels_opt = db_opt.fit_predict(X_pca)

df_pca['cluster'] = labels_opt

n_clusters_opt = len(set(labels_opt)) - (1 if -1 in labels_opt else 0)
noise_opt = (labels_opt == -1).sum()
noise_pct_opt = noise_opt / len(labels_opt) * 100

print(f"\nFinal DBSCAN configuration (eps={best_eps:.2f}, min_samples={best_min_samples}):")
print(f"  Clusters found: {n_clusters_opt}")
print(f"  Noise points: {noise_opt} ({noise_pct_opt:.1f}%)")

# Calculate metrics on signal
mask_signal = labels_opt != -1
X_signal = X_pca[mask_signal]
labels_signal = labels_opt[mask_signal]

if len(set(labels_signal)) > 1:
    final_sil = silhouette_score(X_signal, labels_signal)
    final_db = davies_bouldin_score(X_signal, labels_signal)
    final_ch = calinski_harabasz_score(X_signal, labels_signal)
    print(f"  Silhouette (signal): {final_sil:.4f}")
    print(f"  Davies-Bouldin: {final_db:.4f}")
    print(f"  Calinski-Harabasz: {final_ch:.4f}")
else:
    print("  Could not calculate silhouette: less than 2 clusters in signal.")

# 6. Physical archetypes analysis
print("\nCluster profile analysis:")
df_clean['cluster'] = labels_opt

profile_cols = ['lap_duration', 'st_speed', 'throttle_pct_full', 'tyre_age']
profiles = df_clean.groupby('cluster')[profile_cols].agg(['count', 'mean', 'std'])
print(profiles.to_string())

# Save results for comparison plot
df_pca.to_parquet('experiment_pca_scores.parquet')
print("Saved experiment PCA scores to experiment_pca_scores.parquet")

# Plot PCA Scatter
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=df_pca['PC1'],
    y=df_pca['PC2'],
    hue=df_pca['cluster'].astype(str),
    palette='Set1',
    alpha=0.6,
    edgecolor=None
)
plt.title(f"DBSCAN Clusters on New PCA Space (No Race Name Grouping)\neps={best_eps:.2f}, min_samples={best_min_samples}")
plt.savefig('experiment_scatter_pca.png', dpi=150)
print("Saved scatter plot to experiment_scatter_pca.png")

# Compare shapes and lap counts
original_laps = 3004  # Laps in telemetry_pca_v4.parquet
current_laps = len(df_clean)
print(f"\nLap counts comparison:")
print(f"  Original laps (grouped outlier filter): {original_laps}")
print(f"  New laps (global outlier filter): {current_laps}")
print(f"  Difference: {current_laps - original_laps} laps ({(current_laps - original_laps)/original_laps*100:+.1f}%)")

print("\nFinished successfully.")
