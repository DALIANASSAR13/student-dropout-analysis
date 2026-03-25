import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import shutil
import sys

CONTAINER_PLOTS = 'plots'
OUTPUT_DIR      = '/mnt/user-data/outputs'
OUTPUT_PLOTS    = os.path.join(OUTPUT_DIR, 'plots')

def save_plot(filename):
    """Save current figure to container AND laptop output folder."""
    container_path = os.path.join(CONTAINER_PLOTS, filename)
    output_path    = os.path.join(OUTPUT_PLOTS,    filename)
    plt.savefig(container_path, dpi=150, bbox_inches='tight')
    shutil.copy(container_path, output_path)
    print(f"   📊 Saved: plots/{filename}  →  outputs/plots/{filename}")
    plt.close()

def save_csv(df, filename):
    """Save dataframe to container AND laptop output folder."""
    container_path = filename
    output_path    = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(container_path, index=False)
    shutil.copy(container_path, output_path)
    print(f"   💾 Saved: {filename}  →  outputs/{filename}")

def cluster(input_path='data_preprocessed.csv'):
    df = pd.read_csv(input_path)
    print(f"✅ cluster.py: Received dataset — {len(df)} rows.")

    feature_cols = ['Admission grade', 'Age at enrollment', 'Debtor', 'Scholarship holder']
    X = df[feature_cols].values

    os.makedirs(CONTAINER_PLOTS, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS,    exist_ok=True)

    #  Elbow + Silhouette 
    inertias, sil_scores = [], []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(k_range), inertias,   marker='o', color='#2c3e50', linewidth=2)
    axes[0].set_title('Elbow Method (Inertia)',  fontsize=12, fontweight='bold')
    axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].plot(list(k_range), sil_scores, marker='s', color='#16a085', linewidth=2)
    axes[1].set_title('Silhouette Score',        fontsize=12, fontweight='bold')
    axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.suptitle('Choosing Optimal k for K-Means', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_plot('plot6_elbow_silhouette.png')

    # Final clustering 
    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"   🏆 Best k = {best_k}  (silhouette = {max(sil_scores):.4f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['Cluster'] = km_final.fit_predict(X)

    #  PCA scatter 
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_ * 100
    cluster_colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22']

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(best_k):
        mask = df['Cluster'] == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=f'Cluster {c}', color=cluster_colors[c],
                   alpha=0.45, s=20, edgecolors='none')
    centroids_2d = pca.transform(km_final.cluster_centers_)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               marker='X', s=180, color='black', zorder=5, label='Centroids')
    ax.set_title(f'K-Means (k={best_k}) — PCA Projection', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({var[0]:.1f}% var)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% var)')
    ax.legend(loc='upper right', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_plot('plot7_kmeans_pca.png')

    # Cluster profile bars 
    profile = df.groupby('Cluster')[feature_cols].mean().round(3)
    print("\n   📋 Cluster Profiles:\n", profile.to_string())

    fig, axes = plt.subplots(1, len(feature_cols), figsize=(13, 4))
    for i, feat in enumerate(feature_cols):
        vals = [profile.loc[c, feat] for c in range(best_k)]
        axes[i].bar(range(best_k), vals,
                    color=[cluster_colors[c] for c in range(best_k)], edgecolor='white')
        axes[i].set_title(feat, fontsize=9, fontweight='bold')
        axes[i].set_xticks(range(best_k))
        axes[i].set_xticklabels([f'C{c}' for c in range(best_k)], fontsize=8)
        axes[i].spines[['top', 'right']].set_visible(False)
    plt.suptitle('Mean Feature Value per Cluster', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot('plot8_cluster_profiles.png')

    #  Save clustered CSV 
    save_csv(df, 'data_clustered.csv')

    print(f"\n✅ cluster.py: All results saved to container AND laptop outputs/.")
    print("🏁 Pipeline complete!\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'data_preprocessed.csv'
    cluster(path)
