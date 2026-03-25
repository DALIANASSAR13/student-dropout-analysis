import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
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

def visualize(input_path='data_preprocessed.csv'):
    df = pd.read_csv(input_path)
    print(f"✅ visualize.py: Loaded dataset — {len(df)} rows, {df.shape[1]} cols.")

    os.makedirs(CONTAINER_PLOTS, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS,    exist_ok=True)

    target_labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    colors        = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}
    df['Target Label'] = df['Target'].map(target_labels)

    # Target distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df['Target'].value_counts().sort_index()
    bars = ax.bar([target_labels[i] for i in counts.index],
                  counts.values,
                  color=[colors[i] for i in counts.index],
                  edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Student Outcome Distribution', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xlabel('Outcome')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_plot('plot1_target_distribution.png')

    # Admission grade histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    for t, label in target_labels.items():
        ax.hist(df[df['Target'] == t]['Admission grade'],
                bins=30, alpha=0.6, label=label, color=colors[t], edgecolor='none')
    ax.set_title('Admission Grade Distribution by Outcome', fontsize=13, fontweight='bold')
    ax.set_xlabel('Admission Grade (scaled)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_plot('plot2_admission_grade_dist.png')

    #  Age boxplot
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [df[df['Target'] == t]['Age at enrollment'].values for t in [0, 1, 2]]
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, t in zip(bp['boxes'], [0, 1, 2]):
        patch.set_facecolor(colors[t])
        patch.set_alpha(0.7)
    ax.set_xticklabels([target_labels[t] for t in [0, 1, 2]])
    ax.set_title('Age at Enrollment by Outcome', fontsize=13, fontweight='bold')
    ax.set_ylabel('Age (scaled)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_plot('plot3_age_boxplot.png')

    #Debtor & Scholarship grouped bar
    fig, ax = plt.subplots(figsize=(8, 4))
    xpos  = range(3)
    width = 0.35
    debtor_rates  = [df[df['Target'] == t]['Debtor'].mean() * 100           for t in [0,1,2]]
    scholar_rates = [df[df['Target'] == t]['Scholarship holder'].mean() * 100 for t in [0,1,2]]
    ax.bar([p - width/2 for p in xpos], debtor_rates,  width, label='Debtor %',      color='#e74c3c', alpha=0.8)
    ax.bar([p + width/2 for p in xpos], scholar_rates, width, label='Scholarship %', color='#3498db', alpha=0.8)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([target_labels[t] for t in [0, 1, 2]])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Debtor & Scholarship Rates by Outcome', fontsize=13, fontweight='bold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_plot('plot4_debtor_scholarship.png')

    #  Scatter grade vs age
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, label in target_labels.items():
        sub = df[df['Target'] == t]
        ax.scatter(sub['Admission grade'], sub['Age at enrollment'],
                   label=label, color=colors[t], alpha=0.35, s=18, edgecolors='none')
    ax.set_title('Admission Grade vs Age at Enrollment', fontsize=13, fontweight='bold')
    ax.set_xlabel('Admission Grade (scaled)')
    ax.set_ylabel('Age at Enrollment (scaled)')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_plot('plot5_scatter_grade_age.png')

    print(f"\n✅ visualize.py: 5 plots → container ({CONTAINER_PLOTS}/) AND laptop ({OUTPUT_PLOTS}/)")
    print("🔁 Handing off to cluster.py ...\n")

    subprocess.run([sys.executable, 'cluster.py', input_path], check=True)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'data_preprocessed.csv'
    visualize(path)
