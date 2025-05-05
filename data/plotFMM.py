import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/FMM.csv')

# --- Increase default font sizes for the figure ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Create figure with two subplots: left for the plot, right for the table
fig, (ax_plot, ax_table) = plt.subplots(
    1, 2, figsize=(14, 6),
    gridspec_kw={'width_ratios': [3, 1.5]}
)

# --- Left: linear‐scale errorbar plot WITHOUT N=100 ---
mask = df['N'] != 100
ax_plot.errorbar(
    df.loc[mask, 'N'],
    df.loc[mask, 'mean_time_s'],
    yerr=df.loc[mask, 'std_time_s'],
    fmt='o-',
    capsize=5
)
ax_plot.set_xlabel('N')
ax_plot.set_ylabel('Time to compute one step (s)')

# --- Right: table for N = 10, 100, 1000 (unchanged) ---
selected = (
    df[df['N'].isin([10, 100, 1000])]
      [['N', 'mean_time_s', 'mean_mem_MB']]
      .rename(columns={
          'mean_time_s': 'Time (s)',
          'mean_mem_MB': 'Memory (MB)'
      })
)
selected_str = selected.copy()
selected_str['N'] = selected_str['N'].astype(int).astype(str)
selected_str['Time (s)'] = selected_str['Time (s)'].map(lambda x: f"{x:.4f}")
selected_str['Memory (MB)'] = selected_str['Memory (MB)'].map(lambda x: f"{x:.4f}")

tbl = ax_table.table(
    cellText=selected_str.values,
    colLabels=selected_str.columns,
    cellLoc='center',
    colLoc='center',
    loc='center'
)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(1)
    cell.set_text_props(ha='center', va='center')
    if row == 0:
        cell.set_facecolor('#e0e0e0')
        cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor('white')
tbl.scale(2, 2.5)
ax_table.axis('off')

plt.tight_layout()
plt.savefig('figures/FMMcomplexity.png')
plt.show()
