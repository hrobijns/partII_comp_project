import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/BH.csv')

# --- Increase default font sizes for the figure ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Filter out N=100 for plotting only
df_plot = df[df['N'] != 100]

# Create figure with two subplots
fig, (ax_plot, ax_table) = plt.subplots(
    1, 2, figsize=(14, 6),
    gridspec_kw={'width_ratios': [3, 1.5]}
)

# --- Left: errorbar plot without N=100 ---
ax_plot.errorbar(
    df_plot['N'], 
    df_plot['mean_time_s'],
    yerr=df_plot['std_time_s'],
    fmt='o-',
    capsize=5
)
ax_plot.set_xlabel('N')
ax_plot.set_ylabel('Time to compute one step (s)')

# --- Right: nicely formatted table for N = 10, 100, 1000 ---
selected = (
    df[df['N'].isin([10, 100, 1000])]
      [['N', 'mean_time_s', 'mean_mem_MB']]
      .rename(columns={
          'mean_time_s': 'Time (s)',
          'mean_mem_MB':   'Memory (MB)'
      })
)

# Convert to strings for display
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

# Style the table
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
plt.savefig('figures/BHcomplexity.png', dpi=300)
plt.show()

