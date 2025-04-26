import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('data/BH.csv')

# Create the error‐bar plot
plt.figure(figsize=(8, 6))
plt.errorbar(
    df['N'],
    df['mean_time_s'],
    yerr=df['std_time_s'],
    fmt='o-',
    capsize=5,
    elinewidth=1,
    markeredgewidth=1
)

# Labels and title
plt.xlabel('Number of bodies N')
plt.ylabel('Mean time for a steps (s)')
plt.title('Benchmark: Time vs N (with ±1σ error bars)')

# Grid & layout
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()