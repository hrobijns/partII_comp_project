
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Paths to your CSV files
    df_naive = pd.read_csv('data/naive.csv')
    df_bh    = pd.read_csv('data/BH.csv')
    df_fmm   = pd.read_csv('data/FMM.csv')
    return df_naive, df_bh, df_fmm

def plot_time_complexity(df_naive, df_bh, df_fmm):
    # Extract N and time columns
    Nn = df_naive['N']
    t_naive       = df_naive['naive_time_mean']
    t_vectorised  = df_naive['vector_time_mean']
    Nb = df_bh['N']
    t_bh          = df_bh['mean_time_s']
    Nf = df_fmm['N']
    t_fmm         = df_fmm['mean_time_s']

    plt.figure(figsize=(8, 6))
    plt.plot(Nn, t_naive,      marker='o', linestyle='-', label='Naive')
    plt.plot(Nn, t_vectorised, marker='s', linestyle='--', label='Naive (vectorised)')
    plt.plot(Nb, t_bh,         marker='^', linestyle='-.', label='Barnes–Hut')
    plt.plot(Nf, t_fmm,        marker='v', linestyle=':',  label='Fast Multipole')

    plt.xlabel('Number of Particles (N)')
    plt.ylabel('Mean Time (s)')
    plt.title('Time Complexity vs. N for Different N-Body Algorithms')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def main():
    df_naive, df_bh, df_fmm = load_data()
    plot_time_complexity(df_naive, df_bh, df_fmm)

if __name__ == '__main__':
    main()
