import matplotlib.pyplot as plt

# Data
N = [10, 100, 1000, 10000, 100000]
FMM = [0.0017, 0.0802, 1.5144, 16.3539, 251.7621]
vec = [0.0002, 0.0010, 0.0622, 37.6033]
BH = [0.0037, 0.1071, 1.8496, 21.0681, 302.3337]
naive = [0.00134, 0.10108, 12.58636]

# Create the plot
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(N, FMM, marker = 'o', markersize = 4, label='FMM')
plt.plot(N[:len(vec)], vec, marker = 'o',markersize = 4,label='naïve (vectorised)')
plt.plot(N[:len(BH)], BH, marker = 'o',markersize = 4,label='BH')
plt.plot(N[:len(naive)], naive, marker = 'o',markersize = 4,label='naïve')

# Labels, title, legend, grid
plt.xlabel('Number of bodies (N)')
plt.ylabel('Time to compute one step (s)')
#plt.title('n-body Simulation Timing vs N')
plt.legend()
plt.grid(True)

# Layout and display
plt.tight_layout()
plt.savefig('figures/timingcomparison.png', dpi=300)
plt.show()
