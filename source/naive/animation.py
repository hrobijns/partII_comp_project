import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from simulation import SimulationVectorised

class Animation:
    def __init__(self, sim, steps=200, interval=50, xlim=(-3,3), ylim=(-3,3)):
        self.sim = sim
        self.steps = steps
        self.interval = interval  # in milliseconds

        # red for +1, blue for -1
        self.colors = np.where(self.sim.charge > 0, 'red', 'blue')

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_facecolor('white')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.scat = self.ax.scatter(self.sim.pos[:,0],
                                    self.sim.pos[:,1],
                                    s=5,
                                    c=self.colors)

        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.steps,
            interval=self.interval,
            blit=True
        )

    def update(self, frame):
        self.sim.step()
        self.scat.set_offsets(self.sim.pos)
        return (self.scat,)

    def save(self, filename, fps=30):
        # Use FFMpegWriter to save as MP4
        writer = FFMpegWriter(fps=fps)
        # dpi can be tuned; default is fine
        self.ani.save(filename, writer=writer)
        print(f"Saved animation to {filename}")

    def show(self):
        plt.show()


def main():
    np.random.seed(42)
    N = 1000
    pos    = np.random.uniform(-1, 1, (N, 2))
    vel    = np.random.uniform(-0.5, 0.5, (N, 2))
    charge = np.random.choice([-1, +1], size=N)
    sim = SimulationVectorised(pos, vel, charge)

    anim = Animation(sim, steps=500, interval=20)
    # Save to MP4 before (or instead of) showing:
    anim.save("figures/simulation.mp4", fps=30)
    # If you still want to pop up the window afterward:
    anim.show()

if __name__ == '__main__':
    main()