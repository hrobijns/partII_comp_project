import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List
import numpy as np
from .core import Simulation, Body
from .utils import total_energy, total_momentum

class MotionAnimation:
    """Animate the motion of bodies only."""
    def __init__(
        self,
        bodies: List[Body],
        sim: Simulation,
        steps: int = 200,
        interval: int = 50,
    ):
        self.sim = sim
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_facecolor('black')
        self.scat = [
            self.ax.plot([], [], 'wo', markersize=b.mass*2)[0]
            for b in bodies
        ]
        self.ani = FuncAnimation(
            self.fig,
            self._update,
            frames=steps,
            interval=interval,
            blit=True
        )

    def _update(self, i):
        self.sim.step()
        for sc, b in zip(self.scat, self.sim.bodies):
            sc.set_data(b.position[0], b.position[1])
        return self.scat

    def show(self):
        plt.show()

class EnergyMomentumAnimation:
    """Animate motion plus track % changes in energy & momentum."""
    def __init__(
        self,
        bodies: List[Body],
        sim: Simulation,
        steps: int = 500,
        interval: int = 50,
    ):
        self.sim = sim
        self.steps = steps
        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, figsize=(12,5))
        # left: motion
        self.ax1.set_facecolor('black')
        self.scat = [
            self.ax1.plot([],[],'wo',markersize=b.mass*2)[0]
            for b in bodies
        ]
        # right: metrics
        self.energy_data, self.mom_data = [], []
        self.energy_line, = self.ax2.plot([],[],label='Energy Δ%')
        self.mom_line, = self.ax2.plot([],[],label='Momentum Δ%')
        self.ax2.legend(); self.ax2.set_xlabel('Step');
        self.ax2.set_ylabel('% Change')
        self.init_energy = None
        self.init_mom = None
        self.ani = FuncAnimation(
            self.fig,
            self._update,
            frames=steps,
            interval=interval,
            blit=True
        )

    def _update(self, i):
        self.sim.step()
        # update motion
        for sc, b in zip(self.scat, self.sim.bodies):
            sc.set_data(b.position[0], b.position[1])
        # compute metrics
        E = total_energy(self.sim.bodies)
        P = np.linalg.norm(total_momentum(self.sim.bodies))
        if i==0:
            self.init_energy, self.init_mom = E, P
        self.energy_data.append(100*(E-self.init_energy)/abs(self.init_energy))
        self.mom_data.append(100*(P-self.init_mom)/abs(self.init_mom))
        self.energy_line.set_data(range(len(self.energy_data)), self.energy_data)
        self.mom_line.set_data(range(len(self.mom_data)), self.mom_data)
        return self.scat + [self.energy_line, self.mom_line]

    def show(self):
        plt.tight_layout()
        plt.show()