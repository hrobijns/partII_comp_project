import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Gravitational constant in AU^3 M_sun^-1 day^-2
G = 2.959122082855911e-4  
dt = 1/24  # timestep (1 hour)


class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2)


class Simulation:
    def __init__(self, bodies):
        self.bodies = bodies

    def compute_forces(self, e=1e-1):
        for body in self.bodies:
            body.force = np.zeros(2)
        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i != j:
                    diff = body2.position - body1.position
                    dist_squared = np.dot(diff, diff) + e**2
                    dist = np.sqrt(dist_squared)
                    force_mag = G * body1.mass * body2.mass / dist_squared
                    force_vec = force_mag * diff / dist
                    body1.force += force_vec

    def move(self):
        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt
        for body in self.bodies:
            body.position += body.velocity * dt
        self.compute_forces()
        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt

    def total_energy(self, e=1e-1):
        kinetic = sum(0.5 * b.mass * np.linalg.norm(b.velocity)**2 for b in self.bodies)
        potential = 0
        for i, b1 in enumerate(self.bodies):
            for j, b2 in enumerate(self.bodies):
                if i < j:
                    r_squared = np.sum((b1.position - b2.position)**2)
                    potential -= G * b1.mass * b2.mass / np.sqrt(r_squared + e**2)
        return kinetic + potential

    def total_momentum(self):
        total = np.zeros(2)
        for b in self.bodies:
            total += b.mass * b.velocity
        return total


class Animation:
    def __init__(self, bodies, simulation, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval

        self.fig, (self.ax_anim, self.ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))
        self.ax_anim.set_facecolor("black")
        self.ax_anim.set_xlim(-5, 5)
        self.ax_anim.set_ylim(-5, 5)
        self.ax_anim.set_xticks([])
        self.ax_anim.set_yticks([])

        self.scatters = [
            self.ax_anim.plot([], [], "wo", markersize=body.mass * 2)[0] 
            for body in bodies
        ]

        # Energy tracking
        self.energy_data = []
        self.energy_steps = []
        self.energy_line, = self.ax_metrics.plot([], [], "r-", label="Total Energy Δ%")
        self.initial_energy = None
        self.initial_energy_line = None

        # Momentum tracking
        self.momentum_data = []
        self.momentum_line, = self.ax_metrics.plot([], [], "b--", label="Total Momentum Δ%")
        self.initial_momentum = None

        self.ax_metrics.set_title("Conservation: Energy & Momentum (% Change)")
        self.ax_metrics.set_xlabel("Step")
        self.ax_metrics.set_ylabel("% Change")
        self.ax_metrics.set_xlim(0, self.steps)
        self.ax_metrics.set_ylim(-1, 1)  # Initial range

        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.steps, interval=self.interval, repeat=False
        )

    def update(self, frame):
        self.simulation.move()

        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])

        current_energy = self.simulation.total_energy()
        current_momentum = self.simulation.total_momentum()
        momentum_mag = np.linalg.norm(current_momentum)

        if frame == 0:
            self.initial_energy = current_energy
            self.initial_momentum = momentum_mag
            energy_percent_change = 0
            momentum_percent_change = 0
        else:
            energy_percent_change = 100 * (current_energy - self.initial_energy) / abs(self.initial_energy)
            momentum_percent_change = 100 * (momentum_mag - self.initial_momentum) / abs(self.initial_momentum)

        self.energy_data.append(energy_percent_change)
        self.momentum_data.append(momentum_percent_change)
        self.energy_steps.append(frame)

        if frame == 0:
            self.initial_energy_line = self.ax_metrics.axhline(
                y=0, color="black", linestyle="dotted", label="Initial Energy/Momentum"
            )
            self.ax_metrics.legend()

        self.energy_line.set_data(self.energy_steps, self.energy_data)
        self.momentum_line.set_data(self.energy_steps, self.momentum_data)

        min_val = min(min(self.energy_data), min(self.momentum_data))
        max_val = max(max(self.energy_data), max(self.momentum_data))
        padding = (max_val - min_val) * 0.1 if max_val != min_val else 1
        self.ax_metrics.set_ylim(min_val - padding, max_val + padding)

        return self.scatters + [self.energy_line, self.momentum_line]

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    bodies = [
        Body(
            position=np.random.uniform(-5, 5, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(100)
    ]

    sim = Simulation(bodies)
    sim.compute_forces()  # Initialize forces
    anim = Animation(bodies, sim, steps=1000)
    anim.show()
