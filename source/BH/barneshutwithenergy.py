import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 2.959122082855911e-4  # AU^3 M_sun^-1 day^-2
dt = 1/24
theta = 0.5
e = 1e-1


class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2)


class QuadTree:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.center_of_mass = np.array([0.0, 0.0])
        self.total_mass = 0.0
        self.bodies = []
        self.children = None

    def insert(self, body):
        if not self.bodies and self.total_mass == 0:
            self.bodies.append(body)
            self.center_of_mass = body.position
            self.total_mass = body.mass
            return

        if self.children is None:
            self.subdivide()

        self.total_mass += body.mass
        self.center_of_mass = (self.center_of_mass * (self.total_mass - body.mass)
                               + body.position * body.mass) / self.total_mass

        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return

    def subdivide(self):
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        self.children = [
            QuadTree(self.x_min, x_mid, self.y_min, y_mid),
            QuadTree(x_mid, self.x_max, self.y_min, y_mid),
            QuadTree(self.x_min, x_mid, y_mid, self.y_max),
            QuadTree(x_mid, self.x_max, y_mid, self.y_max)
        ]
        for body in self.bodies:
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        self.bodies = []

    def contains(self, position):
        return self.x_min <= position[0] < self.x_max and self.y_min <= position[1] < self.y_max

    def compute_force(self, body, theta):
        if not self.total_mass or (len(self.bodies) == 1 and self.bodies[0] is body):
            return np.array([0.0, 0.0])

        dx, dy = self.center_of_mass - body.position
        distance = np.sqrt(dx**2 + dy**2)
        width = self.x_max - self.x_min

        if width / distance < theta or not self.children:
            force_magnitude = G * body.mass * self.total_mass / (distance**2 + e**2)
            return force_magnitude * np.array([dx, dy]) / distance

        total_force = np.array([0.0, 0.0])
        for child in self.children:
            total_force += child.compute_force(body, theta)
        return total_force


class Simulation:
    def __init__(self, bodies, space_size, theta):
        self.bodies = bodies
        self.space_size = space_size
        self.theta = theta
        self.compute_forces()

    def compute_forces(self):
        root = QuadTree(-self.space_size, self.space_size, -self.space_size, self.space_size)
        for body in self.bodies:
            root.insert(body)
        for body in self.bodies:
            body.force = root.compute_force(body, self.theta)

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
        return sum((b.mass * b.velocity for b in self.bodies), start=np.zeros(2))


class Animation:
    def __init__(self, bodies, simulation, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval

        self.fig, (self.ax_anim, self.ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))
        self.ax_anim.set_facecolor("black")
        self.ax_anim.set_xlim(-2, 2)
        self.ax_anim.set_ylim(-2, 2)
        self.ax_anim.set_xticks([])
        self.ax_anim.set_yticks([])

        self.scatters = [
            self.ax_anim.plot([], [], "wo", markersize=body.mass * 2)[0] 
            for body in bodies
        ]

        self.energy_data = []
        self.momentum_data = []
        self.steps_list = []

        self.energy_line, = self.ax_metrics.plot([], [], "r-", label="Total Energy Δ%")
        self.momentum_line, = self.ax_metrics.plot([], [], "b--", label="Total Momentum Δ%")
        self.initial_energy = None
        self.initial_momentum = None

        self.ax_metrics.set_title("Energy & Momentum (% Change)")
        self.ax_metrics.set_xlabel("Step")
        self.ax_metrics.set_ylabel("Percent Change")
        self.ax_metrics.set_xlim(0, self.steps)
        self.ax_metrics.set_ylim(-1, 1)
        self.ax_metrics.legend()

        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.steps, interval=self.interval, repeat=False
        )

    def update(self, frame):
        self.simulation.move()

        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])

        current_energy = self.simulation.total_energy()
        current_momentum_mag = np.linalg.norm(self.simulation.total_momentum())

        if frame == 0:
            self.initial_energy = current_energy
            self.initial_momentum = current_momentum_mag
            energy_percent = 0
            momentum_percent = 0
        else:
            energy_percent = 100 * (current_energy - self.initial_energy) / abs(self.initial_energy)
            momentum_percent = 100 * (current_momentum_mag - self.initial_momentum) / abs(self.initial_momentum)

        self.energy_data.append(energy_percent)
        self.momentum_data.append(momentum_percent)
        self.steps_list.append(frame)

        self.energy_line.set_data(self.steps_list, self.energy_data)
        self.momentum_line.set_data(self.steps_list, self.momentum_data)

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
            position=np.random.uniform(-200, 200, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(100)
    ]

    sim = Simulation(bodies, space_size=2, theta=0.5)
    sim.compute_forces()
    anim = Animation(bodies, sim, steps=1000)
    anim.show()
