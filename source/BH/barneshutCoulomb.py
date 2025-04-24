import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters (dimensionless units)
k = 1.0          # Coulomb constant (in chosen units)
dt = 0.01        # time step
theta = 0.5      # Barnes-Hut opening angle
soft = 1e-1      # softening length

class Body:
    """Represents a particle with position, velocity, charge, mass, and net force."""
    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge   = charge   # coupling for Coulomb force
        self.mass     = mass     # inertial mass
        self.force    = np.zeros(2)

class QuadTree:
    """Quadtree for Barnes-Hut approximation of Coulomb forces."""
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.center_of_charge = np.array([0.0, 0.0])
        self.total_charge     = 0.0
        self.bodies           = []
        self.children         = None

    def insert(self, body):
        """Insert a body, updating total_charge and center_of_charge."""
        if not self.bodies and self.total_charge == 0:
            # empty node
            self.bodies.append(body)
            self.center_of_charge = body.position.copy()
            self.total_charge     = body.charge
            return

        if self.children is None:
            self.subdivide()

        # update multipole: monopole only
        prev_q = self.total_charge
        self.total_charge = prev_q + body.charge
        # new center of charge
        self.center_of_charge = (
            self.center_of_charge * prev_q + body.position * body.charge
        ) / self.total_charge

        # propagate to correct quadrant
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
        # re-insert existing bodies
        for body in self.bodies:
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        self.bodies = []

    def contains(self, pos):
        return (self.x_min <= pos[0] < self.x_max) and (self.y_min <= pos[1] < self.y_max)

    def compute_force(self, body, theta):
        """Recursively compute net Coulomb force on 'body'."""
        if self.total_charge == 0 or (len(self.bodies) == 1 and self.bodies[0] is body):
            return np.zeros(2)

        # vector from target body to node's center-of-charge
        dx, dy = body.position - self.center_of_charge
        dist = np.hypot(dx, dy)
        width = self.x_max - self.x_min

        # if sufficiently far, use monopole
        if (width / dist < theta) or self.children is None:
            # Coulomb force magnitude with softening
            f = k * body.charge * self.total_charge / (dist**2 + soft**2)
            return f * np.array([dx, dy]) / (dist + 1e-16)

        # otherwise sum over children
        total = np.zeros(2)
        for child in self.children:
            total += child.compute_force(body, theta)
        return total

class Simulation:
    """Runs the time integration using Velocity-Verlet and Barnes-Hut."""
    def __init__(self, bodies, space_size, theta):
        self.bodies = bodies
        self.space_size = space_size
        self.theta = theta
        self.compute_forces()

    def compute_forces(self):
        root = QuadTree(-self.space_size, self.space_size,
                        -self.space_size, self.space_size)
        for b in self.bodies:
            root.insert(b)
        for b in self.bodies:
            b.force = root.compute_force(b, self.theta)

    def move(self):
        # half-step velocity update
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt
        # full-step position update
        for b in self.bodies:
            b.position += b.velocity * dt
        # recompute forces
        self.compute_forces()
        # second half-step velocity update
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt

class Animation:
    """Visualizes the simulation with Matplotlib."""
    def __init__(self, bodies, sim, steps=500, interval=30):
        self.bodies = bodies
        self.sim = sim
        self.steps = steps
        self.interval = interval
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-sim.space_size, sim.space_size)
        self.ax.set_ylim(-sim.space_size, sim.space_size)
        self.scat = [self.ax.plot([], [], 'wo', markersize=4)[0] for _ in bodies]
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps,
                                 interval=self.interval, blit=True)

    def update(self, i):
        self.sim.move()
        for sc, b in zip(self.scat, self.bodies):
            sc.set_data(b.position[0], b.position[1])
        return self.scat

    def show(self):
        plt.show()

if __name__ == "__main__":
    # one-component Coulomb gas: same-sign charges, equal mass
    np.random.seed(42)
    N = 200
    bodies = [
        Body(
            position=np.random.uniform(-10,10,2),
            velocity=np.random.uniform(-0.1,0.1,2),
            charge=1.0,
            mass=1.0
        ) for _ in range(N)
    ]
    sim = Simulation(bodies, space_size=10, theta=theta)
    anim = Animation(bodies, sim)
    anim.show()
