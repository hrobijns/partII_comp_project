import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import comb as binomial
from math import factorial
import cmath

G = 1  # Set G = 1 for 2D log potential
dt = 1 / 1000  # timestep in days (1 hour)

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2)

class Node:
    def __init__(self, bounds, particles=None):
        self.bounds = bounds
        self.particles = particles if particles else []
        self.children = []
        self.center = None
        self.outer_expansion = None
        self.inner_expansion = None
        self.parent = None
        self.total_mass = 0

    def is_leaf(self):
        return len(self.children) == 0

class Simulation:
    def __init__(self, bodies, max_depth=10, max_particles_per_node=5, expansion_order=5, epsilon=0.1):
        self.bodies = bodies
        self.max_depth = max_depth
        self.max_particles_per_node = max_particles_per_node
        self.p = expansion_order
        self.epsilon = epsilon
        self.rebuild_tree()

    def rebuild_tree(self):
        self.root_node = self.build_tree(self.bodies)
        self.compute_center_and_expansions(self.root_node)
        self.compute_forces()

    def build_tree(self, bodies):
        margin = 1e-3
        xs = [b.position[0] for b in bodies]
        ys = [b.position[1] for b in bodies]
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        bounds = (min_x, min_y, max_x, max_y)
        return self._build_tree(bodies, bounds)

    def _build_tree(self, particles, bounds, depth=0):
        if len(particles) <= self.max_particles_per_node or depth >= self.max_depth:
            return Node(bounds, particles)

        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        quadrants = [
            (bounds[0], bounds[1], cx, cy),
            (cx, bounds[1], bounds[2], cy),
            (bounds[0], cy, cx, bounds[3]),
            (cx, cy, bounds[2], bounds[3])
        ]

        children = []
        for q in quadrants:
            q_particles = [p for p in particles if q[0] <= p.position[0] < q[2] and q[1] <= p.position[1] < q[3]]
            if q_particles:
                child = self._build_tree(q_particles, q, depth + 1)
                child.parent = None
                children.append(child)

        node = Node(bounds)
        node.children = children
        for child in children:
            child.parent = node
        return node

    def compute_center_and_expansions(self, node):
        p = self.p
        node.center = complex((node.bounds[0] + node.bounds[2]) / 2,
                              (node.bounds[1] + node.bounds[3]) / 2)
        node.outer_expansion = np.zeros(p + 1, dtype=complex)

        if node.is_leaf():
            for b in node.particles:
                z = complex(*b.position) - node.center
                for k in range(p + 1):
                    node.outer_expansion[k] += b.mass * z**k / factorial(k)
        else:
            for child in node.children:
                self.compute_center_and_expansions(child)
                dz = child.center - node.center
                for n in range(p + 1):
                    for k in range(n + 1):
                        if k == 0:
                            node.outer_expansion[n] += child.outer_expansion[n]
                        else:
                            node.outer_expansion[n] += child.outer_expansion[n - k] * dz**k

    def compute_local_expansions(self, node, parent_inner=None):
        p = self.p
        node.inner_expansion = np.zeros(p + 1, dtype=complex)
        if parent_inner is not None:
            dz = node.center - node.parent.center
            for n in range(p + 1):
                for k in range(n, p + 1):
                    node.inner_expansion[n] += parent_inner[k] * binomial(k, n) * dz**(k - n)

        if node.parent:
            for nbr in self.get_well_separated_nodes(node):
                dz = node.center - nbr.center
                for n in range(p + 1):
                    for k in range(p + 1):
                        node.inner_expansion[n] += nbr.outer_expansion[k] * (-1)**k * factorial(k + n) / (dz**(k + n + 1) * factorial(k) * factorial(n))

        for child in node.children:
            self.compute_local_expansions(child, node.inner_expansion)

    def compute_forces(self):
        self.compute_local_expansions(self.root_node)
        for body in self.bodies:
            node = self.find_leaf(self.root_node, body)
            z = complex(*body.position) - node.center
            force = 0
            for k in range(1, self.p + 1):
                force += -k * node.inner_expansion[k] * z**(k - 1)
            softened_force = np.array([force.real, force.imag]) * G

            # Total force = multipole + softened direct
            softened_force += self.direct_interactions(body, node)
            body.force = softened_force

    def direct_interactions(self, body, node):
        f = np.zeros(2)
        eps2 = self.epsilon ** 2
        for other in node.particles:
            if other is not body:
                r_vec = other.position - body.position
                r2 = np.dot(r_vec, r_vec) + eps2
                f += G * other.mass * r_vec / r2
        return f

    def find_leaf(self, node, body):
        if node.is_leaf():
            return node
        x, y = body.position
        for child in node.children:
            bx0, by0, bx1, by1 = child.bounds
            if bx0 <= x < bx1 and by0 <= y < by1:
                return self.find_leaf(child, body)
        return node

    def get_well_separated_nodes(self, node):
        if node.parent is None:
            return []
        neighbors = []
        for sibling in node.parent.children:
            if sibling != node:
                neighbors.append(sibling)
        return neighbors

    def move(self):
        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt
            body.position += body.velocity * dt

        self.rebuild_tree()

        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt

class Animation:
    def __init__(self, bodies, simulation, steps=100, interval=200):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor("black")
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.scatters = [
            self.ax.plot([], [], "wo", markersize=body.mass)[0]
            for body in bodies
        ]

        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.steps, interval=self.interval, repeat=True
        )

    def update(self, frame):
        self.simulation.move()
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        return self.scatters

    def show(self):
        plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    bodies = [
        Body(
            position=np.random.uniform(-1, 1, 2),
            velocity=np.random.uniform(-0.01, 0.01, 2),
            mass=np.random.uniform(0.1,0.5),
        )
        for _ in range(20)
    ]

    sim = Simulation(bodies, expansion_order=4)
    anim = Animation(bodies, sim)
    anim.show()