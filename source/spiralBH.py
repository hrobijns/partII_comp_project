import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 2.959122082855911e-4  # AU^3 M_sun^-1 day^-2
dt = 1/24  # 1 hour timestep
theta = 0.5  # Barnes-Hut parameter
e = 1e-1  # Softening factor

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
    def __init__(self, bodies, space_size, theta, black_hole):
        self.bodies = bodies
        self.space_size = space_size
        self.theta = theta
        self.black_hole = black_hole
        self.compute_forces()

    def compute_forces(self):
        root = QuadTree(-self.space_size, self.space_size, -self.space_size, self.space_size)
        for body in self.bodies:
            root.insert(body)
        for body in self.bodies:
            body.force = root.compute_force(body, self.theta)

    def move(self):
        for body in self.bodies:
            if body is not self.black_hole:
                body.velocity += 0.5 * (body.force / body.mass) * dt
                body.position += body.velocity * dt
                

        self.compute_forces()

        for body in self.bodies:
            if body is not self.black_hole:
                body.velocity += 0.5 * (body.force / body.mass) * dt


class Animation:
    def __init__(self, bodies, simulation, black_hole, steps=100, interval=50, output_file="figures/spiralBH.mp4"):
        self.bodies = bodies
        self.simulation = simulation
        self.black_hole = black_hole
        self.steps = steps
        self.interval = interval
        self.output_file = output_file  # Define the output_file argument here
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.scatters = []

        for body in bodies:
            if body is not self.black_hole:
                scatter = self.ax.plot([], [], 'wo', markersize=1, markeredgewidth=0)[0]
                self.scatters.append(scatter)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # This makes sure the animation object is retained
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps,
                                 interval=self.interval, repeat=True)

        # Call tight_layout to ensure minimal whitespace
        self.fig.tight_layout()

    def update(self, frame):
        print(f"Rendering frame {frame + 1}/{self.steps}")
        self.simulation.move()
        idx = 0
        for body in self.bodies:
            if body is not self.black_hole:
                self.scatters[idx].set_data(body.position[0], body.position[1])
                idx += 1
        return self.scatters
    
    def save(self):
        # Save the animation to the specified output file
        self.ani.save(self.output_file, writer='ffmpeg', dpi=300)

    def show(self):
        plt.show()


def generate_spiral_galaxy(n_bodies, arms=2, arm_strength=0.5, spread=0.5, radius=1.5):
    bodies = []
    for i in range(n_bodies):
        # Radial distribution, denser near center
        r = radius * np.random.power(2.5)
        
        # Spiral angle: looser connection to radial distance
        base_theta = r * 4  # logarithmic spiral
        arm_offset = ((i % arms) / arms) * 2 * np.pi
        theta_noise = np.random.normal(scale=spread * (1 - arm_strength))
        theta = base_theta + arm_offset + theta_noise
        
        # Add perpendicular noise to spread stars around the arm
        offset_radius = r + np.random.normal(scale=spread * arm_strength)

        x = offset_radius * np.cos(theta)
        y = offset_radius * np.sin(theta)
        position = np.array([x, y])

        # Calculate orbital velocity around center (black hole mass = 100 M_sun)
        distance = np.linalg.norm(position)
        speed = np.sqrt(G * 100 / (distance + 0.1))
        direction = np.array([position[1], -position[0]])
        direction /= np.linalg.norm(direction)
        velocity = direction * speed

        # Star mass
        mass = np.random.uniform(0.05, 0.5)
        bodies.append(Body(position, velocity, mass))
    
    return bodies


#####################################################################################################

if __name__ == "__main__":
    np.random.seed(42)

    # Add central black hole
    black_hole = Body(position=[0, 0], velocity=[0, 0], mass=100.0)

    # Generate spiral galaxy
    stars = generate_spiral_galaxy(n_bodies=2000)
    bodies = [black_hole] + stars


    simulation = Simulation(bodies, space_size=1, theta=theta, black_hole=black_hole)
    anim = Animation(bodies, simulation, black_hole=black_hole, steps=100, interval=20)

    anim.save()