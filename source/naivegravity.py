import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# note that units are in terms of AU (distance), day (time) and solar mass (M_sun, mass)
G = 2.959122082855911e-4  # gravitational constant in AU^3 M_sun^-1 day^-2
dt = 1/24  # time step (in units of days, e.g. 1/24 is equivalent to steps of an hour)


class Body:
    """Represents a celestial body with position, velocity, mass and net force."""
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)  # in AU
        self.velocity = np.array(velocity, dtype=float)  # in AU/day
        self.mass = mass  # in solar masses (M_sun)
        self.force = np.zeros(2)  # in AU/day^2, initialised to zero


class Simulation:
    """Handles the physics of the n-body simulation."""
    def __init__(self, bodies):
        self.bodies = bodies

    def compute_forces(self, e = 1e-2) :
        """Naively computes pair-wise gravitational forces between all bodies."""        
        for body in self.bodies:
            body.force = np.zeros(2) # clear previous force calculation

        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i != j:
                    diff_vector = body2.position - body1.position
                    distance = np.linalg.norm(diff_vector)
                    
                    force_magnitude = G * body1.mass * body2.mass / (distance + e) **2
                    force_vector = force_magnitude * diff_vector / distance
                    body1.force += force_vector

    def move(self):
        """Updates the position and velocity of all bodies using leapfrog integration."""
        for body in self.bodies:
            body.position += body.velocity * dt  # update position
        
        self.compute_forces()  # compute new forces after updating the positions
        
        for body in self.bodies:
            body.velocity += (body.force / body.mass) * dt  # update velocity


class Animation:
    """Handles visualization and formatting of the simulation using Matplotlib."""
    def __init__(self, bodies, simulation, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor("black")
        self.ax.set_xlim(-2, 2)  # in AU
        self.ax.set_ylim(-2, 2)  # in AU
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # create a scatter point for each body (size scaled by mass)
        self.scatters = [
            self.ax.plot([], [], "wo", markersize=body.mass * 2)[0] 
            for body in bodies
        ]

        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.steps, interval=self.interval, repeat=True
        )

    def update(self, frame):
        """Updates the animation frame-by-frame."""
        self.simulation.move()
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        return self.scatters

    def show(self):
        plt.show()

###############################################################################################

if __name__ == "__main__":
    np.random.seed(28)
    bodies = [
        Body(
            position=np.random.uniform(-2, 2, 2),  # in AU
            velocity=np.random.uniform(-0.05, 0.05, 2),  # in AU/day
            mass=np.random.uniform(0.1, 1),  # in M_sun
        )
        for _ in range(3)
    ]

    simulation = Simulation(bodies)
    anim = Animation(bodies, simulation)
    anim.show()
