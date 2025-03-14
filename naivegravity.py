#################################################################################
# import the necessary libraries, and define relevant constants

import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.animation import FuncAnimation

G = 6.67430e-11  # gravitational constant
mass = 5e26 # masses of the body (on the order of the mass of venus)
dt = 3600 # time step (day)

#################################################################################
# the direct/naive approach to the n-body gravitational problem, O(n^2)

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.array([0.0, 0.0])  # 2D force vector

    def pairwise_force(self, other):
        assert self is not other
        diff_vector = other.position - self.position
        distance = np.linalg.norm(diff_vector)
        assert np.abs(distance) > 6e6, 'Bodies collided!'
        f_mag = G * self.mass * other.mass / (distance**2)
        f = f_mag * diff_vector / np.linalg.norm(diff_vector)  # Only 2D forces
        return f
    
    def total_energy(bodies):
        total_kinetic = sum(0.5 * body.mass * np.linalg.norm(body.velocity)**2 for body in bodies)
        total_potential = 0
        for b1, b2 in itertools.combinations(bodies, 2):
            distance = np.linalg.norm(b2.position - b1.position)
            total_potential -= G * b1.mass * b2.mass / distance
        return total_kinetic + total_potential
    
    def move(bodies):
        pairs = itertools.combinations(bodies, 2)
        # initialize force vectors
        for b in bodies:
            b.force = np.array([0.0, 0.0])  # 2D force vector
        # calculate force vectors
        for b1, b2 in pairs:
            f = b1.pairwise_force(b2)
            b1.force += f
            b2.force -= f
        # update velocities based on force, update positions based on velocity
        for body in bodies:
            body.velocity += body.force / body.mass * dt
            body.position += body.velocity * dt

#################################################################################
# create an animation, with a plot of energy            

class Animation:
    def __init__(self, bodies, steps=100, interval=50):
        self.bodies = bodies
        self.steps = steps
        self.interval = interval
        self.fig, (self.ax, self.ax_energy) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-1e11, 1e11)
        self.ax.set_ylim(-1e11, 1e11)
        self.scatters = [self.ax.plot([], [], 'wo', markersize=2)[0] for _ in bodies]
        
        self.energy_data = []  # to store the total energy over time
        self.time_data = []    # to store the time steps

        # initialize the energy plot with a line object (this will be updated)
        self.energy_line, = self.ax_energy.plot([], [], color='blue')
        self.ax_energy.set_title("Total Energy Over Time")
        self.ax_energy.set_xlabel("Time (days)")
        self.ax_energy.set_ylabel("Total Energy (J)")

        # initial limits for the energy plot
        self.ax_energy.set_xlim(0, self.steps * dt)
        self.ax_energy.set_ylim(1e33, 2e33)  # initial limits, will be updated dynamically

        # removing axes for the animation plot
        self.ax.set_xticks([])  
        self.ax.set_yticks([]) 
        self.ax.set_xlabel('')  
        self.ax.set_ylabel('') 

        self.ani = FuncAnimation(self.fig, self.update, frames=99999, interval=self.interval, repeat=True)

        self.energy_min = float('inf')
        self.energy_max = float('-inf')

    def update(self, frame):
        Body.move(self.bodies)
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        
        # update energy plot
        total_energy = Body.total_energy(self.bodies)
        self.energy_data.append(total_energy)
        self.time_data.append(frame * dt)

        self.energy_min = min(self.energy_min, total_energy)
        self.energy_max = max(self.energy_max, total_energy)

        self.energy_line.set_data(self.time_data, self.energy_data)
        self.ax_energy.set_ylim(self.energy_min * 1.1, self.energy_max * 1.1)  
        self.ax_energy.set_xlim(0, frame * dt)  # extend x-axis as time progresses
        
        return self.scatters, self.energy_line
    
    def show(self):
        plt.show()

#################################################################################
# simulate, and plot the force calculations as a function of n (number of bodies)

np.random.seed(56)
bodies = [
    Body(
        position=np.random.uniform(-1e11, 1e11, 2),
        velocity=np.random.uniform(-3e3, 3e3, 2),
        mass=mass
    ) for _ in range(100)
]

anim = Animation(bodies)
anim.show()