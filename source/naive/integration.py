import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from simulation import SimulationVectorised, k, soft, dt

def compute_energy(pos, vel, charge, mass):
    """Total energy (kinetic + potential) for 2D Coulomb/log system."""
    KE = 0.5 * np.sum(mass[:, None] * vel**2)
    N = pos.shape[0]
    PE = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[i] - pos[j]
            r = np.sqrt(np.dot(diff, diff) + soft**2)
            PE -= k * charge[i] * charge[j] * np.log(r)
    return KE + PE

def euler_step(pos, vel, charge, mass):
    """One explicit Euler step under 2D Coulomb forces."""
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff * diff, axis=2) + soft**2
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    qq = np.outer(charge, charge)
    Fmat = k * qq[:, :, None] * diff * inv_r2[:, :, None]
    force = np.sum(Fmat, axis=1)
    vel_new = vel + (force / mass[:, None]) * dt
    pos_new = pos + vel * dt
    return pos_new, vel_new

class ComparisonAnimation:
    """Left: particle scatter (Verlet vs Euler).
       Right: evolving %ΔE for both integrators."""
    def __init__(self, sim_v, pos_e, vel_e, charge, mass, dt, steps=500,
                 xlim=(-2,2), ylim=(-2,2), interval=50):
        self.sim_v   = sim_v
        self.pos_e   = pos_e.copy()
        self.vel_e   = vel_e.copy()
        self.charge  = charge
        self.mass    = mass
        self.dt      = dt
        self.steps   = steps

        # record %ΔE: row 0 = Verlet, row 1 = Euler
        self.Ehist   = np.zeros((2, steps))
        self.idx     = np.arange(steps)

        # initial energies
        self.E0_v = compute_energy(sim_v.pos, sim_v.vel, charge, mass)
        self.E0_e = compute_energy(pos_e, vel_e, charge, mass)

        # set up figure + axes
        self.fig, (self.ax_sim, self.ax_E) = plt.subplots(
            1, 2, figsize=(10,5), gridspec_kw={'width_ratios':[1,1.2]}
        )

        # --- Left: scatter of both sims ---
        self.ax_sim.set_xlim(*xlim); self.ax_sim.set_ylim(*ylim)
        self.ax_sim.set_xticks([]); self.ax_sim.set_yticks([])
        self.scat_v = self.ax_sim.scatter(
            sim_v.pos[:,0], sim_v.pos[:,1], s=30, label='kick-drift-kick'
        )
        self.scat_e = self.ax_sim.scatter(
            pos_e[:,0],   pos_e[:,1],   s=30,  label='Euler'
        )
        self.ax_sim.legend(loc='upper right')

        # --- Right: %ΔE plot ---
        self.lines = []
        for label in ('kick-drift-kick', 'Euler'):
            line, = self.ax_E.plot([], [], label=label)
            self.lines.append(line)
        self.ax_E.axhline(0, color='black', linestyle=':')
        self.ax_E.set_xlim(0, steps+1)
        self.ax_E.set_ylim(-10, 20)   # adjust if needed
        self.ax_E.set_xlabel('step')
        self.ax_E.set_ylabel('% ΔE')
        self.ax_E.legend(loc='upper right')
        self.ax_E.grid(True)

        # create the animation
        self.ani = FuncAnimation(
            self.fig, self.update, frames=steps,
            interval=interval, blit=True
        )

    def update(self, frame):
        # 1) advance Verlet
        self.sim_v.step()
        Ev = compute_energy(self.sim_v.pos, self.sim_v.vel,
                            self.charge, self.mass)
        self.Ehist[0, frame] = (Ev - self.E0_v)/abs(self.E0_v)*100

        # 2) advance Euler
        self.pos_e, self.vel_e = euler_step(
            self.pos_e, self.vel_e, self.charge, self.mass
        )
        Ee = compute_energy(self.pos_e, self.vel_e,
                            self.charge, self.mass)
        self.Ehist[1, frame] = (Ee - self.E0_e)/abs(self.E0_e)*100

        # update scatters
        self.scat_v.set_offsets(self.sim_v.pos)
        self.scat_e.set_offsets(self.pos_e)

        # update error curves
        for i, line in enumerate(self.lines):
            line.set_data(self.idx[:frame+1], self.Ehist[i,:frame+1])

        return [self.scat_v, self.scat_e, *self.lines]

    def show(self):
        plt.tight_layout()
        plt.show()

    def save(self, filename, fps=30):
        writer = FFMpegWriter(fps=fps, codec='h264')
        self.ani.save(filename, writer=writer)

if __name__ == '__main__':
    # -- initialize --
    N = 75
    np.random.seed(38)
    pos0   = np.random.uniform(-2,2,(N,2))
    vel0   = np.random.normal(0,0.1,(N,2))
    charge = np.random.uniform(-1,1,N)
    mass   = np.random.uniform(0.5,1.5,N)

    # prepare Verlet sim
    sim_v = SimulationVectorised(pos0.copy(), vel0.copy(), charge, mass)
    sim_v.compute_forces()

    # Prepare and run animation
    anim = ComparisonAnimation(
        sim_v, pos0, vel0, charge, mass, dt,
        steps=500, xlim=(-2,2), ylim=(-2,2), interval=5
    )
    #anim.show()

    anim.save('figures/integration.mp4', fps=100)