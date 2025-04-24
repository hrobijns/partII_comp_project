import time
import numpy as np
import matplotlib.pyplot as plt
# import the pieces from your FMM module
from working import Body, build_tree, _outer_mpexp, _inner

# Benchmark parameters
expansion_order = 4    # your multipole order
steps           = 5    # repeat each measurement this many times
tree_thresh     = 10   # same as in your Simulation
init_velocity   = (0.0, 0.0)

# Range of N to test
n_bodies_list = np.linspace(10, 5000, 10, dtype=int)

# Storage for averaged timings
tree_times = []
eval_times = []

for N in n_bodies_list:
    print(f"Testing N = {N}")
    np.random.seed(48)
    bodies = [
        Body(position=tuple(np.random.uniform(-1000,1000,2)),
             velocity=init_velocity,
             mass=np.random.uniform(0.1,1.0))
        for _ in range(N)
    ]

    t_build_acc = 0.0
    t_eval_acc  = 0.0

    for _ in range(steps):
        # --- time tree build ---
        tb0 = time.time()
        tree = build_tree(bodies, tree_thresh)
        tb1 = time.time()
        t_build_acc += (tb1 - tb0)

        # reset potentials & forces
        for p in bodies:
            p.phi = p.fx = p.fy = 0.0

        # --- time FMM evaluation ---
        te0 = time.time()
        _outer_mpexp(tree.root, expansion_order)
        tree.root.inner = np.zeros(expansion_order+1, dtype=complex)
        any(_inner(child) for child in tree.root)
        te1 = time.time()
        t_eval_acc += (te1 - te0)

    # average over repeats
    tree_times.append(t_build_acc / steps)
    eval_times.append(t_eval_acc  / steps)


total_times = [t_build + t_eval
               for t_build, t_eval in zip(tree_times, eval_times)]

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(n_bodies_list, tree_times,
        marker='o', linestyle='-',
        label='Quadtree build time')
ax.plot(n_bodies_list, total_times,
        marker='s', linestyle='--',
        label='Total FMM time')

ax.set_xlabel('Number of bodies')
ax.set_ylabel('Time per step (s)')
ax.set_title(f'FMM Timing (order={expansion_order})')
ax.legend()
ax.grid(False)

plt.tight_layout()
plt.show()