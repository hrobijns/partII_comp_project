import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 2.959122082855911e-4
dt = 1 / 24
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
            QuadTree(self.x_min, x_mid, self.y_min, y_mid),  # SW (D)
            QuadTree(x_mid, self.x_max, self.y_min, y_mid),  # SE (C)
            QuadTree(self.x_min, x_mid, y_mid, self.y_max),  # NW (A)
            QuadTree(x_mid, self.x_max, y_mid, self.y_max),  # NE (B)
        ]
        for body in self.bodies:
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        self.bodies = []

    def contains(self, position):
        return self.x_min <= position[0] < self.x_max and self.y_min <= position[1] < self.y_max

    def draw(self, ax):
        ax.plot([self.x_min, self.x_max], [self.y_min, self.y_min], 'gray', lw=0.5)
        ax.plot([self.x_min, self.x_max], [self.y_max, self.y_max], 'gray', lw=0.5)
        ax.plot([self.x_min, self.x_min], [self.y_min, self.y_max], 'gray', lw=0.5)
        ax.plot([self.x_max, self.x_max], [self.y_min, self.y_max], 'gray', lw=0.5)
        if self.children:
            for child in self.children:
                child.draw(ax)


def draw_tree_structure(
    ax, node, x=0, y=0, x_offset=8, y_offset=2, depth=1, node_radius=0.15,
    label_children=False, labels=None, label_offsets=None
):
    is_filled = node.total_mass > 0

    if is_filled:
        ax.plot(x, y, 'ko', markersize=6)
    else:
        ax.plot(x, y, marker='o', markersize=6, markerfacecolor='none', markeredgecolor='k')

    if node.children:
        num_children = len(node.children)
        spread = x_offset / (depth)
        if depth == 2:
            spread += 0.9
        labels = ['A', 'B', 'C', 'D']  # Order of child quadrants

        for i, child in enumerate(node.children):
            dx = (i - (num_children - 1) / 2) * spread
            child_x = x + dx
            child_y = y - y_offset

            vec = np.array([child_x - x, child_y - y])
            dist = np.linalg.norm(vec)
            vec = vec / dist if dist != 0 else np.zeros(2)

            start = np.array([x, y]) + vec * node_radius * 0.5
            end = np.array([child_x, child_y]) - vec * node_radius 
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', lw=0.8)

            # Label the first-level children
            if depth == 1 and label_children and labels and label_offsets:
                label_offset = label_offsets[i]
                ax.text(
                    child_x + label_offset[0],
                    child_y + label_offset[1],
                    labels[i],
                    fontsize=11,
                    weight='bold'
                )

            draw_tree_structure(ax, child, child_x, child_y, x_offset, y_offset, depth + 1, node_radius)


# --- Generate Data & Build Quadtree ---
np.random.seed(36)
bodies = [
    Body(
        position=np.random.uniform(-1, 1, 2),
        velocity=np.random.uniform(-0.05, 0.05, 2),
        mass=np.random.uniform(0.1, 1),
    )
    for _ in range(10)
]

space_size = 1
root = QuadTree(-space_size, space_size, -space_size, space_size)
for body in bodies:
    root.insert(body)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
labels = ['A', 'B', 'C', 'D']
label_offsets = [
    (-0.25, 0.2),   # D (SW)
    (-1.0,  0.2),  # C (SE)
    (-0.15, 0.2),   # A (NW)
    (0.0,   0.15)   # B (NE)
]

# Plot 1: Quadtree spatial grid
ax1.set_facecolor('black')
ax1.set_xlim(-space_size, space_size)
ax1.set_ylim(-space_size, space_size)
ax1.set_xticks([])
ax1.set_yticks([])
for body in bodies:
    ax1.plot(body.position[0], body.position[1], 'wo', markersize=body.mass * 5 + 2)
root.draw(ax1)
#ax1.set_title('Quadtree Spatial Partition', color='white')
ax1.tick_params(colors='white')

# Add quadrant labels: A (NW), B (NE), C (SE), D (SW)
margin = 0.05
ax1.text(-1 + margin, 1 - margin - 0.04, 'C', color='white', fontsize=12, weight='bold')  # NW
ax1.text(1 - 2*margin, 1 - margin -0.04, 'D', color='white', fontsize=12, weight='bold')  # NE
ax1.text(1 - 2*margin, -1 + margin, 'B', color='white', fontsize=12, weight='bold')  # SE
ax1.text(-1 + margin, -1 + margin, 'A', color='white', fontsize=12, weight='bold')  # SW

# Plot 2: Quadtree tree diagram
ax2.axis('off')
draw_tree_structure(
    ax2, root,
    label_children=True,
    labels=labels,
    label_offsets=label_offsets
)

#plt.savefig("figures/quadtree_plot.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()