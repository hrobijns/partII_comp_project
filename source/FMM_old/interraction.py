import matplotlib.pyplot as plt

def plot_quadtree(root, particles=None, show_particles=True, show_centers=True, show_interactions=False):
    """Plot the quadtree structure, optionally showing particle locations and interaction links."""
    fig, ax = plt.subplots(figsize=(8,8))

    # Recursively draw nodes
    def draw_node(node):
        x_min, x_max, y_min, y_max = node.boundary
        width = x_max - x_min
        height = y_max - y_min
        ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor='black', fill=False, lw=0.7))
        
        if show_centers:
            cx, cy = node.center
            ax.plot(cx, cy, 'r.', markersize=4)

        if not node.is_leaf():
            for child in node.children:
                draw_node(child)

    draw_node(root)

    # Draw particles
    if show_particles and particles is not None:
        xs = [p.x for p in particles]
        ys = [p.y for p in particles]
        ax.scatter(xs, ys, color='blue', s=10, label='Particles')

    # Draw interactions
    if show_interactions:
        def draw_interactions(node):
            if node.is_leaf():
                for inter in node.interaction_set:
                    x1, y1 = node.center
                    x2, y2 = inter.center
                    ax.plot([x1, x2], [y1, y2], 'g--', lw=0.5)
            else:
                for child in node.children:
                    draw_interactions(child)
        draw_interactions(root)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('Quadtree Structure')
    plt.grid(True)
    plt.legend()
    plt.show()


from quadtree import build_tree
from simulation import Particle

particles = [
    Particle(0.0, 0.0, 1.0),
    Particle(2.0, 0.0, -1.0),
    Particle(1.0, 1.0, 0.5),
    Particle(-1.0, -1.0, -0.5),
    Particle(-2.0, 2.0, 1.5)
]

tree = build_tree(particles, tree_thresh=1)

# plot tree only
#plot_quadtree(tree.root, particles=particles)

# plot tree + centers + interactions
plot_quadtree(tree.root, particles=particles, show_particles=True, show_centers=True, show_interactions=True)