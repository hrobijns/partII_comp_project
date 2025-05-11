import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from quadtree import QuadTree

class Body:
    """Simple charged particle."""
    def __init__(self, position, charge=1.0):
        self.position = np.array(position, dtype=float)
        self.charge = charge


def build_tree(bounds, bodies):
    root = QuadTree(*bounds)
    for b in bodies:
        root.insert(b)
    return root


def plot_spatial(node, ax):
    """Recursively plot the rectangle for each node."""
    width = node.x_max - node.x_min
    height = node.y_max - node.y_min
    rect = Rectangle((node.x_min, node.y_min), width, height,
                     fill=False, edgecolor='gray', linewidth=0.8)
    ax.add_patch(rect)
    if node.children:
        for c in node.children:
            plot_spatial(c, ax)


def collect_positions(node, depth=0, y_counter=[0], positions=None):
    if positions is None:
        positions = {}
    if node.children is None:
        y = y_counter[0]
        positions[node] = (depth, y)
        y_counter[0] += 1
    else:
        child_ys = []
        for c in node.children:
            collect_positions(c, depth+1, y_counter, positions)
            child_ys.append(positions[c][1])
        positions[node] = (depth, sum(child_ys)/len(child_ys))
    return positions


def plot_tree_diagram(root, ax, r=0.25, min_span=0.1):
    """
    r        = per generation shrink ratio
    min_span = minimum allowed distance 
    """
    positions = {}
    def recurse(node, depth=0, x=0.5):
        positions[node] = (x, depth)
        if node.children:
            n = len(node.children)
            span = max(r**depth, min_span)
            for i, child in enumerate(node.children):
                # equally‐spaced offsets in [-span/2, +span/2]
                offset = ((2*i + 1 - n) / n) * (span/2)
                recurse(child, depth+1, x + offset)

    recurse(root)

    # draw links
    for node, (x0, y0) in positions.items():
        for c in getattr(node, "children") or []:
            x1, y1 = positions[c]
            ax.plot([x0, x1], [y0, y1], "-k", linewidth=0.7)

    # scatter nodes
    inhabited = [n for n in positions if n.total_charge > 0]
    empty     = [n for n in positions if n.total_charge == 0 and not n.children]

    if inhabited:
        xi, yi = zip(*(positions[n] for n in inhabited))
        ax.scatter(xi, yi, s=30, marker="o", facecolors="k", edgecolors="k")
    if empty:
        xe, ye = zip(*(positions[n] for n in empty))
        ax.scatter(xe, ye, s=30, marker="o", facecolors="none", edgecolors="k")

    ax.set_frame_on(False)
    ax.axis("off")
    ax.invert_yaxis()


def main():
    np.random.seed(42)

    # generate random bodies
    N = 10
    pts = np.random.rand(N, 2)
    bodies = [Body(p) for p in pts]

    # build quadtree over [0,1]x[0,1]
    bounds = (0.0, 1.0, 0.0, 1.0)
    root = build_tree(bounds, bodies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # left: spatial partition + particles
    plot_spatial(root, ax1)
    ax1.scatter(pts[:,0], pts[:,1], c='blue', s=20)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    # label corners
    corners = {'SW': (bounds[0], bounds[2]-0.03), 'SE': (bounds[1]+0.03, bounds[2]-0.03),
               'NW': (bounds[0], bounds[3]), 'NE': (bounds[1]+0.04, bounds[3])}
    for label, (x, y) in corners.items():
        ax1.text(x, y, label, va='bottom', ha='right')

    # right: tree diagram
    plot_tree_diagram(root, ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()