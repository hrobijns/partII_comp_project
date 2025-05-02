import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple, List

from quadtree import QuadTree, QuadNode

class Body:
    def __init__(self, position: Tuple[float, float]):
        self.position = position

def find_path_to_leaf(node: QuadNode, pos: Tuple[float, float], path: List[QuadNode]=None) -> List[QuadNode]:
    """Return the list of nodes from `node` down to the leaf containing `pos`."""
    if path is None:
        path = []
    path.append(node)
    if node.is_leaf():
        return path
    for c in node.children:
        if c.contains(pos):
            return find_path_to_leaf(c, pos, path)
    raise ValueError(f"Position {pos} not found in any leaf")

def draw_grid_at_level(ax, node: QuadNode, target_level: int, greyed: List[QuadNode]):
    """
    Draws the quadtree grid at `target_level`, but stops subdividing
    inside any region listed in `greyed`.
    """
    # If this node has been greyed, draw its boundary and do not subdivide further
    if node in greyed:
        ax.add_patch(Rectangle(
            (node.x_min, node.y_min),
            node.x_max - node.x_min,
            node.y_max - node.y_min,
            fill=False, edgecolor="black", linewidth=0.8
        ))
        return

    # If we've reached the desired level, draw this cell
    if node.level == target_level:
        ax.add_patch(Rectangle(
            (node.x_min, node.y_min),
            node.x_max - node.x_min,
            node.y_max - node.y_min,
            fill=False, edgecolor="black", linewidth=0.8
        ))
        return

    # Otherwise, subdivide further (unless it's a leaf, but in uniform tree it won't be)
    for c in node.children:
        draw_grid_at_level(ax, c, target_level, greyed)

def main():
    random.seed(42)

    # 1) Generate 300 random body positions
    bodies = [Body((random.random(), random.random())) for _ in range(300)]

    # 2) Build the uniform quadtree (~1–5 points per leaf)
    qt = QuadTree(bodies, boundary=(0,0,1,1), max_per_leaf=3)

    # 3) Compute the path from root to the chosen body's leaf
    target_body = bodies[5]
    path = find_path_to_leaf(qt.root, target_body.position)

    # 4) Prepare figure: one column per level EXCLUDING the root (level 0)
    path = path[1:]  # drop the root
    n_levels = len(path)
    fig, axes = plt.subplots(1, n_levels, figsize=(4*n_levels, 4), constrained_layout=True)

    # Keep track of all far-field interactions already handled
    handled: List[QuadNode] = []

    for ax, node in zip(axes, path):
        level = node.level

        # a) Draw grid exactly at this level, skipping subdivisions in greyed regions
        draw_grid_at_level(ax, qt.root, level, handled)

        # b) Grey out all previously handled interactions at their native resolution
        for inter in handled:
            ax.add_patch(Rectangle(
                (inter.x_min, inter.y_min),
                inter.x_max - inter.x_min,
                inter.y_max - inter.y_min,
                facecolor="grey", alpha=0.4, linewidth=0, zorder=1
            ))

        # c) Highlight the current node in green
        ax.add_patch(Rectangle(
            (node.x_min, node.y_min),
            node.x_max - node.x_min,
            node.y_max - node.y_min,
            facecolor="green", alpha=0.4, zorder=2
        ))

        # d) Highlight neighbors in red
        for nb in node.neighbors:
            ax.add_patch(Rectangle(
                (nb.x_min, nb.y_min),
                nb.x_max - nb.x_min,
                nb.y_max - nb.y_min,
                facecolor="red", alpha=0.4, zorder=2
            ))

        # e) Highlight new interaction-list cells in blue, then mark them handled
        for inter in node.interaction_list:
            ax.add_patch(Rectangle(
                (inter.x_min, inter.y_min),
                inter.x_max - inter.x_min,
                inter.y_max - inter.y_min,
                facecolor="blue", alpha=0.3, zorder=2
            ))
        handled.extend(node.interaction_list)

        # Increase title font size for clarity
        ax.set_title(f"Level {level}", fontsize=24)
        ax.set_aspect("equal")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.axis("off")
    #plt.savefig('figures/FMMinteractionlist.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
