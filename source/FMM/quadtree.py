from collections import deque
import numpy as np 

class QuadtreeNode:
    def __init__(self, boundary, threshold, parent=None):
        # boundary: (x_min, x_max, y_min, y_max)
        self.boundary = boundary
        self.threshold = threshold
        self.parent = parent
        self.particles = []
        self.children = []  # NW, NE, SW, SE
        self.nearest_neighbors = []
        self._interaction_set = []

    @property
    def center(self):
        x_min, x_max, y_min, y_max = self.boundary
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)

    def is_leaf(self):
        return len(self.children) == 0

    def get_points(self):
        return self.particles if self.is_leaf() else []

    def __iter__(self):
        return iter(self.children)

    def insert(self, particle):
        if self.is_leaf():
            if len(self.particles) < self.threshold:
                self.particles.append(particle)
                return
            # subdivide
            self.subdivide()
            for p in self.particles:
                self._insert_into_children(p)
            self.particles = []
        self._insert_into_children(particle)

    def _insert_into_children(self, particle):
        x, y = particle.pos
        x_min, x_max, y_min, y_max = self.boundary
        x_mid, y_mid = self.center

        if x <= x_mid:
            if y <= y_mid:
                idx = 2  # SW
            else:
                idx = 0  # NW
        else:
            if y <= y_mid:
                idx = 3  # SE
            else:
                idx = 1  # NE

        self.children[idx].insert(particle)

    def subdivide(self):
        x_min, x_max, y_min, y_max = self.boundary
        x_mid, y_mid = self.center
        # NW, NE, SW, SE
        self.children = [
            QuadtreeNode((x_min, x_mid, y_mid, y_max), self.threshold, parent=self),
            QuadtreeNode((x_mid, x_max, y_mid, y_max), self.threshold, parent=self),
            QuadtreeNode((x_min, x_mid, y_min, y_mid), self.threshold, parent=self),
            QuadtreeNode((x_mid, x_max, y_min, y_mid), self.threshold, parent=self),
        ]

    @property
    def interaction_set(self):
        return self._interaction_set


def build_tree(particles, tree_thresh, bbox=None):
    # determine bounding box
    if bbox:
        x_min, x_max, y_min, y_max = bbox
    else:
        xs = [p.x for p in particles]
        ys = [p.y for p in particles]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        dx, dy = x_max - x_min, y_max - y_min
        if dx > dy:
            y_max = y_min + dx
        else:
            x_max = x_min + dy
        pad = 1e-6 * max(dx, dy)
        x_min, x_max = x_min - pad, x_max + pad
        y_min, y_max = y_min - pad, y_max + pad

    root = QuadtreeNode((x_min, x_max, y_min, y_max), tree_thresh)
    for p in particles:
        root.insert(p)

    _assign_neighbors(root)
    return type('Tree', (), {'root': root})()


def _touching(b1, b2):
    x1_min, x1_max, y1_min, y1_max = b1
    x2_min, x2_max, y2_min, y2_max = b2
    return not (x1_max < x2_min or x2_max < x1_min or
                y1_max < y2_min or y2_max < y1_min)

def _shares_edge(b1, b2):
    """Return True if two boxes share an edge (not just a corner)."""
    x1_min, x1_max, y1_min, y1_max = b1
    x2_min, x2_max, y2_min, y2_max = b2

    # Check for x-aligned touching
    x_touch = (np.isclose(x1_max, x2_min) or np.isclose(x2_max, x1_min))
    x_overlap = not (y1_max <= y2_min or y2_max <= y1_min)

    # Check for y-aligned touching
    y_touch = (np.isclose(y1_max, y2_min) or np.isclose(y2_max, y1_min))
    y_overlap = not (x1_max <= x2_min or x2_max <= x1_min)

    return (x_touch and x_overlap) or (y_touch and y_overlap)


def _assign_neighbors(root):
    """Assign nearest neighbors and interaction sets in quadtree."""
    root.nearest_neighbors = []   # The root has no neighbors by default.
    root._interaction_set = []

    queue = deque([root])

    while queue:
        node = queue.popleft()

        if node.is_leaf():
            continue  # Nothing more to do at leaves

        # For each child of the node
        for child in node.children:
            child.nearest_neighbors = []
            child._interaction_set = []

            # 1. Siblings (direct neighbors)
            for sibling in node.children:
                if sibling is not child and _shares_edge(child.boundary, sibling.boundary):
                    child.nearest_neighbors.append(sibling)

            # 2. Cousins (parent's nearest neighbors' children)
            for pn in node.nearest_neighbors:
                if not pn.is_leaf():
                    for cousin in pn.children:
                        if _shares_edge(child.boundary, cousin.boundary):
                            child.nearest_neighbors.append(cousin)

            # 3. Interaction set = touching but not nearest neighbor
            for pn in node.nearest_neighbors:
                if not pn.is_leaf():
                    for cousin in pn.children:
                        if _touching(child.boundary, cousin.boundary) and cousin not in child.nearest_neighbors:
                            child._interaction_set.append(cousin)

            queue.append(child)