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


def build_tree(particles, tree_thresh, bbox=None, boundary='wall'):
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


def _assign_neighbors(root):
    # Root has no neighbors or interaction set
    root.nearest_neighbors = []
    root._interaction_set = []

    queue = deque([root])
    while queue:
        node = queue.popleft()
        # For each child, build its neighbor list from siblings + parent's neighbors
        for child in node.children:
            neigh = []
            # 1) siblings
            for sib in node.children:
                if sib is not child and _touching(sib.boundary, child.boundary):
                    neigh.append(sib)
            # 2) cousins (children of node's neighbors)
            for pnb in node.nearest_neighbors:
                for cousin in pnb.children:
                    if _touching(cousin.boundary, child.boundary):
                        neigh.append(cousin)
            child.nearest_neighbors = neigh

            # Interaction set = cousins that are *not* in neigh
            iset = []
            for pnb in node.nearest_neighbors:
                for cousin in pnb.children:
                    if cousin not in neigh and cousin is not child:
                        iset.append(cousin)
            child._interaction_set = iset

            queue.append(child)