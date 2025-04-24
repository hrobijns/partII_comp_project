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
        # If leaf and under threshold, store here
        if self.is_leaf():
            if len(self.particles) < self.threshold:
                self.particles.append(particle)
                return
            # Otherwise subdivide
            self.subdivide()
            # Re-insert existing particles
            for p in self.particles:
                self._insert_into_children(p)
            self.particles = []
        # Insert new particle into appropriate child
        self._insert_into_children(particle)

    def _insert_into_children(self, particle):
        x, y = particle.pos
        x_min, x_max, y_min, y_max = self.boundary
        x_mid, y_mid = self.center
        # Determine quadrant
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
        # NW
        nw = (x_min, x_mid, y_mid, y_max)
        # NE
        ne = (x_mid, x_max, y_mid, y_max)
        # SW
        sw = (x_min, x_mid, y_min, y_mid)
        # SE
        se = (x_mid, x_max, y_min, y_mid)
        self.children = [
            QuadtreeNode(nw, self.threshold, parent=self),
            QuadtreeNode(ne, self.threshold, parent=self),
            QuadtreeNode(sw, self.threshold, parent=self),
            QuadtreeNode(se, self.threshold, parent=self)
        ]

    @property
    def interaction_set(self):
        return self._interaction_set


def build_tree(particles, tree_thresh, bbox=None, boundary='wall'):
    # Determine bounding box if not provided
    if bbox:
        x_min, x_max, y_min, y_max = bbox
    else:
        xs = [p.x for p in particles]
        ys = [p.y for p in particles]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # make square
        dx = x_max - x_min
        dy = y_max - y_min
        if dx > dy:
            y_max = y_min + dx
        else:
            x_max = x_min + dy
        # pad slightly
        pad = 1e-6 * max(dx, dy)
        x_min -= pad; x_max += pad; y_min -= pad; y_max += pad
    # Build
    root = QuadtreeNode((x_min, x_max, y_min, y_max), tree_thresh)
    for p in particles:
        root.insert(p)
    # Assign neighbors and interaction sets
    _assign_neighbors(root)
    return type('Tree', (), {'root': root})()


def _assign_neighbors(root):
    # Collect nodes by level
    levels = {}
    queue = [(root, 0)]
    while queue:
        node, lvl = queue.pop(0)
        if lvl not in levels:
            levels[lvl] = []
        levels[lvl].append(node)
        for c in node.children:
            queue.append((c, lvl+1))
    # Compute nearest neighbors for each level
    for lvl, nodes in levels.items():
        if lvl == 0:
            root.nearest_neighbors = []
            continue
        for node in nodes:
            nn = []
            x1_min, x1_max, y1_min, y1_max = node.boundary
            for other in nodes:
                if other is node:
                    continue
                x2_min, x2_max, y2_min, y2_max = other.boundary
                # check intersection or touching
                if not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min):
                    nn.append(other)
            node.nearest_neighbors = nn
    # Compute interaction sets for levels >= 1
    for lvl, nodes in levels.items():
        if lvl == 0:
            continue
        for node in nodes:
            iset = []
            P = node.parent
            if P:
                for pnb in P.nearest_neighbors:
                    for child in pnb.children:
                        if child not in node.nearest_neighbors and child is not node:
                            iset.append(child)
            node._interaction_set = iset