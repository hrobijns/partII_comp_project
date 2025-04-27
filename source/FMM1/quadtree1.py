class QuadNode:
    """
    A node in the quadtree, representing an axis-aligned square region.
    """
    def __init__(self, x_min, y_min, x_max, y_max, level=0, parent=None):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.level = level
        self.parent = parent
        self.points = []
        self.children = []
        self.neighbors = []          # Near-field interaction (adjacent cells)
        self.interaction_list = []   # Far-field interaction (well-separated cells)

    def contains(self, point):
        """Check if the point (x, y) lies within this node's region."""
        x, y = point
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def is_leaf(self):
        """A node is a leaf if it has no children."""
        return len(self.children) == 0

    def subdivide(self):
        """Split this node into 4 equal quadrants and reassign points."""
        x_mid = 0.5 * (self.x_min + self.x_max)
        y_mid = 0.5 * (self.y_min + self.y_max)
        # Order: NW, NE, SW, SE
        self.children = [
            QuadNode(self.x_min, y_mid, x_mid, self.y_max, self.level+1, self),
            QuadNode(x_mid,    y_mid, self.x_max, self.y_max, self.level+1, self),
            QuadNode(self.x_min, self.y_min, x_mid,    y_mid, self.level+1, self),
            QuadNode(x_mid,    self.y_min, self.x_max, y_mid,    self.level+1, self),
        ]
        # Re-distribute stored points into children
        for p in self.points:
            for child in self.children:
                if child.contains(p):
                    child.points.append(p)
                    break
        self.points.clear()

    @property
    def center(self):
        return ((self.x_min + self.x_max)*0.5,
                (self.y_min + self.y_max)*0.5)

    def get_points(self):
        return self.points

    def __iter__(self):
        return iter(self.children)

class Quadtree:
    """
    A quadtree for 2D FMM. Builds the tree, computes near-field neighbors
    and far-field interaction lists for each node.
    """
    def __init__(self, points, boundary, max_points=10, max_level=5):
        # boundary = (x_min, y_min, x_max, y_max)
        self.root = QuadNode(*boundary)
        self.max_points = max_points
        self.max_level = max_level
        # Insert all points
        for p in points:
            self._insert(self.root, p)
        # Build neighbor lists and interaction lists
        self._compute_neighbors()
        self._compute_interactions()

    def _insert(self, node, point):
        """Insert a point into the quadtree, subdividing as needed."""
        if not node.contains(point):
            return False
        # If leaf and capacity not reached, or at max depth
        if node.is_leaf():
            if len(node.points) < self.max_points or node.level >= self.max_level:
                node.points.append(point)
                return True
            else:
                node.subdivide()
        # Otherwise, forward to children
        for child in node.children:
            if self._insert(child, point):
                return True
        return False

    def get_leaves(self):
        """Return a list of all leaf nodes."""
        leaves = []
        def _collect(n):
            if n.is_leaf():
                leaves.append(n)
            else:
                for c in n.children:
                    _collect(c)
        _collect(self.root)
        return leaves

    def _compute_neighbors(self):
        """Compute near-field neighbors for each node at each level."""
        # Group nodes by level
        levels = {}
        def _collect(n):
            levels.setdefault(n.level, []).append(n)
            for c in n.children:
                _collect(c)
        _collect(self.root)
        # For each level, check pairwise adjacency
        for lvl, nodes in levels.items():
            for node in nodes:
                node.neighbors.clear()
                for other in nodes:
                    if other is not node and self._is_neighbor(node, other):
                        node.neighbors.append(other)

    def _is_neighbor(self, a, b):
        """Two nodes are neighbors if their regions touch or overlap."""
        return not (
            a.x_max < b.x_min or a.x_min > b.x_max or
            a.y_max < b.y_min or a.y_min > b.y_max
        )

    def _compute_interactions(self):
        """Compute far-field interaction list for each non-root node."""
        # Collect all non-root nodes
        nodes = []
        def _collect(n):
            if n is not self.root:
                nodes.append(n)
            for c in n.children:
                _collect(c)
        _collect(self.root)

        for node in nodes:
            parent = node.parent
            # For each neighbor of the parent
            for pn in parent.neighbors:
                # Use children if subdivided, else the parent neighbor itself
                candidates = pn.children if pn.children else [pn]
                for c in candidates:
                    if c.level != node.level or c is node:
                        continue
                    # Add to interaction if not a near neighbor
                    if not self._is_neighbor(node, c):
                        node.interaction_list.append(c)
