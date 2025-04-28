class QuadNode:
    """A single a node in a quadtree"""
    def __init__(self, x_min, y_min, x_max, y_max, level=0, parent=None):
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.level    = level
        self.parent   = parent # i.e None for root cell
        self.points   = []    # bodies in this cell
        self.children = []    # four sub‐cells
        self.neighbors        = []  # near‐field (adjacent) cells at same level
        self.interaction_list = []  # far‐field (non-adjacent) cells at same level
        self.outer = None   # holds multipole expansion
        self.inner = None   # holds local (Taylor) expansion    

    def is_leaf(self):
        return len(self.children) == 0 # boolean as to whether a cell is a leaf

    def contains(self, pos):
        x, y = pos
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)
        # boolean as to whether a position is within a cell

    @property
    def center(self):
        return ((self.x_min + self.x_max) * 0.5,
                (self.y_min + self.y_max) * 0.5)

    def subdivide(self):
        """Subdividing a node into its four children"""
        xm = 0.5*(self.x_min + self.x_max)
        ym = 0.5*(self.y_min + self.y_max)
        # create 4 children
        self.children = [
            QuadNode(self.x_min, self.y_min, xm,     ym,     self.level+1, self),
            QuadNode(xm,         self.y_min, self.x_max, ym,     self.level+1, self),
            QuadNode(self.x_min, ym,         xm,     self.y_max, self.level+1, self),
            QuadNode(xm,         ym,         self.x_max, self.y_max, self.level+1, self),
        ]
        # re‐distribute any existing points
        for p in self.points:
            for c in self.children:
                if c.contains(p.position):
                    c.points.append(p)
                    break
        self.points = []

    def insert(self, body, max_points):
        """Insert a body into a node"""
        if not self.contains(body.position):
            return False
        if self.is_leaf() and len(self.points) < max_points:
            self.points.append(body)
            return True
        if self.is_leaf():
            self.subdivide()
        for c in self.children:
            if c.insert(body, max_points):
                return True
        return False

    def __iter__(self):
        """Allows for iteration through children of the node"""
        return iter(self.children)

    def get_points(self):
        """Return all bodies in this node (including children)"""
        if self.is_leaf():
            return list(self.points)
        pts = []
        for c in self.children:
            pts.extend(c.get_points())
        return pts


class QuadTree:
    def __init__(self, bodies, boundary, max_points=1):
        """
        bodies     : iterable with position (2-tuple) and any data you like (e.g. charge)
        boundary   : (x_min, y_min, x_max, y_max)
        max_points : maximum bodies per leaf before subdivision
        """
        x_min, y_min, x_max, y_max = boundary
        self.root       = QuadNode(x_min, y_min, x_max, y_max)
        self.max_points = max_points

        # build the tree
        for b in bodies:
            inserted = self.root.insert(b, max_points)
            if not inserted:
                raise ValueError(f"Body at {b.position} failed to be placed in the tree")

        # then compute neighbor and interaction lists
        self._compute_neighbors()
        self._compute_interactions()

    def _collect_by_level(self):
        """Creates dictionary of nodes at each level"""
        levels = {}
        def _rec(node):
            levels.setdefault(node.level, []).append(node)
            for c in node.children:
                _rec(c)
        _rec(self.root)
        return levels

    def _is_neighbor(self, a, b):
        """True if 'a' and 'b' touch or overlap in 2D."""
        return not (
            a.x_max < b.x_min or a.x_min > b.x_max or
            a.y_max < b.y_min or a.y_min > b.y_max
        )

    def _compute_neighbors(self):
        """Populate each node's neighbors with same-level adjacent cells."""
        levels = self._collect_by_level()
        for lvl, nodes in levels.items():
            for node in nodes:
                node.neighbors.clear()
                for other in nodes:
                    if other is not node and self._is_neighbor(node, other):
                        node.neighbors.append(other)

    def _compute_interactions(self):
        """
        For each non-root node, build interaction list of far-field cells 
        (i.e. children of parent's neighbors that are not themselves direct neighbors).
        """
        all_nodes = []
        def _rec(node):
            if node is not self.root:
                all_nodes.append(node)
            for c in node.children:
                _rec(c)
        _rec(self.root)

        for node in all_nodes:
            node.interaction_list.clear()
            parent = node.parent
            for pn in parent.neighbors:
                candidates = pn.children if pn.children else [pn]
                for c in candidates:
                    if c.level != node.level or c is node:
                        continue
                    # only far‐field if they do _not_ touch
                    if not self._is_neighbor(node, c):
                        node.interaction_list.append(c)