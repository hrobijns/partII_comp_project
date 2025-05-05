from typing import List, Tuple, Optional, Dict, Any


class QuadNode:
    """A single node in a quadtree."""
    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        level: int = 0,
        parent: Optional["QuadNode"] = None
    ):
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.level    = level
        self.parent   = parent    # None for root
        self.points   = []        # bodies in this cell
        self.children: List[QuadNode] = []
        self.neighbors: List[QuadNode] = []        # same-level adjacent cells
        self.interaction_list: List[QuadNode] = [] # same-level non-adjacent cells
        self.outer = None   # multipole expansion
        self.inner = None   # local (Taylor) expansion
        self.fx = 0.0
        self.fy = 0.0

    def is_leaf(self) -> bool:
        return not self.children

    def contains(self, pos: Tuple[float, float]) -> bool:
        x, y = pos
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) * 0.5,
                (self.y_min + self.y_max) * 0.5)

    def subdivide(self) -> None:
        """Split this node into 4 children."""
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        self.children = [
            QuadNode(self.x_min, self.y_min, xm,     ym,     self.level+1, self),
            QuadNode(xm,         self.y_min, self.x_max, ym,     self.level+1, self),
            QuadNode(self.x_min, ym,         xm,     self.y_max, self.level+1, self),
            QuadNode(xm,         ym,         self.x_max, self.y_max, self.level+1, self),
        ]
        # no redistribution here (we distribute after uniform build)


class QuadTree:
    """Fully symmetric quadtree with uniform leaf size based on max points per leaf."""
    def __init__(
        self,
        bodies: List[Any],
        boundary: Tuple[float, float, float, float],
        max_per_leaf: int = 5
    ):
        x_min, y_min, x_max, y_max = boundary
        self.root = QuadNode(x_min, y_min, x_max, y_max)
        self.bodies = bodies
        n = len(bodies)
        # determine uniform subdivision depth
        depth = 0
        while n / (4 ** depth) > max_per_leaf:
            depth += 1
        self.depth = depth

        # uniformly subdivide every node to that depth
        self._uniform_subdivide(self.root, depth)

        # distribute bodies into leaves
        for b in bodies:
            leaf = self._find_leaf(self.root, b.position)
            leaf.points.append(b)

        # compute neighbor & interaction lists
        self._compute_neighbors()
        self._compute_interactions()

    def _uniform_subdivide(self, node: QuadNode, target_depth: int) -> None:
        if node.level >= target_depth:
            return
        node.subdivide()
        for c in node.children:
            self._uniform_subdivide(c, target_depth)

    def _find_leaf(self, node: QuadNode, pos: Tuple[float, float]) -> QuadNode:
        if node.is_leaf():
            return node
        for c in node.children:
            if c.contains(pos):
                return self._find_leaf(c, pos)
        raise ValueError(f"Position {pos} not found in any leaf")

    def _collect_by_level(self) -> Dict[int, List[QuadNode]]:
        levels: Dict[int, List[QuadNode]] = {}
        def _rec(n: QuadNode):
            levels.setdefault(n.level, []).append(n)
            for c in n.children:
                _rec(c)
        _rec(self.root)
        return levels

    @staticmethod
    def _is_neighbor(a: QuadNode, b: QuadNode) -> bool:
        return not (
            a.x_max  < b.x_min or
            a.x_min  > b.x_max or
            a.y_max  < b.y_min or
            a.y_min  > b.y_max
        )

    def _compute_neighbors(self) -> None:
        levels = self._collect_by_level()
        for nodes in levels.values():
            for node in nodes:
                node.neighbors.clear()
                for other in nodes:
                    if other is not node and self._is_neighbor(node, other):
                        node.neighbors.append(other)

    def _descendants_at_level(self, node: QuadNode, target_level: int) -> List[QuadNode]:
        if node.level == target_level or node.is_leaf():
            return [node]
        result: List[QuadNode] = []
        for c in node.children:
            result.extend(self._descendants_at_level(c, target_level))
        return result

    def _compute_interactions(self) -> None:
        all_nodes: List[QuadNode] = []
        def _rec(n: QuadNode):
            for c in n.children:
                all_nodes.append(c)
                _rec(c)
        _rec(self.root)

        for node in all_nodes:
            node.interaction_list.clear()
            parent = node.parent
            if parent is None:
                continue
            tgt_lvl = node.level
            for pn in parent.neighbors:
                candidates = self._descendants_at_level(pn, tgt_lvl)
                for c in candidates:
                    if c is not node and not self._is_neighbor(node, c):
                        node.interaction_list.append(c)