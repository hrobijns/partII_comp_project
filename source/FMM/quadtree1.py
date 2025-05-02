import math
from typing import List, Tuple, Optional, Dict, Any

class QuadNode:
    """A single node in a quadtree, with (i,j) grid coords on its level."""
    __slots__ = (
        'x_min','y_min','x_max','y_max',
        'level','parent','points','children',
        'neighbors','interaction_list',
        'outer','inner','i','j'
    )

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
        self.neighbors: List[QuadNode] = []
        self.interaction_list: List[QuadNode] = []
        self.outer = None   # multipole expansion
        self.inner = None   # local (Taylor) expansion
        # integer grid coordinates on this level:
        self.i: Optional[int] = None
        self.j: Optional[int] = None

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
        """Split this node into 4 children (no redistribution of bodies)."""
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        lvl = self.level + 1
        self.children = [
            QuadNode(self.x_min, self.y_min, xm,     ym,     lvl, self),
            QuadNode(xm,         self.y_min, self.x_max, ym,     lvl, self),
            QuadNode(self.x_min, ym,         xm,     self.y_max, lvl, self),
            QuadNode(xm,         ym,         self.x_max, self.y_max, lvl, self),
        ]

class QuadTree:
    """Uniform quadtree with fast (i,j)-indexed neighbor & interaction lists."""
    def __init__(
        self,
        bodies: List[Any],                       # must have .position = (x,y)
        boundary: Tuple[float, float, float, float],
        max_per_leaf: int = 5
    ):
        x_min, y_min, x_max, y_max = boundary
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max

        # 1) build root & choose uniform depth
        self.root = QuadNode(x_min, y_min, x_max, y_max)
        n = len(bodies)
        depth = 0
        while n / (4 ** depth) > max_per_leaf:
            depth += 1
        self.depth = depth

        # 2) uniformly subdivide all the way down
        self._uniform_subdivide(self.root, depth)

        # 3) collect nodes by level & assign (i,j) coordinates
        self.level_nodes = self._collect_by_level()
        self.level_maps: Dict[int, Dict[Tuple[int,int], QuadNode]] = {}
        for lvl, nodes in self.level_nodes.items():
            delta = (x_max - x_min) / (2 ** lvl)
            grid: Dict[Tuple[int,int], QuadNode] = {}
            for node in nodes:
                i = int(math.floor((node.x_min - x_min) / delta))
                j = int(math.floor((node.y_min - y_min) / delta))
                node.i, node.j = i, j
                grid[(i, j)] = node
            self.level_maps[lvl] = grid

        # 4) distribute bodies into leaves
        for b in bodies:
            leaf = self._find_leaf(self.root, b.position)
            leaf.points.append(b)

        # 5) build neighbors & interactions in linear time
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
        raise ValueError(f"Position {pos} not found in quadtree")

    def _collect_by_level(self) -> Dict[int, List[QuadNode]]:
        levels: Dict[int, List[QuadNode]] = {}
        def _rec(n: QuadNode):
            levels.setdefault(n.level, []).append(n)
            for c in n.children:
                _rec(c)
        _rec(self.root)
        return levels

    def _compute_neighbors(self) -> None:
        """For each node at each level, look up its 8 grid neighbors in O(1)."""
        for lvl, nodes in self.level_nodes.items():
            grid = self.level_maps[lvl]
            for node in nodes:
                node.neighbors.clear()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        nbr = grid.get((node.i + di, node.j + dj))
                        if nbr:
                            node.neighbors.append(nbr)

    def _compute_interactions(self) -> None:
        """For each node, the interaction list is all children of its
           parent’s neighbors that are not direct neighbors of itself."""
        # flatten all nodes:
        all_nodes = [n for lvl in self.level_nodes for n in self.level_nodes[lvl]]
        for node in all_nodes:
            node.interaction_list.clear()
            if node.parent is None:
                continue
            for pn in node.parent.neighbors:
                # pn.children live exactly one level below pn, which matches node.level
                for c in pn.children:
                    if c is not node and c not in node.neighbors:
                        node.interaction_list.append(c)