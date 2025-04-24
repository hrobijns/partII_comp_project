import numpy as np
from scipy.special import binom
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 10**(-30)
# ---- Data Structures ----

class Point():
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.pos = (x, y)


class Body(Point):
    def __init__(self, x, y, mass, velocity=None):
        super().__init__(x, y)
        self.m = mass
        self.phi = 0
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0])


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ---- Quadtree Implementation ----

class Node():
    DISREGARD = (1, 2, 0, 3)
    CORNER_CHILDREN = (3, 2, 0, 1)

    def __init__(self, width, height, x0, y0, points=None, children=None, parent=None, level=0):
        self._points = []
        self._children = children
        self._cneighbors = 4 * [None]
        self._nneighbors = None
        self._cindex = 0
        self.parent = parent
        self.x0, self.y0, self.w, self.h = x0, y0, width, height
        self.verts = ((x0, x0 + width), (y0, y0 + height))
        self.center = (x0 + width / 2, y0 + height / 2)
        self.level = level
        self.inner = None
        self.outer = None
        if points is not None:
            self.add_points(points)

    def __iter__(self):
        if self._has_children():
            for child in self._children:
                yield child

    def __len__(self):
        return len(self._points) if self._points is not None else 0

    def _has_children(self):
        return self._children is not None

    def _get_child(self, i):
        return self._children[i] if self._children else self

    def _split(self):
        if self._has_children(): return
        w, h = self.w / 2, self.h / 2
        x0, y0 = self.verts[0][0], self.verts[1][0]
        self._children = [
            Node(w, h, xi, yi, points=self._points, level=self.level + 1, parent=self)
            for yi in (y0 + h, y0) for xi in (x0, x0 + w)
        ]
        for i, c in enumerate(self._children):
            c._cindex = i

    def _contains(self, x, y):
        return self.verts[0][0] <= x < self.verts[0][1] and self.verts[1][0] <= y < self.verts[1][1]

    def is_leaf(self):
        return not self._has_children()

    def thresh_split(self, thresh):
        if len(self) > thresh:
            self._split()
        if self._has_children():
            for child in self._children:
                child.thresh_split(thresh)

    def set_cneighbors(self):
        if not self._has_children():
            return  # Prevent trying to iterate over None

        for i, child in enumerate(self._children):
            sn = (abs(1 + (i ^ 1) - i), abs(1 + (i ^ 2) - i))
            child._cneighbors[sn[0]] = self._children[i ^ 1]
            child._cneighbors[sn[1]] = self._children[i ^ 2]
            pn = tuple(set((0, 1, 2, 3)) - set(sn))
            nc = lambda j, k: j ^ ((k + 1) % 2 + 1)
            for idx in pn:
                cn = self._cneighbors[idx]
                child._cneighbors[idx] = cn._get_child(nc(i, pn[1])) if cn else None
            child.set_cneighbors()  # Recursive call on child

    def add_points(self, points):
        if self._has_children():
            for child in self._children:
                filtered = [p for p in points if child._contains(p.x, p.y)]
                child.add_points(filtered)
        else:
            for p in points:
                if self._contains(p.x, p.y):
                    self._points.append(p)
                    p.node = self

    def get_points(self):
        return self._points

    def traverse(self):
        for child in self._children if self._has_children() else []:
            yield from child.traverse()
        yield self

    @property
    def nearest_neighbors(self):
        if self._nneighbors is not None:
            return self._nneighbors
        nn = [cn._cneighbors[(i + 1) % 4] for i, cn in enumerate(self._cneighbors)
              if cn and cn.level == self.level]
        nn += [cn._cneighbors[(i + 1) % 4]._get_child(self.CORNER_CHILDREN[i])
               for i, cn in enumerate(self._cneighbors)
               if cn and cn._cneighbors[(i + 1) % 4] and cn.level < self.level and
               i != self.DISREGARD[self._cindex]]
        self._nneighbors = [n for n in self._cneighbors + nn if n]
        return self._nneighbors

    def interaction_set(self):
        return [c for n in self.parent.nearest_neighbors for c in (n if n._has_children() else [n])
                if c not in self.nearest_neighbors]


class QuadTree():
    def __init__(self, points, thresh, bbox=(1, 1), boundary='wall'):
        self.threshold = thresh
        self.root = Node(*bbox, points=points, level=0, parent=None)
        self.root._cneighbors = 4 * [self.root if boundary == 'periodic' else None]
        self.root.add_points(points)
        self.root.thresh_split(thresh)
        self.root.set_cneighbors()

    def __len__(self):
        return sum(len(node) for node in self.root.traverse())

    @property
    def depth(self):
        return max(node.level for node in self.root.traverse())

    def traverse_nodes(self):
        yield from self.root.traverse()


def build_tree(points, tree_thresh=None, bbox=None, boundary='wall'):
    if bbox is None:
        coords = np.array([(p.x, p.y) for p in points])
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)

        padding = 0.1
        width = max_x - min_x + 2 * padding
        height = max_y - min_y + 2 * padding
        x0 = min_x - padding
        y0 = min_y - padding
        bbox = (width, height, x0, y0)

    if tree_thresh is None:
        tree_thresh = 5

    return QuadTree(points, tree_thresh, bbox=bbox, boundary=boundary)

# ---- FMM Core ----

def multipole(bodies, center=(0, 0), nterms=5):
    coeffs = np.empty(nterms + 1, dtype=complex)
    coeffs[0] = sum(b.m for b in bodies)
    coeffs[1:] = [sum([-b.m * complex(b.x - center[0], b.y - center[1]) ** k / k for b in bodies])
                  for k in range(1, nterms + 1)]
    return coeffs


def _shift_mpexp(coeffs, z0):
    shift = np.empty_like(coeffs)
    shift[0] = coeffs[0]
    shift[1:] = [sum([coeffs[k] * z0 ** (l - k) * binom(l - 1, k - 1) - coeffs[0] * z0 ** l / l
                      for k in range(1, l)]) for l in range(1, len(coeffs))]
    return shift


def _outer_mpexp(tnode, nterms):
    if tnode.is_leaf():
        tnode.outer = multipole(tnode.get_points(), center=tnode.center, nterms=nterms)
    else:
        tnode.outer = np.zeros(nterms + 1, dtype=complex)
        for child in tnode:
            _outer_mpexp(child, nterms)
            z0 = complex(*child.center) - complex(*tnode.center)
            tnode.outer += _shift_mpexp(child.outer, z0)


def _convert_oi(coeffs, z0):
    inner = np.empty_like(coeffs)
    inner[0] = sum((coeffs[k] / z0 ** k) * (-1) ** k for k in range(1, len(coeffs))) + coeffs[0] * np.log(abs(z0) + 1e-2)    
    inner[1:] = [(1 / z0 ** l) * sum((coeffs[k] / z0 ** k) * binom(l + k - 1, k - 1) * (-1) ** k
                                      for k in range(1, len(coeffs))) - coeffs[0] / (z0 ** l * l)
                 for l in range(1, len(coeffs))]
    return inner


def _shift_texp(coeffs, z0):
    return [sum(coeffs[k] * binom(k, l) * (-z0) ** (k - l)
                for k in range(l, len(coeffs)))
            for l in range(len(coeffs))]


def _inner(tnode):
    z0 = complex(*tnode.parent.center) - complex(*tnode.center)
    tnode.inner = _shift_texp(tnode.parent.inner, z0)
    for tin in tnode.interaction_set():
        z0 = complex(*tin.center) - complex(*tnode.center)
        tnode.inner += _convert_oi(tin.outer, z0)
    if tnode.is_leaf():
        z0 = complex(*tnode.center)
        for p in tnode.get_points():
            z = complex(*p.pos)
            p.phi -= np.real(np.polyval(tnode.inner[::-1], z - z0))
        for nn in tnode.nearest_neighbors:
            potentialDDS(tnode.get_points(), nn.get_points())
        _ = potentialDS(tnode.get_points())
    else:
        for child in tnode:
            _inner(child)

def gradient_at_point(point, tnode):
    z = complex(*point.pos)
    z0 = complex(*tnode.center)
    dz = z - z0
    grad_phi = sum(
        k * tnode.inner[k] * dz ** (k - 1)
        for k in range(1, len(tnode.inner))
    )
    # Return negative gradient: ∇φ = -E
    return -np.array([grad_phi.real, grad_phi.imag])

def potential(bodies, bbox=None, tree_thresh=None, nterms=5, boundary='wall'):
    tree = build_tree(bodies, tree_thresh, bbox=bbox, boundary=boundary)

    _outer_mpexp(tree.root, nterms)
    
    # Use root's center to avoid zero division
    z0 = complex(*tree.root.center)
    tree.root.inner = _convert_oi(tree.root.outer, z0)
    
    if tree.root.is_leaf():
        _inner(tree.root)
    else:
        for child in tree.root:
            _inner(child)
    
    return tree


def potentialDDS(bodies, sources):
    for body in bodies:
        for source in sources:
            r = distance(body.pos, source.pos)
            body.phi -= body.m * np.log((r**2+1e-2)**(1/2))


def potentialDS(bodies):
    phi = np.zeros(len(bodies))
    for i, b in enumerate(bodies):
        for s in (bodies[:i] + bodies[i + 1:]):
            r = distance(b.pos, s.pos)
            b.phi -= b.m * np.log((r**2+1e-2)**(1/2))
        phi[i] = b.phi
    return phi

class Simulation:
    def __init__(self, bodies, dt=0.01, nterms =5):
        self.bodies = bodies
        self.dt = dt
        self.nterms = nterms

        for b in self.bodies:
            if not hasattr(b, "velocity"):
                b.velocity = np.zeros(2)
            if not hasattr(b, "accel"):
                b.accel = np.zeros(2)

        self.compute_forces()  # Initial accel
        print(f"[init] Sample accel: {self.bodies[0].accel}")
        print(f"[init] Sample velocity after init: {self.bodies[0].velocity}")

        # Initial half-step for leapfrog
        for b in self.bodies:
            b.velocity -= 0.5 * self.dt * b.accel

    def compute_forces(self):
        for b in self.bodies:
            b.phi = 0

        self.tree = potential(self.bodies, nterms=self.nterms)

        # Make sure each body has an updated node after tree is built
        for node in self.tree.traverse_nodes():
            for b in node.get_points():
                b.node = node

        for b in self.bodies:
            node = getattr(b, "node", None)
            grad = gradient_at_point(b, node)
            b.accel = grad

    def move(self):
        dt = self.dt
        # Position update
        for b in self.bodies:
            b.x += b.velocity[0] * dt
            b.y += b.velocity[1] * dt
            b.pos = (b.x, b.y)
        # Recompute acceleration
        self.compute_forces()

        # Velocity full-step update
        for b in self.bodies:
            b.velocity += b.accel * dt

def phi_at_point(pos, bodies):
    phi = 0
    for b in bodies:
        r = distance(pos, b.pos) + 1e-5
        phi -= G * b.m * np.log((r**2+1e-2)**(1/2))
    return phi

class Animation:
    def __init__(self, bodies, simulation):
        self.bodies = bodies
        self.sim = simulation

        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter(
            [b.x for b in bodies],
            [b.y for b in bodies],
            s=[abs(b.m) * 5 for b in bodies]
        )
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=50)

    def update(self, frame):
        self.sim.move()
        positions = np.array([[b.x, b.y] for b in self.bodies])
        self.scat.set_offsets(positions)
        return self.scat

    def show(self):
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    init_velocity = [0.0, 0.0]  # customize here
    nterms = 7  # customize here

    bodies = [
        Body(
            x=np.random.uniform(-2, 2),
            y=np.random.uniform(-2, 2),
            mass=np.random.uniform(-1, 1),
            velocity=init_velocity
        )
        for _ in range(10)
    ]
    bodies.append(
    Body(0.0, 0.0, mass=10.0, velocity=[0.0, 0.0])
)

    simulation = Simulation(bodies, nterms=nterms)
    anim = Animation(bodies, simulation)
    anim.show()