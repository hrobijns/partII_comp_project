import numpy as np

# --- physical / numerical constants ---
k     = 1.0    # Coulomb constant
soft  = 1.0    # softening length (same units as positions)
dt    = 0.01   # time step
theta = 0.5    # Barnes-Hut opening angle

# --- QuadTree for Coulombic BH ---
class QuadTree:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.center_of_charge = np.zeros(2)
        self.total_charge     = 0.0
        self.bodies           = []
        self.children         = None

    def contains(self, pos):
        return (self.x_min <= pos[0] < self.x_max and
                self.y_min <= pos[1] < self.y_max)

    def insert(self, body):
        # empty leaf?
        if self.total_charge == 0 and not self.bodies:
            self.bodies.append(body)
            self.center_of_charge = body.position.copy()
            self.total_charge     = body.charge
            return

        # subdivide if needed
        if self.children is None:
            self._subdivide()

        # update tot. charge & centroid:
        old_q = self.total_charge
        self.total_charge += body.charge
        self.center_of_charge = (
            old_q * self.center_of_charge
            + body.charge * body.position
        ) / self.total_charge

        # pass body to correct child
        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return

    def _subdivide(self):
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        self.children = [
            QuadTree(self.x_min, xm,    self.y_min, ym),
            QuadTree(xm,     self.x_max, self.y_min, ym),
            QuadTree(self.x_min, xm,    ym,         self.y_max),
            QuadTree(xm,     self.x_max, ym,         self.y_max),
        ]
        # re-insert any existing body
        for b in self.bodies:
            for c in self.children:
                if c.contains(b.position):
                    c.insert(b)
                    break
        self.bodies = []

    def compute_force(self, body, theta):
        # no charge here, or it's the same single body → zero
        if self.total_charge == 0 or (len(self.bodies)==1 and self.bodies[0] is body):
            return np.zeros(2)

        # vector from subject to this node's CoC
        diff = body.position - self.center_of_charge 
        r2   = diff.dot(diff)
        r    = np.sqrt(r2 + soft**2)
        width = self.x_max - self.x_min

        # BH criterion: if small enough, use this node as one charge
        if width/r < theta or self.children is None:
            # mirror the brute-force: f = k q₁ q₂ diff / (r_soft²)
            return k * body.charge * self.total_charge * diff / (r**2)
        else:
            # otherwise, recurse into children
            F = np.zeros(2)
            for c in self.children:
                F += c.compute_force(body, theta)
            return F
