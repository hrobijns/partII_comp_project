import numpy as np

k = 1.0      # Coulomb constant (in appropriate simulation units)
soft = 1.0   # softening length
dt = 0.0001    # time step 
theta = 0.5  # theta

class QuadTree:
    """QuadTree node for Barnes-Hut algorithm in 2D."""
    def __init__(self, x_min, x_max, y_min, y_max):
        # spatial bounds of this node
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        # combined center-of-charge of contained bodies (2D vector)
        self.center_of_charge = np.zeros(2)
        # total charge of contained bodies
        self.total_charge = 0.0
        # Body instances it contains
        self.bodies = []
        # children it has, (if leaf, none)
        self.children = None

    def contains(self, pos):
        """
        Check whether a 2D position lies within this node's bounds.
        Returns True if x_min <= pos[0] < x_max and y_min <= pos[1] < y_max.
        """
        return (self.x_min <= pos[0] < self.x_max and
                self.y_min <= pos[1] < self.y_max)

    def insert(self, body):
        """
        Insert a charged Body into the tree:
        - If this node is an empty leaf, store the body here.
        - Otherwise, ensure children exist (subdivide if needed),
          update this node's total_charge and center_of_charge,
          and delegate insertion to the appropriate child.
        """
        # case 1: empty leaf node
        if self.total_charge == 0 and not self.bodies:
            self.bodies.append(body)
            self.center_of_charge = body.position.copy()
            self.total_charge = body.charge
            return

        # case 2: non-empty or internal node
        # subdivide this leaf if it hasn't been already
        if self.children is None:
            self._subdivide()

        # update this node's aggregated charge & center-of-charge
        old_q = self.total_charge
        self.total_charge += body.charge
        self.center_of_charge = (
            old_q * self.center_of_charge
            + body.charge * body.position
        ) / self.total_charge

        # delegate to one of the four children based on body position
        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return

    def _subdivide(self):
        """
        Split this node's region into four quadrants and create child nodes.
        Then re-insert any existing body into the appropriate quadrant.
        """
        # compute midpoints
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        # create four children: SW, SE, NW, NE quadrants
        self.children = [
            QuadTree(self.x_min, xm,    self.y_min, ym),  # SW
            QuadTree(xm,     self.x_max, self.y_min, ym),  # SE
            QuadTree(self.x_min, xm,    ym,         self.y_max),  # NW
            QuadTree(xm,     self.x_max, ym,         self.y_max),  # NE
        ]
        # re-insert the existing body (if any) into correct child
        for b in self.bodies:
            for c in self.children:
                if c.contains(b.position):
                    c.insert(b)
                    break
        # clear bodies list for this internal node
        self.bodies = []

    def compute_force(self, body, theta):
        """
        Recursively compute Coulomb force on 'body' from this node, 
        using BH theta criterion.

        Returns:
            np.ndarray: 2D force vector on 'body'.
        """
        # empty region or same body -> no force
        if self.total_charge == 0 or (len(self.bodies) == 1 and self.bodies[0] is body):
            return np.zeros(2)

        diff = body.position - self.center_of_charge
        r2 = diff.dot(diff)
        r = np.sqrt(r2 + soft**2) # softening
        width = self.x_max - self.x_min

        # BH criterion
        if width / r < theta or self.children is None:
            return k * body.charge * self.total_charge * diff / (r**2)
        else:
            # otherwise, sum forces from each child cell
            F = np.zeros(2)
            for c in self.children:
                F += c.compute_force(body, theta)
            return F
