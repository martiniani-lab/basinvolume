from __future__ import division
from __future__ import print_function
from builtins import object
import numpy as np
from pele.potentials import HS_WCA
from pele.optimize import LBFGS_CPP
from basinvolume.utils import volume_nball
from basinvolume.utils import simple_overlap_check
from pele.distance import Distance


class ThrowAndQuench(object):
    """
    Throws particles of defined HS radii into a box uniformly at random
    and quenches to find a legal initial configuration.
    """

    def __init__(self, nr_particles, hard_phi, radii, alpha=0.2, epsilon=1, seed=42):
        # 4 throw particles with these radii into a box uniformly and minimize with LBFGS
        print("throw and quench")
        self.nr_particles = nr_particles
        self.hard_phi = hard_phi
        self.radii = radii
        self.alpha = alpha
        self.epsilon = epsilon
        self.seed = seed
        #
        self.boxdim = 2
        self.box_length = np.power(
            np.sum(np.asarray([volume_nball(r, self.boxdim) for r in self.radii]))
            / self.hard_phi,
            1 / self.boxdim,
        )
        self.boxvec = np.ones(self.boxdim) * self.box_length
        self.find_legal_initial_condition()

    def find_legal_initial_condition(self):
        print("attampting to find legal initial condition")
        self.rcut = 2 * (1 + self.alpha) * np.amax(self.radii)
        self.potential = HS_WCA(
            distance_method=Distance.PERIODIC,
            use_cell_lists=True,
            ndim=self.boxdim,
            eps=self.epsilon,
            sca=self.alpha,
            radii=self.radii,
            boxvec=self.boxvec,
        )
        self.nr_dof = self.boxdim * self.nr_particles
        np.random.seed(self.seed)
        illegal = True
        iteration = 0
        while illegal:
            iteration += 1
            print(("iteration", iteration))
            illegal = self.sample_and_minimize()
        print("done")

    def sample_and_minimize(self):
        self.coordinates = np.random.uniform(
            -0.5 * self.box_length, 0.5 * self.box_length, self.nr_dof
        )
        optimizer = LBFGS_CPP(self.coordinates, self.potential)
        optimizer.run(1e7)
        if not optimizer.get_result().success:
            print("minimization failed")
            return True
        self.coordinates = optimizer.get_result().coords.copy()
        return simple_overlap_check(self.coordinates, self.radii, self.box_length)
