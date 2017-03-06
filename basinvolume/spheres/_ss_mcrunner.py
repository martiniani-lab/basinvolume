from __future__ import print_function
import numpy as np
from pele.distance import Distance
from mcpele.monte_carlo import _BaseMCRunner
from basinvolume.monte_carlo import RecordDisp2Histogram
from basinvolume.utils import reduce_coordinates

class BaseSpheresMCrunner(_BaseMCRunner):
    """
    this class sets all the basic parameters for a soft spheres mcrunner class
    and basic functions.
    If you want the particle to be all soft set hs_radii = radii and sca=(ss/hs-1)=0
    """
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 hs_radii, boxv, sca, rattlers=None, k=1.0, dtol=1e-3, eps=1.,
                 hmin=0, hmax=1, hbinsize=0.001, report_steps=0,
                 pt_eq_niter=0, seeds=None, use_cell_lists=True,
                 record_histogram=False, distance_method=Distance.PERIODIC, use_frozen=False,
                 frozen_atoms=None, rcontainer=None):
        # construct base class
        if use_frozen:
            assert distance_method is Distance.CARTESIAN and frozen_atoms is not None
            red_coords = reduce_coordinates(full_coords, frozen_atoms, len(boxv))
        else:
            red_coords = full_coords
        super(BaseSpheresMCrunner, self).__init__(potential, red_coords, temperature, niter)

        self.boxv = boxv
        self.bdim = len(boxv)
        self.origin = np.array(origin)
        self.red_origin = np.array(origin)
        self.hs_radii = np.array(hs_radii)
        self.red_radii = np.array(hs_radii)
        if use_frozen:
            self.red_radii = np.delete(self.red_radii, frozen_atoms)
            self.red_origin = reduce_coordinates(self.red_origin, frozen_atoms, self.bdim)
            assert len(self.red_radii) == (len(self.hs_radii) - len(frozen_atoms))
            assert len(self.red_origin) == self.ndim
            assert rcontainer is not None
        self.sca = sca
        self.dtol = dtol
        self.eps = eps
        self.nparticles = len(self.red_radii)
        self.use_cell_lists = use_cell_lists
        self.use_frozen = use_frozen
        self.frozen_atoms = frozen_atoms
        self.distance_method = distance_method
        self.rcontainer = rcontainer
        self.equilibration_steps = report_steps + pt_eq_niter
        self.hmin = hmin
        self.hmax = hmax
        self.binsize = hbinsize

        #manage array of rattlers, if not rattler: 1 -> jammed dof
        #                                          0 -> rattler dof
        if (rattlers is None):
            self.rattlers = np.array([1. for _ in xrange(self.ndim)], dtype='d')
        else:
            self.rattlers = np.array(rattlers, dtype='d')
        if self.use_frozen:
            self.rattlers = reduce_coordinates(self.rattlers, frozen_atoms, self.bdim)
        assert(len(self.rattlers) == self.ndim)
        assert(self.rattlers.all() >= 0 and self.rattlers.all() <= 1)

        #rcut set to largest particle diameter
        self.rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca)
        if self.use_cell_lists:
            if np.amin(self.boxv) // self.rcut <= 3:
                print ("warning: use_cell_lists flag was set, rcut is too large though")
                print ("setting use_cell_lists to False")
                self.use_cell_lists = False
        self.ncellx_scale = 1.0

        # compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds

        # get potential and minimizer
        self.pot_optimizer = self.get_pot_optimizer()
        self.optimizer = self.get_optimizer()

        # construct base test/action classes
        if record_histogram:
            self._set_record_histogram(self.hmin, self.hmax, self.binsize)

        # construct custom test/action/takestep classes
        self.set_report_steps(report_steps)
        self._set_conf_tests()
        self._set_takestep(stepsize)
        self._set_accept_tests()
        self._set_actions()

    def _set_record_histogram(self, hmin, hmax, binsize):
        self.histogram = RecordDisp2Histogram(self.red_origin, self.rattlers, self.bdim, hmin, hmax,
                                              binsize, self.equilibration_steps)
        self.add_action(self.histogram)

    def _set_takestep(self, stepsize):
        raise NotImplementedError

    def _set_actions(self):
        raise NotImplementedError

    def _set_accept_tests(self):
        raise NotImplementedError

    def _set_conf_tests(self):
        raise NotImplementedError

    def get_pot_optimizer(self):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError

    def _get_check_same_minimum(self):
        raise NotImplementedError

    def set_control(self, c):
        raise NotImplementedError

    def get_stepsize(self):
        return self.takestep.get_stepsize()

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super(BaseSpheresMCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status
