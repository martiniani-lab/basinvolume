from __future__ import print_function
import numpy as np
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CheckSphericalContainer
from basinvolume.monte_carlo import RecordDisp2Histogram
from basinvolume.monte_carlo import CheckOverlapPeriodic, CheckOverlapCartesian
from basinvolume.monte_carlo import CheckOverlapCartesianCellLists
from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists
from basinvolume.utils import reduce_coordinates

class SpheresMCrunner(_BaseMCRunner):
    """
    this class sets all the basic parameters for a soft spheres mcrunner class
    and basic functions
    """
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 hs_radii, boxv, sca, rattlers=None, k=1.0, dtol=1e-3, eps=1.,
                 hmin=0, hmax=1, hbinsize=0.001, report_steps=0,
                 pt_eq_niter=0, seeds=None, use_cell_lists=True,
                 record_histogram=False, use_periodic=True, use_frozen=False,
                 frozen_atoms=None, rcontainer=None):
        # construct base class
        assert not (use_frozen and use_periodic)
        if use_frozen:
            assert not use_periodic and frozen_atoms is not None
            red_coords = reduce_coordinates(full_coords, frozen_atoms, len(boxv))
        else:
            red_coords = full_coords
        super(SpheresMCrunner, self).__init__(potential, red_coords, temperature, niter)

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
        self.use_periodic = use_periodic
        self.rcontainer = rcontainer
        self.equilibration_steps = report_steps + pt_eq_niter
        self.binsize = hbinsize

        # manage array of rattlers, if not rattler: 1 -> jammed dof
        #                                           0 -> rattler dof
        if (rattlers is None):
            self.rattlers = np.array([1. for _ in xrange(self.ndim)], dtype='d')
        else:
            self.rattlers = np.array(rattlers, dtype='d')
        if self.use_frozen:
            self.rattlers = reduce_coordinates(self.rattlers, frozen_atoms, self.bdim)
        assert(len(self.rattlers) == self.ndim)
        assert(self.rattlers.all() >= 0 and self.rattlers.all() <= 1)

        # compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds

        # get potential and minimizer
        self.pot_optmizer = self.get_pot_optimizer()
        self.optmizer = self.get_optimizer()

        # construct base test/action classes
        if record_histogram:
            self._set_record_histogram(hmin, hmax)
        self._set_base_conftests()

        # construct custom test/action/takestep classes
        self.set_report_steps(report_steps)
        self._set_takestep(stepsize)
        self._set_accept_tests()
        self._set_actions()

    def build_base_conftests(self):
        if self.use_frozen:
            self.conftest0 = CheckSphericalContainer(self.rcontainer, self.bdim)
            self.add_conf_test(self.conftest0)
        if self.use_periodic:
            if self.use_cell_lists:
                self.conftest1 = CheckOverlapPeriodicCellLists(self.hs_radii,
                                                               self.boxv, ncellx_scale=self.ncellx_scale,
                                                               use_frozen=self.use_frozen,
                                                               frozen_atoms=self.frozen_atoms,
                                                               reference_coords=self.origin)

            else:
                self.conftest1 = CheckOverlapPeriodic(self.hs_radii,
                                                      self.boxv, use_frozen=self.use_frozen,
                                                      reference_coords=self.origin,
                                                      frozen_atoms=self.frozen_atoms)
        else:
            if self.use_cell_lists:
                self.conftest1 = CheckOverlapCartesianCellLists(self.hs_radii,
                                                                self.boxv, ncellx_scale=self.ncellx_scale,
                                                                use_frozen=self.use_frozen,
                                                                frozen_atoms=self.frozen_atoms,
                                                                reference_coords=self.origin)
            else:
                self.conftest1 = CheckOverlapCartesian(self.hs_radii,
                                                       self.bdim, use_frozen=self.use_frozen,
                                                       reference_coords=self.origin,
                                                       frozen_atoms=self.frozen_atoms)

        self.conftest2 = self.get_check_same_minimum()
        self.add_late_conf_test(self.conftest1)
        self.add_late_conf_test(self.conftest2)

    def _set_record_histogram(self, hmin, hmax):
        self.histogram = RecordDisp2Histogram(self.red_origin, self.rattlers, self.bdim, hmin, hmax,
                                              self.binsize, self.equilibration_steps)
        self.add_action(self.histogram)

    def _set_takestep(self, stepsize):
        raise NotImplementedError

    def _set_actions(self):
        raise NotImplementedError

    def _set_accept_tests(self):
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
        status = super(SpheresMCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status