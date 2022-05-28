import numpy as np
from mcpele.monte_carlo import _BaseMCRunner

class _BaseGeomMCrunner(_BaseMCRunner):
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 geometry="cube", geom_params=[1], k=1.0, hmin=0, hmax=1, hbinsize=0.001,
                 report_steps=0, pt_eq_niter=0, seeds=None):
        super(_BaseGeomMCrunner, self).__init__(potential, full_coords, temperature, niter)

        self.geometry = geometry
        self.geom_params = geom_params
        self.bdim = len(full_coords)
        self.ndof = len(full_coords)
        self.origin = np.array(origin)
        self.red_origin = origin #necessary for pt
        self.rattlers = np.ones(self.ndof)
        self.set_control(k)
        self.equilibration_steps = report_steps + pt_eq_niter
        self.hmin, self.hmax, self.hbinsize = hmin, hmax, hbinsize
        # compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                         seed_metropolis=np.random.randint(i32max),
                         seed_cloud=np.random.randint(i32max),
                         seed_oracle=np.random.randint(i32max),
                         seed_record_drop_r=np.random.randint(i32max))
        self.seeds = seeds

        #set up pele:MC
        self._set_takestep(stepsize)
        self._set_accept_tests()
        self._set_conf_tests()
        self._set_actions()
        self.set_report_steps(report_steps)

    def _get_oracle(self):
        raise NotImplementedError

    def _get_cloud_params(self, cloud_radius, nr_cloud_points):
        raise NotImplementedError

    def _set_accept_tests(self):
        raise NotImplementedError

    def _set_conf_tests(self):
        raise NotImplementedError

    def _set_takestep(self, stepsize):
        raise NotImplementedError

    def _set_actions(self):
        raise NotImplementedError

    def set_control(self, c):
        raise NotImplementedError

    def get_stepsize(self):
        return self.takestep.get_stepsize()

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super(_BaseGeomMCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status