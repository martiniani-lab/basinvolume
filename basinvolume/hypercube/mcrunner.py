from __future__ import division, print_function
from builtins import range
import numpy as np
import sys

from pele.potentials import Harmonic
from mcpele.monte_carlo import (
    _BaseMCRunner,
    NullPotential,
    UniformSphericalSampling,
)
from basinvolume.monte_carlo import (
    RecordDisplacementTimeseries,
    CheckHyperCubicContainer,
    CheckHyperSphericalContainer,
    RecordStepsTimeseries,
    RecordDisp2Histogram,
)
from mcpele.monte_carlo import MetropolisTest, RandomCoordsDisplacement
from basinvolume.monte_carlo import SampleUniformSphereGaussian
from mcpele.monte_carlo import SampleGaussian
from basinvolume.monte_carlo import Findk
from basinvolume.utils import write_2d_array_to_hdf5
from basinvolume.base_basinvolume import BaseBVMCRunner, BV_MCRunner_State
from basinvolume.utils import setup_matplotlib_for_hypercube, PlottingMixin

try:
    from mcpele.monte_carlo import ConfTestOR
    from mcpele.monte_carlo import RecordCoordsTimeseries
except Exception as e:
    print(e)

# Set up matplotlib for hypercube (no LaTeX)
setup_matplotlib_for_hypercube()

try:
    import matplotlib.pyplot as plt
    from basinvolume.utils import get_color_cycle
    color_cycle = get_color_cycle()
except ImportError as err:
    print(err)


class HypercubeMCrunner(BaseBVMCRunner, PlottingMixin):
    def __init__(
        self,
        bias_potential,
        full_coords,
        temperature,
        stepsize,
        niter,
        origin,
        sidelength=1,
        bias = "harmonic",
        bias_params=[1.0],
        acceptance=0.2,
        adjustf=0.9,
        hmin=0,
        hmax=1,
        hbinsize=0.001,
        adjustf_niter=1e4,
        adjustf_navg=100,
        pt_eq_niter=0,
        ts_niter=None,
        ts_freq=1,
        seeds=None,
        record_steps_timeseries=False,
        record_steps_timeseries_every=[1],
        record_trajectory=False,
        record_trajectory_npoints=1e4,
        single=False,
        record_histogram=False,
    ):
        # construct base class
        super(HypercubeMCrunner, self).__init__(bias_potential, full_coords, temperature, niter)

        self.nparticles = 1
        self.bdim = len(full_coords)
        self.ndof = self.bdim
        self.origin = np.array(origin)
        self.red_origin = origin  # necessary for pt
        self.rattlers = np.ones(self.bdim)
        self.sidelength = sidelength
        # set bias parameters in bias_potential
        self.bias = bias
        self.bias_potential = bias_potential
        self.set_bias_parameters(bias, bias_params)
        self.equilibration_steps = adjustf_niter + pt_eq_niter
        self.adjustf_niter = adjustf_niter
        # actions parameters
        if ts_niter is None:
            ts_niter = niter
        self.ts_niter = ts_niter
        self.ts_freq = ts_freq
        self.record_trajectory = record_trajectory
        self.record_trajectory_npoints = record_trajectory_npoints
        self.record_steps_timeseries = record_steps_timeseries
        self.record_steps_timeseries_every = record_steps_timeseries_every
        self.record_histogram = record_histogram
        self.hmin = hmin
        self.hmax = hmax
        self.hbinsize = hbinsize
        # takestep paramters
        self.adjustf_navg = adjustf_navg
        self.adjustf = adjustf
        self.acceptance = acceptance
        self.single = single
        print(self.sidelength)
        # compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(
                seed_takestep=np.random.randint(i32max),
                seed_metropolis=np.random.randint(i32max),
            )
        self.seeds = seeds

        # set up pele:MC
        self._set_takestep(stepsize)
        self._set_accept_tests()
        self._set_conf_tests()
        self._set_actions()
        self._set_report_steps()

    def _set_takestep(self, stepsize):
        self.takestep = RandomCoordsDisplacement(
            self.seeds["seed_takestep"],
            stepsize,
            report_interval=self.adjustf_navg,
            factor=self.adjustf,
            min_acc_ratio=self.acceptance,
            max_acc_ratio=self.acceptance,
            single=self.single,
            nparticles=self.nparticles,
            bdim=self.bdim,
        )
        self.set_takestep(self.takestep)

    def _set_accept_tests(self):
        self.metropolis = MetropolisTest(self.seeds["seed_metropolis"])
        self.add_accept_test(self.metropolis)

    def _set_conf_tests(self):
        self.conftest = ConfTestOR()
        conftest = CheckHyperCubicContainer(np.zeros(self.ndof), self.sidelength, self.bdim)
        self.conftest.add_test(conftest)
        self.add_late_conf_test(self.conftest)

    def _set_actions(self):
        self.action_record_displ = RecordDisplacementTimeseries(
            self.origin, self.bdim, self.ts_niter, self.ts_freq, fix_com=False
        )
        self.add_action(self.action_record_displ)
        if self.record_histogram:
            self.binsize = self.hbinsize
            self.histogram = RecordDisp2Histogram(
                self.origin,
                self.rattlers,
                self.bdim,
                self.hmin,
                self.hmax,
                self.binsize,
                self.equilibration_steps,
                fix_com=False,
            )
            self.add_action(self.histogram)
        if self.record_trajectory:
            rte = max(
                int((self.niter - self.equilibration_steps) / self.record_trajectory_npoints),
                1,
            )
            self.record_trajectory = RecordCoordsTimeseries(
                self.ndim, record_every=rte, eqsteps=self.equilibration_steps
            )
            self.add_action(self.record_trajectory)
        if self.record_steps_timeseries:
            self.steps_timeseries_list = []
            self.record_steps_timeseries_every = self.record_steps_timeseries_every
            for freq in self.record_steps_timeseries_every:
                self.steps_timeseries_list.append(
                    RecordStepsTimeseries(
                        self.origin,
                        self.rattlers,
                        self.bdim,
                        self.ts_niter,
                        freq,
                        self.equilibration_steps,
                        fix_com=False,
                    )
                )
                self.add_action(self.steps_timeseries_list[-1])

    def _set_report_steps(self):
        self.set_report_steps(0)

    def set_control(self, c):
        pass

    def set_bias_parameters(self, bias, bias_params, reset=True):  # XXX
        """set new bias parameters, generally called from parallel tempering"""
        if bias == "harmonic":
            self.bias_potential.set_k(bias_params[0])
        elif bias == "radial_gaussian":
            self.bias_potential.set_A(bias_params[0])
            self.bias_potential.set_sig(bias_params[1])
        else:
            raise NotImplementedError("bias={} not implemented".format(bias))
        self.bias_params = bias_params
        if reset:
            self.reset()

    def get_stepsize(self):
        return self.takestep.get_stepsize()

    def run_kmin(self):
        """
        run for kmin calculation, typically k ~ 0.01
        """
        self.run(self.ts_niter)

    def get_displ2_kmin(self):
        return self.action_record_displ.get_mean_variance()

    # The common methods (dump_timeseries, get_timeseries, check_convergence, etc.)
    # are now inherited from BaseBVMCRunner

    def dump_steps_timeseries(self, fname, clear=True):
        """Writes RELATIVE DISPLACEMENTS within the walk"""
        for i, steps_timeseries in enumerate(self.steps_timeseries_list):
            timeseries = np.array(steps_timeseries.get_time_series())
            fname_mod = fname + "_" + str(self.record_steps_timeseries_every[i])
            np.savetxt(fname_mod, timeseries)
            if clear:
                steps_timeseries.clear()
        return


class HypercubeFindkMCrunner(_BaseMCRunner):
    """ """

    def __init__(
        self,
        bias_potential,
        full_coords,
        temperature,
        stepsize,
        niter,
        origin,
        sidelength=1,
        target_acceptance=0.9,
        knavg=500,
        ktol=0.05,
        avgcount=1e6,
        hmin=0,
        hmax=1,
        hbinsize=0.001,
        seeds=None,
    ):
        # construct base class
        super(HypercubeFindkMCrunner, self).__init__(bias_potential, full_coords, temperature, niter)

        self.nparticles = 1
        self.bdim = len(full_coords)
        self.ndof = self.bdim
        self.origin = np.array(origin)
        self.red_origin = origin  # necessary for pt
        self.rattlers = np.ones(self.bdim)
        self.sidelength = sidelength
        self.ktarget = target_acceptance
        self.knavg = knavg
        self.ktol = ktol
        self.equilibration_steps = 0
        self.avgcount = avgcount

        # compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(
                seed_findk=np.random.randint(i32max),
                seed_takestep=np.random.randint(i32max),
            )
        self.seeds = seeds

        # construct test/action classes
        self.histogram = RecordDisp2Histogram(
            self.origin,
            self.rattlers,
            self.ndof,
            hmin,
            hmax,
            hbinsize,
            self.equilibration_steps,
            fix_com=False,
        )

        self.conftest = ConfTestOR()
        conftest = CheckHyperCubicContainer(np.zeros(self.ndof), self.sidelength, self.bdim)
        self.conftest.add_test(conftest)

        self.findk = Findk(
            self.seeds["seed_findk"],
            self.ktarget,
            self.knavg,
            self.ktol,
            self.histogram,
            self.avgcount,
        )

        self.takestep = RandomCoordsDisplacement(self.seeds["seed_takestep"], stepsize)

        self.set_report_steps(0)

        # set up pele:MC
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest)
        self.add_action(self.histogram)
        self.add_action(self.findk)

    def set_control(self, c):
        """set k, this changes the stepsize of findk takestep"""
        print(
            "WARNING: set control is not defined, findk spring constant is set by the findk action",
            file=sys.stderr,
        )

    def get_stepsize(self):
        """print the stepsize of findk takestep"""
        return self.takestep.get_stepsize()

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super(HypercubeFindkMCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status

    def get_k(self):
        """get k from action"""
        return self.findk.get_k()

    def get_entries(self):
        """get entries from histogram"""
        return self.histogram.get_histogram()

    def show_histogram(self):
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in range(len(hist))]) + 0.5 * self.binsize
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()


class HypercubeInnerSphereMCrunner(BaseBVMCRunner, PlottingMixin):
    """ """

    def __init__(
        self,
        bias_potential,
        full_coords,
        temperature,
        stepsize,
        niter,
        origin,
        sidelength=1,
        hmin=0,
        hmax=1,
        hbinsize=0.001,
        ts_niter=None,
        ts_freq=1,
        seeds=None,
        record_histogram=False,
        gaussian_step=True,
    ):
        # construct base class
        super(HypercubeInnerSphereMCrunner, self).__init__(
            bias_potential, full_coords, temperature, niter
        )

        self.nparticles = 1
        self.bdim = len(full_coords)
        self.ndof = self.bdim
        self.origin = np.array(origin)
        self.red_origin = origin  # necessary for pt
        self.rattlers = np.ones(self.bdim)
        self.sidelength = sidelength
        self.k = 1.0 / (stepsize * stepsize)
        self.equilibration_steps = 0
        self.gaussian_step = gaussian_step
        self.hbinsize = hbinsize
        if ts_niter is None:
            ts_niter = niter

        # compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds = seeds

        # construct test/action classes
        if record_histogram:
            self.binsize = hbinsize
            self.histogram = RecordDisp2Histogram(
                self.origin,
                self.rattlers,
                self.ndof,
                hmin,
                hmax,
                self.binsize,
                self.equilibration_steps,
                fix_com=False,
            )
            self.add_action(self.histogram)

        self.conftest = ConfTestOR()
        conftest = CheckHyperCubicContainer(np.zeros(self.ndof), self.sidelength, self.bdim)
        self.conftest.add_test(conftest)

        self.action_record_displ = RecordDisplacementTimeseries(
            self.origin, self.ndof, ts_niter, ts_freq, fix_com=False
        )

        self.set_report_steps(0)
        if self.gaussian_step == True:
            self.takestep = SampleUniformSphereGaussian(
                self.seeds["seed_takestep"], stepsize, self.origin
            )
        else:
            self.takestep = UniformSphericalSampling(
                self.seeds["seed_takestep"], stepsize, origin=self.origin
            )

        # set up pele:MC
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest)
        self.add_action(self.action_record_displ)

    def set_control(self, c):
        """set k"""
        print(
            "WARNING: set control is not defined, spring constant is set through stepsize",
            file=sys.stderr,
        )

    def get_stepsize(self):
        return self.takestep.get_stepsize()

    def get_k(self):
        """bias_potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        return k

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super(HypercubeInnerSphereMCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status

    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin, Emax, num=len(hist), endpoint=False, retstep=True)
        Energies += 0.5 * step
        assert abs(step - self.binsize) < self.binsize / 100
        np.savetxt(fname, np.column_stack((Energies, hist)), delimiter="\t")
        mean, variance = self.histogram.get_mean_variance()
        return mean, variance

    # Common methods inherited from BaseBVMCRunner and PlottingMixin


if __name__ == "__main__":
    # to run harmonic bias_potential go to tests

    import time

    ndim = 100
    origin = np.zeros(ndim)
    bias_potential = NullPotential()
    # build start configuration
    full_coords = np.array(origin)
    k = 2  # 0.4261331121440447 # k=25
    stepsize = np.sqrt(1.0 / k)
    if False:
        print("Find k test: \n\n\n")
        test = HypercubeFindkMCrunner(
            bias_potential, full_coords, 1, stepsize, int(1e8), origin, sidelength=1
        )
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
    if False:
        print("MC test: \n\n\n")
        bias_potential = Harmonic(origin, k, bdim=ndim, com=False)
        test = HypercubeMCrunner(
            bias_potential,
            full_coords,
            1,
            stepsize,
            int(1e5),
            origin,
            sidelength=1,
            record_histogram=True,
        )
        start = time.time()
        test.run()
        end = time.time()
        print("displ2_kmin", test.get_displ2_kmin())
        print(end - start)
    if True:
        print("Inner sphere test: \n\n\n")
        test = HypercubeInnerSphereMCrunner(
            bias_potential,
            full_coords,
            1,
            stepsize,
            int(1e5),
            origin,
            sidelength=1,
            record_histogram=True,
        )
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
        test.show_histogram()
