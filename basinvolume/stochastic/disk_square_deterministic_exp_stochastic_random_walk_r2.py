from __future__ import division

import copy as c
import numpy as np
import matplotlib.pyplot as plt

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CloudTest
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import RecordCloudR2
from mcpele.monte_carlo import RecordCoordsTimeseries

from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile
from basinvolume.monte_carlo import CheckHyperCubicContainer
from basinvolume.monte_carlo import CheckHyperSphericalContainer
from basinvolume.monte_carlo import CheckPowerDecayingProfile
from basinvolume.monte_carlo import RecordAcceptanceHistogram
from basinvolume.utils import *


def get_disk_mean_r2(radius):
    return radius ** 2 / 2
    
def get_disk_exp_mean_r2(u, d):
    return (24 * d ** 4 + 24 * d ** 3 * u + 12 * d ** 2 * u ** 2 + 4 * d * u ** 3 + u ** 4) / (4 * d ** 2 + 4 * d * u + 2 * u ** 2)
    
def get_square_mean_r2(side_length):
    return side_length ** 2 / 6
    
def get_square_exp_mean_r2(u, d):
    return 2 * (6 * d ** 3 + 6 * d ** 2 * u + 3 * d * u ** 2 + u ** 3) / (3 * (d + u))
    
def get_disk_profile(u, x):
    return x <= u

def get_disk_exp_profile(u, d, x):
    if x < u:
        return 1
    return np.exp(-(x - u) / d)
    
def get_disk_power_profile(u, e, x):
    if x < u:
        return 1
    return np.power(u / x, e)
    

class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)
    
    
class OracleMCR2(object):
    def __init__(self, common_pars, cloud_pars, oracle, verbose=False, stepsize=1):
        self.common_pars = common_pars
        self.cloud_pars = cloud_pars
        self.oracle = oracle
        self.verbose = verbose
        self.nr_steps = common_pars["mc_steps"]
        self.temperature = 1
        self.origin = common_pars["origin"]
        self.mc_potential = NullPotential()
        self.mc = MC(self.mc_potential, self.origin, self.temperature, self.nr_steps)
        self.eq_steps = self.nr_steps // 2
        ####
        #self.random_walk = RandomCoordsDisplacement(42, stepsize, single=True, nparticles=1, bdim=2, min_acc_ratio=0, max_acc_ratio=1, report_interval=1000)
        self.random_walk = RandomCoordsDisplacement(42, 3 * stepsize, single=True, nparticles=1, bdim=2, min_acc_ratio=0, max_acc_ratio=1, report_interval=100)
        #self.random_walk = RandomCoordsDisplacement(42, 1, single=True, nparticles=1, bdim=2, min_acc_ratio=0.05, max_acc_ratio=0.8, report_interval=100)
        ####
        self.mc.set_takestep(self.random_walk)
        self.cloud_test = CloudTest(44, 46, cloud_pars["nr_points"], cloud_pars["radius"])
        self.cloud_test.add_conf_test(self.oracle)
        self.mc.add_accept_test(self.cloud_test)
        self.cloud_measure_r2 = RecordCloudR2(self.eq_steps, self.origin)
        self.mc.add_action(self.cloud_measure_r2)
        self.mc.set_report_steps(self.eq_steps)
    def run(self):
        self.mc.run()
        if self.verbose:
            print("common_pars", self.common_pars)
            print("cloud_pars", self.cloud_pars)
            print("Backbone step size:", self.random_walk.get_stepsize())
    def get_r2(self):
        return self.cloud_measure_r2.get_mean()
        

class OracleMCAcc(OracleMCR2):
    def __init__(self, common_pars, cloud_pars, oracle, acc_pars, verbose=False, stepsize=1, cts=None):
        super(OracleMCAcc, self).__init__(common_pars, cloud_pars, oracle, verbose, stepsize)
        self.acc_pars = acc_pars
        self.acc_measurement = RecordAcceptanceHistogram(self.common_pars["origin"], self.acc_pars["rmin"],
                                                         self.acc_pars["rmax"], self.acc_pars["nbins"], self.eq_steps)
        self.mc.add_action(self.acc_measurement)
        if cts is not None:
            self.cts = cts
            self.mc.add_action(self.cts)
    def get_acc(self):
        return self.acc_measurement.get_acceptance_fraction_values()
    def write_cts(self, name):
        ts = self.cts.get_time_series()
        np.savetxt(name, ts[:, 0] ** 2 + ts[:, 1] ** 2)
        
    
class DeterministicPlot(BasicPlot):
    def __init__(self, common_pars, cloud_pars):
        self.common_pars = common_pars
        self.cloud_pars = cloud_pars
        self.out_name = "deterministic_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle ($n_d = " + str(self.cloud_pars["nr_points"]) +
                  "$, $r_c=" + str(self.cloud_pars["radius"]) + "$)", fontsize=19)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r"Radius or side length", fontsize=18)
        plt.ylabel(r"Mean squared displacement", fontsize=18)
        plt.tick_params(labelsize=18)
        self.labels = ["Disk, exact", "Square, exact", "Disk, MC", "Square, MC"]
    def run(self):
        self.compute_r2()
        self.make_plot()
    def compute_r2(self):
        self.mc_disk_r2 = np.ones(len(self.common_pars["disk_radii"]))
        self.mc_square_r2 = np.ones(len(self.common_pars["square_sides"]))
        self.run_disk_mc()
        self.run_square_mc()
    def run_disk_mc(self):
        for i, r in enumerate(self.common_pars["disk_radii"]):
            oracle = CheckHyperSphericalContainer(self.common_pars["origin"], r, 2)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle, stepsize=r)
            mc.run()
            self.mc_disk_r2[i] = mc.get_r2()
    def run_square_mc(self):
        for i, l in enumerate(self.common_pars["square_sides"]):
            oracle = CheckHyperCubicContainer(self.common_pars["origin"], l, 2)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle, stepsize=l)
            mc.run()
            self.mc_square_r2[i] = mc.get_r2()
    def make_plot(self):
        symbols = ["s", "o", "^", "v"]
        rs = self.common_pars["disk_radii"]
        ls = self.common_pars["square_sides"]
        rsp = np.linspace(np.amin(rs), np.amax(rs), 1000)
        lsp = np.linspace(np.amin(ls), np.amax(ls), 1000)
        plt.plot(rsp, get_disk_mean_r2(rsp), label=self.labels[0])
        plt.plot(lsp, get_square_mean_r2(lsp), label=self.labels[1])
        plt.plot(rs, self.mc_disk_r2, symbols[0], label=self.labels[2])
        plt.plot(ls, self.mc_square_r2, symbols[1], label=self.labels[3])
        self.save_and_close()
        

class StochasticPlot(DeterministicPlot):
    def __init__(self, common_pars, cloud_pars):
        super(StochasticPlot, self).__init__(common_pars, cloud_pars)
        self.out_name = "stochastic_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle ($n_d = " + str(self.cloud_pars["nr_points"]) + "$, $r_c=" + str(self.cloud_pars["radius"]) + "$)", fontsize=19)
        self.labels = ["Disk+exp, exact", "Square+exp, exact", "Disk+exp, MC", "Square+exp, MC"]
    def run_disk_mc(self):
        for i, r in enumerate(self.common_pars["disk_radii"]):
            oracle = CheckExponentiallyDecayingProfile(self.common_pars["origin"], r, self.common_pars["decay_length"], cubic=False)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle, stepsize=r)
            mc.run()
            self.mc_disk_r2[i] = mc.get_r2()
    def run_square_mc(self):
        for i, l in enumerate(self.common_pars["square_sides"]):
            oracle = CheckExponentiallyDecayingProfile(self.common_pars["origin"], l / 2, self.common_pars["decay_length"], cubic=True)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle, stepsize=l)
            mc.run()
            self.mc_square_r2[i] = mc.get_r2()
    def make_plot(self):
        symbols = ["s", "o", "^", "v"]
        rs = self.common_pars["disk_radii"]
        ls = self.common_pars["square_sides"]
        rsp = np.linspace(np.amin(rs), np.amax(rs), 1000)
        lsp = np.linspace(np.amin(ls), np.amax(ls), 1000)
        plt.plot(rsp, np.asarray([get_disk_exp_mean_r2(r, self.common_pars["decay_length"]) for r in rsp]), label=self.labels[0])
        plt.plot(rsp, get_disk_mean_r2(rsp), "--", label="Disk, exact")
        plt.plot(lsp, np.asarray([get_square_exp_mean_r2(l / 2, self.common_pars["decay_length"]) for l in lsp]), label=self.labels[1])
        plt.plot(lsp, get_square_mean_r2(lsp), "--", label="Square, exact")
        plt.plot(rs, self.mc_disk_r2, symbols[0], label=self.labels[2])
        plt.plot(ls, self.mc_square_r2, symbols[1], label=self.labels[3])
        self.save_and_close()
    

class DeterministicAcceptancePlot_CloudRadius(BasicPlot):
    def __init__(self, common_pars, cloud_radii, nr_points, acc_pars):
        self.common_pars = common_pars
        self.cloud_radii = cloud_radii
        self.nr_points = nr_points
        self.acc_pars = acc_pars
        #self.disk_radius = common_pars["disk_radii"][len(common_pars["disk_radii"]) // 2]
        self.disk_radius = 5
        self.out_name = "deterministic_acceptance_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle ($n_d = " + str(self.nr_points) +"$)", fontsize=19)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r"Trial backbone point distance from center / disk radius $r_d$", fontsize=18)
        plt.ylabel(r"Acceptance probability", fontsize=18)
        plt.tick_params(labelsize=18)
    def run(self):
        self.compute_acceptance()
        self.make_plot()
    def compute_acceptance(self):
        self.disk_acc = []
        self.run_disk_mc()
    def run_disk_mc(self):
        for cr in self.cloud_radii:
            oracle = CheckHyperSphericalContainer(self.common_pars["origin"], self.disk_radius, 2)
            cloud_pars = dict([("nr_points", self.nr_points), ("radius", cr)])
            cts = RecordCoordsTimeseries(2, record_every=self.common_pars["record_every"])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars, stepsize=self.disk_radius, cts=cts)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()
            mc.write_cts("DeterministicAcceptancePlot_CloudRadius_instant_r2_{}.txt".format(cr / self.disk_radius))
    def make_plot(self):
        symbols = ["s--", "o--", "^--", "v--", "d--", "p--", "<--", ">--"]
        for i, cr in enumerate(self.cloud_radii):
            plt.plot(self.disk_acc_x / self.disk_radius, self.disk_acc[i], symbols[i], label=r"$r_c / r_d =$ {0:.3g}".format(cr / self.disk_radius))
        x = np.linspace(self.acc_pars["rmin"], self.acc_pars["rmax"], 1000)
        y = self.get_profile(x)
        plt.plot(x / self.disk_radius, y, "-.", label="Oracle", color="k")
        self.save_and_close(1)
    def get_profile(self, x):
        return get_disk_profile(self.disk_radius, x)


class StochasticAcceptancePlot_CloudRadius(DeterministicAcceptancePlot_CloudRadius):
    def __init__(self, common_pars, cloud_radii, nr_points, acc_pars):
        super(StochasticAcceptancePlot_CloudRadius, self).__init__(common_pars, cloud_radii, nr_points, acc_pars)
        self.out_name = "stochastic_acceptance_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle ($\exp(-r)$, $n_d =" + str(self.nr_points) + "$)", fontsize=19)
    def run_disk_mc(self):
        for cr in self.cloud_radii:
            oracle = CheckExponentiallyDecayingProfile(self.common_pars["origin"], self.disk_radius, self.common_pars["decay_length"], cubic=False)
            cloud_pars = dict([("nr_points", self.nr_points), ("radius", cr)])
            cts = RecordCoordsTimeseries(2, record_every=self.common_pars["record_every"])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars, verbose=True, stepsize=self.disk_radius, cts=cts)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()
            mc.write_cts("StochasticAcceptancePlot_CloudRadius_instant_r2_{}.txt".format(cr / self.disk_radius))
    def get_profile(self, x):
        return [get_disk_exp_profile(self.disk_radius, self.common_pars["decay_length"], xi) for xi in x]
    
    
class StochasticPowerAcceptancePlot_CloudRadius(StochasticAcceptancePlot_CloudRadius):
    def __init__(self, common_pars, cloud_radii, nr_points, acc_pars):
        super(StochasticPowerAcceptancePlot_CloudRadius, self).__init__(common_pars, cloud_radii, nr_points, acc_pars)
        self.out_name = "stochastic_power_acceptance_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle ($r^{-" + str(common_pars["exponent"]) + "}$, $n_d =" + str(self.nr_points) + "$)", fontsize=19)
    def run_disk_mc(self):
        for cr in self.cloud_radii:
            oracle = CheckPowerDecayingProfile(self.common_pars["origin"], self.disk_radius, self.common_pars["exponent"])
            cloud_pars = dict([("nr_points", self.nr_points), ("radius", cr)])
            cts = RecordCoordsTimeseries(2, record_every=self.common_pars["record_every"])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars, verbose=True, stepsize=self.disk_radius, cts=cts)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()
            mc.write_cts("StochasticPowerAcceptancePlot_CloudRadius_instant_r2_{}.txt".format(cr / self.disk_radius))
    def get_profile(self, x):
        return [get_disk_power_profile(self.disk_radius, self.common_pars["exponent"], xi) for xi in x]
    

class DeterministicAcceptancePlot_DropNumber(DeterministicAcceptancePlot_CloudRadius):
    def __init__(self, common_pars, drop_numbers, acc_pars):
        self.common_pars = common_pars
        self.drop_numbers = drop_numbers
        self.acc_pars = acc_pars
        self.disk_radius = 5
        self.cloud_radius = self.disk_radius
        self.out_name = "deterministic_acceptance_drop_number_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle", fontsize=19)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r"Trial backbone point distance from center / disk radius $r_d$", fontsize=18)
        plt.ylabel(r"Acceptance probability", fontsize=18)
        plt.tick_params(labelsize=18)
    def run_disk_mc(self):
        for np in self.drop_numbers:
            oracle = CheckHyperSphericalContainer(self.common_pars["origin"], self.disk_radius, 2)
            cloud_pars = dict([("nr_points", np), ("radius", self.cloud_radius)])
            cts = RecordCoordsTimeseries(2, record_every=self.common_pars["record_every"])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars, stepsize=self.disk_radius, cts=cts)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()
            mc.write_cts("DeterministicAcceptancePlot_DropNumber_instant_r2_{}.txt".format(np))
    def make_plot(self):
        symbols = ["s--", "o--", "^--", "v--", "d--", "p--", "<--", ">--"]
        for i, n in enumerate(self.drop_numbers):
            plt.plot(self.disk_acc_x / self.disk_radius, self.disk_acc[i], symbols[i], label=r"$n_d =$ " + str(n))
        x = np.linspace(self.acc_pars["rmin"], self.acc_pars["rmax"], 1000)
        y = self.get_profile(x)
        plt.plot(x / self.disk_radius, y, "-.", label="Oracle", color="k")
        self.save_and_close(1)
    def get_profile(self, x):
        return get_disk_profile(self.disk_radius, x)


class StochasticAcceptancePlot_DropNumber(DeterministicAcceptancePlot_DropNumber):
    def __init__(self, common_pars, drop_numbers, acc_pars):
        super(StochasticAcceptancePlot_DropNumber, self).__init__(common_pars, drop_numbers, acc_pars)
        self.out_name = "stochastic_acceptance_drop_number_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle ($\exp(-r)$)", fontsize=19)
    def run_disk_mc(self):
        for np in self.drop_numbers:
            oracle = CheckExponentiallyDecayingProfile(self.common_pars["origin"], self.disk_radius, self.common_pars["decay_length"], cubic=False)
            cloud_pars = dict([("nr_points", np), ("radius", self.cloud_radius)])
            cts = RecordCoordsTimeseries(2, record_every=self.common_pars["record_every"])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars, stepsize=self.disk_radius, verbose=True, cts=cts)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()
            mc.write_cts("StochasticAcceptancePlot_DropNumber_instant_r2_{}.txt".format(np))
    def get_profile(self, x):
        return [get_disk_exp_profile(self.disk_radius, self.common_pars["decay_length"], xi) for xi in x]


class StochasticPowerAcceptancePlot_DropNumber(StochasticAcceptancePlot_DropNumber):
    def __init__(self, common_pars, drop_numbers, acc_pars):
        super(StochasticPowerAcceptancePlot_DropNumber, self).__init__(common_pars, drop_numbers, acc_pars)
        self.out_name = "stochastic_power_acceptance_drop_number_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle ($r^{-" + str(common_pars["exponent"]) + "}$)", fontsize=19)
    def run_disk_mc(self):
        for np in self.drop_numbers:
            oracle = CheckPowerDecayingProfile(self.common_pars["origin"], self.disk_radius, self.common_pars["exponent"])
            cloud_pars = dict([("nr_points", np), ("radius", self.cloud_radius)])
            cts = RecordCoordsTimeseries(2, record_every=self.common_pars["record_every"])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars, stepsize=self.disk_radius, verbose=True, cts=cts)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()
            mc.write_cts("StochasticPowerAcceptancePlot_DropNumber_instant_r2_{}.txt".format(np))
    def get_profile(self, x):
        return [get_disk_power_profile(self.disk_radius, self.common_pars["exponent"], xi) for xi in x]    


if __name__ == "__main__":
    disc_radii = np.linspace(1, 10, 20)
    square_sides = np.linspace(1, 10, 20)
    common_pars = dict([("disk_radii", disc_radii),
        ("square_sides", square_sides),
        ("mc_steps", 1000000),
        #("mc_steps", 50000),
        ("origin", np.zeros(2)),
        ("decay_length", 1),
        ("exponent", 5),
        ("record_every", 100)])
    #cloud_pars = dict([("nr_points", 100),
    cloud_pars = dict([("nr_points", 1000),
        ("radius", 1)])
    cloud_radii = np.linspace(0, 10, 5)
    drop_numbers = np.asarray([1, 10, 100, 1000])
    acc_pars = dict([("rmin", 0), ("rmax", 30), ("nbins", 30)])
    dp = DeterministicPlot(common_pars, cloud_pars)
    dp.run()
    sp = StochasticPlot(common_pars, cloud_pars)
    sp.run()
    dpp = DeterministicAcceptancePlot_CloudRadius(common_pars,
        cloud_radii, cloud_pars["nr_points"], acc_pars)
    dpp.run()
    spp = StochasticAcceptancePlot_CloudRadius(common_pars, cloud_radii,
        cloud_pars["nr_points"], acc_pars)
    spp.run()
    spp_pow = StochasticPowerAcceptancePlot_CloudRadius(common_pars,
        cloud_radii, cloud_pars["nr_points"], acc_pars)
    spp_pow.run()
    dad = DeterministicAcceptancePlot_DropNumber(common_pars,
        drop_numbers, acc_pars)
    dad.run()
    sad = StochasticAcceptancePlot_DropNumber(common_pars, drop_numbers,
        acc_pars)
    sad.run()
    sad_pow = StochasticPowerAcceptancePlot_DropNumber(common_pars,
        drop_numbers, acc_pars)
    sad_pow.run()
    
