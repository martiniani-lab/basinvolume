from __future__ import division

from builtins import str
from builtins import object
import copy as c
import numpy as np
import matplotlib.pyplot as plt

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CloudTest
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import RecordCloudR2

from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile
from basinvolume.monte_carlo import CheckHyperCubicContainer
from basinvolume.monte_carlo import CheckHyperSphericalContainer
from basinvolume.monte_carlo import RecordAcceptanceHistogram
from basinvolume.utils import BasicPlot


def get_disk_mean_r2(radius):
    return radius**2 / 2


def get_disk_exp_mean_r2(u, d):
    return (
        24 * d**4 + 24 * d**3 * u + 12 * d**2 * u**2 + 4 * d * u**3 + u**4
    ) / (4 * d**2 + 4 * d * u + 2 * u**2)


def get_square_mean_r2(side_length):
    return side_length**2 / 6


def get_square_exp_mean_r2(u, d):
    return 2 * (6 * d**3 + 6 * d**2 * u + 3 * d * u**2 + u**3) / (3 * (d + u))


def get_disk_profile(u, x):
    return x <= u


def get_disk_exp_profile(u, d, x):
    if x < u:
        return 1
    return np.exp(-(x - u) / d)


class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)


class OracleMCR2(object):
    def __init__(self, common_pars, cloud_pars, oracle):
        self.common_pars = common_pars
        self.cloud_pars = cloud_pars
        self.oracle = oracle
        self.nr_steps = common_pars["mc_steps"]
        self.temperature = 1
        self.origin = common_pars["origin"]
        self.mc_potential = NullPotential()
        self.mc = MC(self.mc_potential, self.origin, self.temperature, self.nr_steps)
        self.eq_steps = self.nr_steps // 2
        self.random_walk = RandomCoordsDisplacement(
            42, 1, single=True, nparticles=1, bdim=2
        )
        self.mc.set_takestep(self.random_walk)
        self.cloud_test = CloudTest(
            44, 46, cloud_pars["nr_points"], cloud_pars["radius"]
        )
        self.cloud_test.add_conf_test(self.oracle)
        self.mc.add_accept_test(self.cloud_test)
        self.cloud_measure_r2 = RecordCloudR2(self.eq_steps, self.origin)
        self.mc.add_action(self.cloud_measure_r2)
        self.mc.set_report_steps(self.eq_steps)

    def run(self):
        self.mc.run()

    def get_r2(self):
        return self.cloud_measure_r2.get_mean_r2()


class OracleMCAcc(OracleMCR2):
    def __init__(self, common_pars, cloud_pars, oracle, acc_pars):
        super(OracleMCAcc, self).__init__(common_pars, cloud_pars, oracle)
        self.acc_pars = acc_pars
        self.acc_measurement = RecordAcceptanceHistogram(
            self.common_pars["origin"],
            self.acc_pars["rmin"],
            self.acc_pars["rmax"],
            self.acc_pars["nbins"],
            self.eq_steps,
        )
        self.mc.add_action(self.acc_measurement)

    def get_acc(self):
        return self.acc_measurement.get_acceptance_fraction_values()


class DeterministicPlot(BasicPlot):
    def __init__(self, common_pars, cloud_pars):
        self.common_pars = common_pars
        self.cloud_pars = cloud_pars
        self.out_name = "deterministic_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle", fontsize=19)
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
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
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
            mc.run()
            self.mc_disk_r2[i] = mc.get_r2()

    def run_square_mc(self):
        for i, l in enumerate(self.common_pars["square_sides"]):
            oracle = CheckHyperCubicContainer(self.common_pars["origin"], l, 2)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
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
        plt.title(r"Cloud sampling: stochastic oracle", fontsize=19)
        self.labels = [
            "Disk+exp, exact",
            "Square+exp, exact",
            "Disk+exp, MC",
            "Square+exp, MC",
        ]

    def run_disk_mc(self):
        for i, r in enumerate(self.common_pars["disk_radii"]):
            oracle = CheckExponentiallyDecayingProfile(
                self.common_pars["origin"],
                r,
                self.common_pars["decay_length"],
                cubic=False,
            )
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
            mc.run()
            self.mc_disk_r2[i] = mc.get_r2()

    def run_square_mc(self):
        for i, l in enumerate(self.common_pars["square_sides"]):
            oracle = CheckExponentiallyDecayingProfile(
                self.common_pars["origin"],
                l / 2,
                self.common_pars["decay_length"],
                cubic=True,
            )
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
            mc.run()
            self.mc_square_r2[i] = mc.get_r2()

    def make_plot(self):
        symbols = ["s", "o", "^", "v"]
        rs = self.common_pars["disk_radii"]
        ls = self.common_pars["square_sides"]
        rsp = np.linspace(np.amin(rs), np.amax(rs), 1000)
        lsp = np.linspace(np.amin(ls), np.amax(ls), 1000)
        plt.plot(
            rsp,
            np.asarray(
                [get_disk_exp_mean_r2(r, self.common_pars["decay_length"]) for r in rsp]
            ),
            label=self.labels[0],
        )
        plt.plot(rsp, get_disk_mean_r2(rsp), "--", label="Disk, exact")
        plt.plot(
            lsp,
            np.asarray(
                [
                    get_square_exp_mean_r2(l / 2, self.common_pars["decay_length"])
                    for l in lsp
                ]
            ),
            label=self.labels[1],
        )
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
        # self.disk_radius = common_pars["disk_radii"][len(common_pars["disk_radii"]) // 2]
        self.disk_radius = 5
        self.out_name = "deterministic_acceptance_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle", fontsize=19)
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.xlabel(
            r"Backbone point distance from center / disk radius $r_d$", fontsize=18
        )
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
            oracle = CheckHyperSphericalContainer(
                self.common_pars["origin"], self.disk_radius, 2
            )
            cloud_pars = dict([("nr_points", self.nr_points), ("radius", cr)])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()

    def make_plot(self):
        symbols = ["s--", "o--", "^--", "v--", "d--", "p--", "<--", ">--"]
        for i, cr in enumerate(self.cloud_radii):
            plt.plot(
                self.disk_acc_x / self.disk_radius,
                self.disk_acc[i],
                symbols[i],
                label=r"$r_c / r_d =$ {0:.3g}".format(cr / self.disk_radius),
            )
        x = np.linspace(self.acc_pars["rmin"], self.acc_pars["rmax"], 1000)
        y = self.get_profile(x)
        plt.plot(x / self.disk_radius, y, "-.", label="Oracle", color="k")
        plt.arrow(
            1 + self.cloud_radii[-1] / self.disk_radius,
            0.6,
            0,
            -0.05,
            fc="k",
            ec="k",
            head_width=0.07,
            head_length=0.02,
        )
        self.save_and_close(1)

    def get_profile(self, x):
        return get_disk_profile(self.disk_radius, x)


class StochasticAcceptancePlot_CloudRadius(DeterministicAcceptancePlot_CloudRadius):
    def __init__(self, common_pars, cloud_radii, nr_points, acc_pars):
        super(StochasticAcceptancePlot_CloudRadius, self).__init__(
            common_pars, cloud_radii, nr_points, acc_pars
        )
        self.out_name = "stochastic_acceptance_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle", fontsize=19)

    def run_disk_mc(self):
        for cr in self.cloud_radii:
            oracle = CheckExponentiallyDecayingProfile(
                self.common_pars["origin"],
                self.disk_radius,
                self.common_pars["decay_length"],
                cubic=False,
            )
            cloud_pars = dict([("nr_points", self.nr_points), ("radius", cr)])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()

    def get_profile(self, x):
        return [
            get_disk_exp_profile(self.disk_radius, self.common_pars["decay_length"], xi)
            for xi in x
        ]


class DeterministicAcceptancePlot_DropNumber(DeterministicAcceptancePlot_CloudRadius):
    def __init__(self, common_pars, drop_numbers, cloud_radius, acc_pars):
        self.common_pars = common_pars
        self.drop_numbers = drop_numbers
        self.cloud_radius = cloud_radius
        self.acc_pars = acc_pars
        self.disk_radius = 5
        self.out_name = "deterministic_acceptance_drop_number_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle", fontsize=19)
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.xlabel(
            r"Backbone point distance from center / disk radius $r_d$", fontsize=18
        )
        plt.ylabel(r"Acceptance probability", fontsize=18)
        plt.tick_params(labelsize=18)

    def run_disk_mc(self):
        for np in self.drop_numbers:
            oracle = CheckHyperSphericalContainer(
                self.common_pars["origin"], self.disk_radius, 2
            )
            cloud_pars = dict([("nr_points", np), ("radius", self.cloud_radius)])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()

    def make_plot(self):
        symbols = ["s--", "o--", "^--", "v--", "d--", "p--", "<--", ">--"]
        for i, n in enumerate(self.drop_numbers):
            plt.plot(
                self.disk_acc_x / self.disk_radius,
                self.disk_acc[i],
                symbols[i],
                label=r"$n_d =$ " + str(n),
            )
        x = np.linspace(self.acc_pars["rmin"], self.acc_pars["rmax"], 1000)
        y = self.get_profile(x)
        plt.plot(x / self.disk_radius, y, "-.", label="Oracle", color="k")
        plt.arrow(
            1 + self.cloud_radius / self.disk_radius,
            0.6,
            0,
            -0.05,
            fc="k",
            ec="k",
            head_width=0.07,
            head_length=0.02,
        )
        self.save_and_close(1)

    def get_profile(self, x):
        return get_disk_profile(self.disk_radius, x)


class StochasticAcceptancePlot_DropNumber(DeterministicAcceptancePlot_DropNumber):
    def __init__(self, common_pars, drop_numbers, cloud_radius, acc_pars):
        super(StochasticAcceptancePlot_DropNumber, self).__init__(
            common_pars, drop_numbers, cloud_radius, acc_pars
        )
        self.out_name = "stochastic_acceptance_drop_number_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle", fontsize=19)

    def run_disk_mc(self):
        for np in self.drop_numbers:
            oracle = CheckExponentiallyDecayingProfile(
                self.common_pars["origin"],
                self.disk_radius,
                self.common_pars["decay_length"],
                cubic=False,
            )
            cloud_pars = dict([("nr_points", np), ("radius", self.cloud_radius)])
            mc = OracleMCAcc(self.common_pars, cloud_pars, oracle, self.acc_pars)
            mc.run()
            self.disk_acc.append(mc.get_acc())
            self.disk_acc_x = mc.acc_measurement.get_acceptance_distance_values()

    def get_profile(self, x):
        return [
            get_disk_exp_profile(self.disk_radius, self.common_pars["decay_length"], xi)
            for xi in x
        ]


if __name__ == "__main__":
    disc_radii = np.linspace(1, 10, 20)
    square_sides = np.linspace(1, 10, 20)
    common_pars = dict(
        [
            ("disk_radii", disc_radii),
            ("square_sides", square_sides),
            ("mc_steps", 3000000),
            ("origin", np.zeros(2)),
            ("decay_length", 1),
        ]
    )
    cloud_pars = dict([("nr_points", 100), ("radius", 1)])
    cloud_radii = np.linspace(0, 10, 5)
    drop_numbers = np.asarray([1, 10, 100, 1000])
    acc_pars = dict([("rmin", 0), ("rmax", 20), ("nbins", 20)])
    dp = DeterministicPlot(common_pars, cloud_pars)
    dp.run()
    sp = StochasticPlot(common_pars, cloud_pars)
    sp.run()
    dpp = DeterministicAcceptancePlot_CloudRadius(
        common_pars, cloud_radii, cloud_pars["nr_points"], acc_pars
    )
    dpp.run()
    spp = StochasticAcceptancePlot_CloudRadius(
        common_pars, cloud_radii, cloud_pars["nr_points"], acc_pars
    )
    spp.run()
    dad = DeterministicAcceptancePlot_DropNumber(
        common_pars, drop_numbers, cloud_pars["radius"], acc_pars
    )
    dad.run()
    sad = StochasticAcceptancePlot_DropNumber(
        common_pars, drop_numbers, cloud_pars["radius"], acc_pars
    )
    sad.run()
