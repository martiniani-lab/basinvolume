from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile, CheckPowerDecayingProfile
from basinvolume.utils import *

if __name__ == "__main__":
    u = 3
    d = 0.3
    n = range(2, 21)
    cube = []
    sphere = []
    exp_cube = []
    exp_sphere = []
    pow_sphere = []
    for ni in n:
        o = np.ones(ni)
        tc = CheckExponentiallyDecayingProfile(o, u, d, cubic=True)
        ts = CheckExponentiallyDecayingProfile(o, u, d, cubic=False)
        tp = CheckPowerDecayingProfile(o, u, ni+4)
        cube.append((2 * u) ** ni)
        sphere.append(volume_nball(u, ni))
        exp_cube.append(tc.get_exact_volume())
        exp_sphere.append(ts.get_exact_volume())
        pow_sphere.append(tp.get_exact_volume())
    p = BasicPlot()
    plt.axes().set_yscale("log")
    plt.plot(n, cube, "s-", label="Cube")
    plt.plot(n, sphere, "o-", label="Sphere")
    plt.plot(n, exp_cube, "v-", label="Cube + exp")
    plt.plot(n, exp_sphere, "^-", label="Sphere + exp")
    plt.plot(n, pow_sphere, "x-", label="Sphere + pow(n+3)")
    plt.xlabel(r"Dimension")
    plt.ylabel(r"Volume")
    name = "u = " + str(u) + ", d = " + str(d)
    plt.title(name)
    #plt.legend(loc=2, prop={'size':14})
    #plt.show()
    p.out_name = name.replace(" ", "").replace("=", "-") + ".pdf"
    p.save_and_close(2)

    for n in xrange(2,11):
        tc = CheckExponentiallyDecayingProfile(np.zeros(n), 1., 0.1, cubic=True)
        print "n: {} v: {}".format(n, -np.log(tc.get_exact_volume()))