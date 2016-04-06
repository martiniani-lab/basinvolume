from __future__ import division
import numpy as np

from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile
from basinvolume.utils import *

if __name__ == "__main__":
    u = 3
    d = 0.3
    n = range(1, 21)
    cube = []
    sphere = []
    exp_cube = []
    exp_sphere = []
    for ni in n:
        o = np.ones(ni)
        tc = CheckExponentiallyDecayingProfile(o, u, d, cubic=True)
        ts = CheckExponentiallyDecayingProfile(o, u, d, cubic=False)
        cube.append((2 * u) ** ni)
        sphere.append(volume_nball(u, ni))
        exp_cube.append(tc.get_exact_volume())
        exp_sphere.append(ts.get_exact_volume())
    p = BasicPlot()
    plt.axes().set_yscale("log")
    plt.plot(n, cube, "s-", label="Cube")
    plt.plot(n, sphere, "o-", label="Sphere")
    plt.plot(n, exp_cube, "v-", label="Cube + exp")
    plt.plot(n, exp_sphere, "^-", label="Sphere + exp")
    plt.xlabel(r"Dimension")
    plt.ylabel(r"Volume")
    name = "u = " + str(u) + ", d = " + str(d)
    plt.title(name)
    #plt.legend(loc=2, prop={'size':14})
    #plt.show()
    p.out_name = name.replace(" ", "").replace("=", "-") + ".pdf"
    p.save_and_close(2)
