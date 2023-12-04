from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import next
from builtins import range
from basinvolume.utils._utils import _sort_pair
from scipy.misc import factorial

try:
    import numpy as np
    import argparse
    import configparser
    import os
    import re
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from itertools import cycle
    from basinvolume.utils import trymakedir
    import scipy
    from scipy.optimize import fmin
    from scipy.stats import t
    from scipy.interpolate import spline
    from scipy.integrate import romberg, simps, quad, cumtrapz, trapz
    import glob
    from itertools import chain
    import pickle as pickle
    from basinvolume.post_processing import (
        PackingData,
        PackingDataSet,
        BasinAnalysis,
    )
    from basinvolume.experiment_2d.cross_validation_bandwidth_selection import (
        get_bandwidth_estimate,
        get_pdf,
    )
    from basinvolume.post_processing import (
        GeneralisedLogNormal,
        OutlierRemovalUnbiasingEntropyLogOmega,
    )
except ImportError as err:
    print(err)
#######################SET LATEX OPTIONS###################
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc('text.latex',preamble=r'\usepackage{times}')
plt.rcParams.update({"font.size": 18})
plt.rcParams["xtick.major.pad"] = 8
plt.rcParams["ytick.major.pad"] = 8
plt.rcParams.update({"figure.autolayout": True})
##########################################################
####SET COLOUR MAP######
def get_color_cycle():
    cm = plt.get_cmap("Set2")
    color_cycle = cycle([cm(1.0 * i / 7) for i in range(7)])
    return color_cycle


########################
#####################LINE STYLE CYCLER####################
lines = ["-", "--", "-.", ":", "_"]
linecycler = cycle(lines)
##########################################################


def myplot(packing_datasets, figdir="figures"):
    from scipy.optimize import curve_fit

    if not os.path.isabs(figdir):
        figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)
    glob_kappa = 0.1833167718536277
    glob_interc = 0.9692762142432549

    if True:
        # kde free energies predicted vs numerical
        # note the x2 = (nparticles/glob_kappa)*np.log(x2)+glob_interc*nparticles (should be just +N)
        color_cycle = get_color_cycle()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, dataset in enumerate(
            sorted(packing_datasets, key=lambda data: data.nparticles)
        ):
            if len(dataset.pressures) > 0:
                nparticles = dataset.nparticles
                color = next(color_cycle)
                x2 = np.array(dataset.pressures)
                x2 = (nparticles / glob_kappa) * np.log(x2) + glob_interc * nparticles
                bw2 = get_bandwidth_estimate(
                    np.array(x2), kernel="gaussian", method="cross_validation"
                )
                edges2 = np.linspace(np.amin(x2) * 0.5, np.amax(x2) * 1.5, 1000)
                hist2 = get_pdf(x2, edges2, bandwidth=bw2, kernel="gaussian")
                # hist2 *= edges2/(nparticles/glob_kappa)
                # hist2 /= simps(hist2, x=edges2)
                ax.plot(edges2, hist2, "--", color=color, linewidth=3)

        ax.legend(
            frameon=False,
            loc="best",
            numpoints=1,
            markerscale=0.5,
            columnspacing=0.25,
            labelspacing=0.25,
            handlelength=0.4,
        )
        plt.ylabel(r"$p(F)$")
        plt.xlabel(r"$F$")
        # ax.set_xlim((30,340))
        fig.savefig("{0}/plot_{1}.pdf".format(figdir, "predicted_F_dist"))


if __name__ == "__main__":
    pts = BasinAnalysis()
    pts.collect_data_every_set_structure(dir_signature="n*_phi*_phi*_*D_inc*")
    myplot(pts.packing_datasets)
    plt.show()
    plt.close()
