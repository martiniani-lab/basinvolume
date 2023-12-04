from __future__ import division
from __future__ import print_function
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
import matplotlib.pyplot as plt
import numpy as np
import os
from basinvolume.spheres._kmin_mcrunner import KminMCRunner
from pele.distance import get_distance, Distance
from pele.potentials import HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import read_txt, cround
from basinvolume.enums import Minimizer
import time
from pele.optimize._quench import modifiedfire_cpp, lbfgs_cpp, steepest_descent

try:
    from PyCG_DESCENT import CGDescent
except Exception as e:
    print(e)
"""
run tests in
/scratch/sm958/Results/basinvolume_tests/n32_phi88_2D
"""


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([("", a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def _check_no_overlaps(coords, hs_radii, boxv):
    """check that no two particles are overlapping (using nearest image convention)"""
    no_overlap = True
    bdim = len(boxv)
    nparticles = len(coords) // bdim
    for i in xrange(nparticles):
        if no_overlap == True:
            for j in xrange(i, nparticles):
                dij = np.linalg.norm(
                    get_distance(
                        self.coords[i * bdim : (i + 1) * bdim],
                        self.coords[j * bdim : (j + 1) * bdim],
                        bdim,
                        Distance.PERIODIC,
                        box=boxv,
                    )
                )
                if i != j:
                    dmin = hs_radii[i] + hs_radii[j]
                    if dij - dmin <= 0:
                        no_overlap = False
                        break
        else:
            break
    return no_overlap


def get_X(fname="test_data.npz", pppn=[2, 6], nconf=int(2e5)):
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    sim = KminMCRunner(
        "jammed_packing1.xydr",
        k=0,
        hmax=10,
        hbinsize=0.5,
        opt_tol=1e-5,
        opt_nsteps=1e5,
        seeds=seeds,
        niter=5e3,
        adjustf_niter=5e3,
        single=True,
        use_cell_lists=False,
        minimizer=Minimizer.FIRE,
        verbose=False,
    )
    try:
        print("loading data...", end=" ")
        data = np.load(fname)
        X_success = data["X_success"]
        X_out = data["X_out"]
        X_overlap = data["X_overlap"]
        print("done")
    except Exception as e:
        print("failed")
        print(e)
        mcrunner = sim.mcrunner
        adjust_niter = sim.mc_params["adjustf_niter"] = 5e3

        # equilibrate
        print("Equilibrating for {}...".format(adjust_niter), end=" ")
        mcrunner.run()
        print("done")

        # get training data
        X_success = np.empty([0, 64])
        X_out = np.empty([0, 64])
        X_overlap = np.empty([0, 64])

        print("Generating training samples...", end=" ")
        for _ in xrange(nconf):
            mcrunner.one_iteration()
            success = mcrunner.get_success()
            coords = mcrunner.get_trial_coords()
            if success:
                coords = np.reshape(coords, [1, 64])
                X_success = np.concatenate((X_success, coords), axis=0)
            else:
                if _check_no_overlaps(coords, sim.hs_radii, sim.boxv):
                    coords = np.reshape(coords, [1, 64])
                    X_out = np.concatenate((X_out, coords), axis=0)
                else:
                    coords = np.reshape(coords, [1, 64])
                    X_overlap = np.concatenate((X_overlap, coords), axis=0)
        print("done")
        np.savez(fname, X_success=X_success, X_out=X_out, X_overlap=X_overlap)

    return X_success, X_out, X_overlap, sim


def test_minimizer(minimizer, potential, X, origin, Etol=1e-6, dtol=1e-4, **kwargs):
    def test_same_minimum(coords, E):
        if np.abs(E - Eorigin) > Etol:
            xmean, ymean = np.mean(origin[::2]), np.mean(origin[1::2])
            origin[::2] -= xmean
            origin[1::2] -= ymean
            xmean, ymean = np.mean(coords[::2]), np.mean(coords[1::2])
            coords[::2] -= xmean
            coords[1::2] -= ymean
            if np.linalg.norm(coords - origin) / np.sqrt(len(coords)) > dtol:
                return False
        return True

    Eorigin = potential.getEnergy(origin)
    count = 0
    nfev = 0
    Xbool = []
    for coords in X:
        res = minimizer(coords, potential, **kwargs)
        nfev += res.nfev
        if test_same_minimum(res.coords, res.energy):
            count += 1
            Xbool.append(True)
        else:
            Xbool.append(False)
    return np.array(Xbool), count, nfev


def test_minimizer_single(
    minimizer, potential, coords, origin, Etol=1e-6, dtol=1e-4, **kwargs
):
    def test_same_minimum(coords, E):
        if np.abs(E - Eorigin) > Etol:
            xmean, ymean = np.mean(origin[::2]), np.mean(origin[1::2])
            origin[::2] -= xmean
            origin[1::2] -= ymean
            xmean, ymean = np.mean(coords[::2]), np.mean(coords[1::2])
            coords[::2] -= xmean
            coords[1::2] -= ymean
            if np.linalg.norm(coords - origin) / np.sqrt(len(coords)) > dtol:
                return False
        return True

    Eorigin = potential.getEnergy(origin)
    res = minimizer(coords, potential, **kwargs)
    return test_same_minimum(res.coords, res.energy)


def test1(X, potential, origin, nconf, maxstep, fname="test"):
    print("test1 nconf", nconf)

    fire_Xbool, fire_count, fire_nfev = test_minimizer(
        modifiedfire_cpp,
        potential,
        X[:nconf],
        origin,
        tol=1e-7,
        maxstep=maxstep,
        nsteps=int(1e6),
    )
    lbfgs_Xbool, lbfgs_count, lbfgs_nfev = test_minimizer(
        lbfgs_cpp,
        potential,
        X[:nconf],
        origin,
        tol=1e-7,
        M=1,
        maxErise=1e-4,
        maxstep=maxstep / 10,
        nsteps=int(1e6),
    )
    cgd_Xbool, cgd_count, cgd_nfev = test_minimizer(
        CGDescent, potential, X[:nconf], origin, tol=1e-7, nsteps=int(1e6)
    )

    print(
        "accuracy: fire {} lbfgs {} cgd {} ".format(
            fire_count / nconf, lbfgs_count / nconf, cgd_count / nconf
        )
    )

    print(
        "nfev: fire {:e} lbfgs {:e} cgd {:e} ".format(fire_nfev, lbfgs_nfev, cgd_nfev)
    )

    np.savez(
        "xbool_n{}_{}.npz".format(nconf, fname),
        X=X[:nconf],
        fire_Xbool=fire_Xbool,
        lbfgs_Xbool=lbfgs_Xbool,
        cgd_Xbool=cgd_Xbool,
    )


def _plot_simple_projection(X, Xbool=None, color="b", pair=[0, 2], plt_density=False):
    print(len(X))
    if Xbool is None:
        Xbool = np.ones(len(X))
        Xpos = X
    else:
        Xpos = [x for i, x in enumerate(X) if Xbool[i]]
    x, y = [x[pair[0]] for x in Xpos], [x[pair[1]] for x in Xpos]
    if plt_density:
        plot_density(x, y)
    else:
        plt.scatter(x, y, color=color, marker="s", s=10, edgecolor="none")


def plot_file_simple(fname, array_name="fire_Xbool", pair=[7, 3], plt_density=False):
    print("loading data...", end=" ")
    data = np.load(fname)
    X = data["X"]
    try:
        Xbool = data[array_name]
    except:
        Xbool = None
    _plot_simple_projection(X, Xbool=Xbool, pair=pair, plt_density=plt_density)


def _plot_eig_projection(
    sim,
    X,
    Xbool=None,
    origin=None,
    marker="s",
    color="b",
    msize=10,
    alpha=1,
    plt_density=False,
):
    """
    plot a projection along the largest and smallest eigenvector of the difference between origin and configuration
    """
    from pele.utils.hessian import get_sorted_eig

    print(len(X))
    if Xbool is None:
        Xbool = np.ones(len(X))
        Xpos = X
    else:
        Xpos = [x for i, x in enumerate(X) if Xbool[i]]
    if origin is None:
        origin = sim.mcrunner.origin
    hess = sim.mcrunner.pot_optimizer.getEnergyGradientHessian(origin)[2]
    w, v = get_sorted_eig(hess)
    w = np.real(w)
    vmax = v[-1]
    vmin = v[0]
    x, y = [np.dot(x, vmax) for x in Xpos], [np.dot(x, vmin) for x in Xpos]
    if plt_density:
        plot_density(x, y)
    else:
        plt.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=msize,
            alpha=alpha,
            edgecolor="none",
        )
    plt.xlabel(r"$\mathbf{e}_{max}$")
    plt.ylabel(r"$\mathbf{e}_{min}$")


def _plot_dist_projection(
    sim,
    X,
    origin=None,
    orth="min",
    marker="s",
    color="b",
    msize=10,
    alpha=1,
    plt_density=False,
):
    """
    plot a projection along a vector connecting the two structures and a vector perpendicular to it and an eigenvector
    """
    from pele.utils.hessian import get_sorted_eig

    print(len(X))
    if origin is None:
        origin = sim.mcrunner.origin
    hess = sim.mcrunner.pot_optimizer.getEnergyGradientHessian(origin)[2]
    w, v = get_sorted_eig(hess)
    w = np.real(w)
    vmax = v[-1]
    vmin = v[0]
    if orth == "max":
        v = vmax
    else:
        v = vmin
    x = []
    y = []
    for coords in X:
        x.append(np.linalg.norm(coords))
        y.append(np.dot(coords, v) / np.linalg.norm(coords))

    if plt_density:
        plot_density(x, y)
    else:
        plt.scatter(
            x,
            np.arccos(y) / np.pi,
            color=color,
            marker=marker,
            s=msize,
            alpha=alpha,
            edgecolor="none",
        )
    plt.xlabel(r"$|x-x_o|$")
    if orth == "max":
        plt.ylabel(r"$(\mathbf{x}-\mathbf{x}_o) \cdot \mathbf{e}_{max}$")
    else:
        plt.ylabel(r"$(\mathbf{x}-\mathbf{x}_o) \cdot \mathbf{e}_{min}$")


def plot_file_eig(raw_fname, req_fname, array_name="fire_Xbool", plt_density=False):
    X_success, X_out, X_overlap, sim = get_X(
        fname=raw_fname, pppn=[3, 6], nconf=int(1e5)
    )
    print("loading data...", end=" ")
    data = np.load(req_fname)
    X = data["X"]
    try:
        Xbool = data[array_name]
    except:
        Xbool = None
    _plot_eig_projection(sim, X, Xbool=Xbool, plt_density=plt_density)


def _hist_nnb_midpoint(fname, Xin, Xout):
    """
    histogram the distance from the midpoint of all pairs of nearest neighbours to the closest point out of the basin
    note: should check that midpoint is inside the basin!
    """
    array_dist = []
    for i in xrange(len(Xin)):
        dx = 1e100
        for j in xrange(i + 1, len(Xin)):
            dx_trial = np.linalg.norm(Xin[i] - Xin[j])
            if dx_trial < dx:
                nnb_in = (i, j)
        nnb_midpoint = (Xin[nnb_in[0]] + Xin[nnb_in[1]]) / 2
        # find shorted distance from nnb_midpoint to points out
        dx = 1e100
        for x in Xout:
            dx_trial = np.linalg.norm(nnb_midpoint - x)
            if dx_trial < dx:
                dx = dx_trial
        array_dist.append(dx)
    np.savez(fname[:-4] + "_array_dist_midpoint", array_dist=array_dist)
    plt.hist(array_dist)


def _hist_nnb(fname, Xin, Xout):
    """
    histogram the distance from the midpoint of all pairs of nearest neighbours to the closest point out of the basin
    note: should check that midpoint is inside the basin!
    """
    array_dist = []
    for xin in Xin:
        dx = 1e100
        for x in Xout:
            dx_trial = np.linalg.norm(xin - x)
            if dx_trial < dx:
                dx = dx_trial
        array_dist.append(dx)
    np.savez(fname[:-4] + "_array_dist", array_dist=array_dist)
    return array_dist


def hist_nnb(fname, Xin, Xout):
    try:
        print("loading data...", end=" ")
        f = fname[:-4] + "_array_dist"
        data = np.load(f)
        array_dist = data["array_dist"]
        print("done")
    except:
        array_dist = _hist_nnb(fname, Xin, Xout)

    plt.hist(array_dist, normed=True, bins=14)
    plt.xlabel(r"$|x_{in}-x_{out}|_{nnb}$")
    plt.ylabel(r"$p(|x_{in}-x_{out}|_{nnb})$")


def plot_density(x, y):
    """
    use a one class svm to fit the density. Takes the full positive array
    """
    from sklearn import svm

    xmin, xmax = np.amin(x) - abs(np.amin(x) * 0.05), np.amax(x) + abs(
        np.amax(x) * 0.05
    )
    ymin, ymax = np.amin(y) - abs(np.amin(y) * 0.05), np.amax(y) + abs(
        np.amax(y) * 0.05
    )
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
    # fit the model
    X_train = np.column_stack((x, y))
    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=30)
    print("fitting...", end=" ")
    clf.fit(X_train)
    print("done")
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title("Basin One class fit")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="red")
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="orange")
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white")
    plt.axis("tight")
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.legend(
        [a.collections[0], b1],
        ["learned frontier", "training observations"],
        loc="best",
        framealpha=0.5,
        fancybox=True,
    )


def classify_points(fname="test_data10k.npz"):
    """
    this is a utility function to convert old format data to new format for analysis
    """
    data = np.load(fname)
    X_success = data["X_success"]
    X_fail = data["X_fail"]
    seeds = dict(seed_takestep=1, seed_metropolis=2)
    sim = KminMCRunner(
        "jammed_packing1.xydr",
        k=0,
        hmax=10,
        hbinsize=0.5,
        opt_tol=1e-7,
        seeds=seeds,
        niter=5e3,
        single=True,
        use_cell_lists=False,
        verbose=False,
    )
    X_out = []
    X_overlap = []
    for x in X_fail:
        if _check_no_overlaps(x, sim.hs_radii, sim.boxv):
            X_out.append(x)
        else:
            X_overlap.append(x)
    np.savez(
        fname[:-4] + "_classified",
        X_success=X_success,
        X_out=X_out,
        X_overlap=X_overlap,
    )


def _walk_eig_direction(
    sim,
    index_evec=-1,
    stepsize=0.001,
    distance_array=[],
    te_array=[],
    ev_array=[],
):
    """
    Xbool is an array indicating wether a particuar configuration should be included
    """
    from pele.utils.hessian import get_sorted_eig

    origin = sim.mcrunner.origin
    hess = sim.mcrunner.pot_optimizer.getEnergyGradientHessian(origin)[2]
    # print hess
    w, v = get_sorted_eig(hess)
    w = np.real(w)
    if w[index_evec] < 1e-3:
        print("rattler eigenvector")
        return 0
    evec = v[index_evec] / np.linalg.norm(v[index_evec])
    out = False
    x = np.array(sim.mcrunner.origin)
    d = 0
    backtrack_count = 0
    while out == False:
        x += evec * stepsize
        d += stepsize
        success = _check_no_overlaps(x, sim.mcrunner.hs_radii, sim.mcrunner.boxv)
        if success:
            success = test_minimizer_single(
                CGDescent,
                sim.mcrunner.pot_optimizer,
                x,
                origin,
                tol=1e-7,
                nsteps=int(1e6),
            )
            # success = test_minimizer_single(modifiedfire_cpp, sim.mcrunner.pot_optimizer, x, origin, tol=1e-7, maxstep=0.01, nsteps=int(1e6))
        # print success, stepsize
        if not success and backtrack_count < 10:
            x -= evec * stepsize
            d -= stepsize
            stepsize /= 2
            if stepsize < 1e-12:
                break
            out = False
            backtrack_count += 1
        elif not success:
            x -= evec * stepsize
            d -= stepsize
            out = True
        else:
            backtrack_count = 0
            out = False
    distance_array.append(d)
    ev_array.append(w[index_evec])
    te_array.append(sim.mcrunner.pot_optimizer.getEnergy(x))
    print(d)


def _walk_eig_loop(fname, ndim=128, npackings=250):
    pppn = [3, 6]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    distance_array = []
    te_array = (
        []
    )  # transition state energy (energy at point where we fall out from basin)
    ev_array = []
    for i in xrange(ndim):
        for j in xrange(npackings):
            try:
                sim = KminMCRunner(
                    "jammed_packing{}.xydr".format(j),
                    k=0,
                    hmax=10,
                    hbinsize=0.5,
                    opt_tol=1e-5,
                    opt_nsteps=1e5,
                    seeds=seeds,
                    niter=5e3,
                    adjustf_niter=5e3,
                    single=True,
                    use_cell_lists=False,
                    minimizer=Minimizer.FIRE,
                    verbose=False,
                )
                _walk_eig_direction(
                    sim,
                    stepsize=0.01,
                    index_evec=i,
                    distance_array=distance_array,
                    te_array=te_array,
                    ev_array=ev_array,
                )
            except:
                pass
    return distance_array, te_array, ev_array


def walk_eig(fname):
    try:
        print("loading data...", end=" ")
        f = fname[:-4] + "_walk_eig.npz"
        data = np.load(f)
        distance_array, te_array, ev_array = (
            data["distance_array"],
            data["te_array"],
            data["ev_array"],
        )
        print("done")
    except:
        print("failed")
        distance_array, te_array, ev_array = _walk_eig_loop(fname)
        np.savez(
            fname[:-4] + "_walk_eig",
            distance_array=distance_array,
            te_array=te_array,
            ev_array=ev_array,
        )
    plt.figure()
    plt.scatter(ev_array, distance_array, color="k", marker="s", s=2, edgecolor="none")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\mathbf{x}_o + \delta \mathbf{e}_{\lambda}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(fname[:-4] + "_lamb_dx.pdf")

    from scipy.stats import binned_statistic

    plt.figure()
    dx_means, bin_edges, binnumber = binned_statistic(
        ev_array, distance_array, statistic="mean", bins=20
    )
    bin_means = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in xrange(len(dx_means))]
    plt.plot(bin_means, dx_means, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\mathbf{x}_o + \delta \mathbf{e}_{\lambda}$")
    plt.savefig(fname[:-4] + "_lamb_dx_mean.pdf")

    plt.figure()
    plt.scatter(ev_array, te_array, color="k", marker="s", s=2, edgecolor="none")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\Delta E$")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(fname[:-4] + "_lamb_te.pdf")

    plt.figure()
    te_means, bin_edges, binnumber = binned_statistic(
        ev_array, te_array, statistic="mean", bins=8
    )
    bin_means = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in xrange(len(te_means))]
    plt.plot(bin_means, te_means, marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\Delta E$")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(fname[:-4] + "_lamb_te_mean.pdf")


##import time series routines


def _import_time_series(explore_dir):
    """
    import time series in an array
    """
    import glob

    timeseries = []
    series_order = []
    for subdir, dirs, files in os.walk(explore_dir):
        for dir in dirs:
            if dir.isdigit():
                path = os.path.join(explore_dir, dir)
                file_list = glob.glob(path + "/TimeSeries*")
                file_list = sorted(file_list, key=lambda x: int(x.split(".")[1]))
                series_order.append(int(dir))
                series = []
                for series_path in file_list:
                    series.extend(read_txt(series_path))
                timeseries.append(series)
    X = np.array(timeseries)
    Y = series_order
    return np.array([x for (y, x) in sorted(zip(Y, X))])


def _import_ks(explore_dir):
    """
    import spring constants
    """
    karray = []
    path = os.path.join(explore_dir, "temperatures")
    f = open(path, "r")
    while True:
        k = f.readline()
        if not k:
            break
        karray.extend([float(k)])
    # kmax is not included because we don't have a time series for it
    # karray = np.array(karray[::-1], dtype='d')
    return karray


def build_histogram(explore_dir, bins=100):
    all_timeseries = _import_time_series(explore_dir)
    karray = _import_ks(explore_dir)
    hist_ts = np.empty(bins)
    for timeseries in all_timeseries:
        hist_ts = np.vstack((hist_ts, np.histogram(timeseries, bins)[0]))
    return hist_ts


def main(fname="test_data20k.npz"):
    X_success, X_out, X_overlap, sim = get_X(fname=fname, pppn=[3, 6], nconf=int(1e5))
    pot = sim.mcrunner.pot_optimizer
    maxstep = sim.mc_params["opt_maxstep"]
    origin = sim.mcrunner.origin
    nconf = len(X_success)
    print("nconf ", len(X_overlap))
    # align com to 0
    xmean, ymean = np.mean(origin[::2]), np.mean(origin[1::2])
    origin[::2] -= xmean
    origin[1::2] -= ymean

    for i, x in enumerate(X_success):
        xm, ym = np.mean(x[::2]), np.mean(x[1::2])
        X_success[i][::2] -= xm
        X_success[i][1::2] -= ym

    for i, x in enumerate(X_out):
        xm, ym = np.mean(x[::2]), np.mean(x[1::2])
        X_out[i][::2] -= xm
        X_out[i][1::2] -= ym

    for i, x in enumerate(X_overlap):
        xm, ym = np.mean(x[::2]), np.mean(x[1::2])
        X_overlap[i][::2] -= xm
        X_overlap[i][1::2] -= ym
    # Xf=X_fail
    # test1(X_success, pot, origin, nconf, maxstep, fname=fname)
    # print "false positives"
    # test1(X_fail, pot, origin, nconf, maxstep)
    # _plot_simple_projection(Xf, np.ones(len(Xf)), color='r', pair=[2,3], plt_density=False)
    if False:
        msize = 0.5
        plt.figure()
        _plot_eig_projection(
            sim,
            X_success,
            origin=origin,
            color="b",
            msize=msize,
            plt_density=False,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xsuc_plot.pdf")
        plt.figure()
        _plot_eig_projection(
            sim,
            X_out,
            origin=origin,
            color="k",
            msize=msize,
            plt_density=False,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xout_plot.pdf")
        plt.figure()
        _plot_eig_projection(
            sim,
            X_overlap,
            origin=origin,
            color="r",
            msize=msize,
            plt_density=False,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xove_plot.pdf")
    if False:
        plt.figure()
        msize = 0.5
        _plot_eig_projection(
            sim,
            X_overlap,
            origin=origin,
            color="r",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        _plot_eig_projection(
            sim,
            X_success,
            origin=origin,
            color="b",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        _plot_eig_projection(
            sim,
            X_out,
            origin=origin,
            color="k",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        _plot_eig_projection(
            sim,
            np.array([origin]),
            origin=origin,
            color="g",
            msize=10,
            plt_density=False,
            alpha=1,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xall_plot.pdf")
    if False:
        walk_eig(fname)
    ##########################################################################
    #########from here on we deal with distances from the origin##############
    ##########################################################################
    for i, x in enumerate(X_success):
        X_success[i] -= origin
    for i, x in enumerate(X_out):
        X_out[i] -= origin
    for i, x in enumerate(X_overlap):
        X_overlap[i] -= origin

    if False:
        msize = 0.5
        plt.figure()
        _plot_eig_projection(
            sim,
            X_success,
            origin=origin,
            color="b",
            msize=msize,
            plt_density=False,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xsuc_diff_plot.pdf")
        plt.figure()
        _plot_eig_projection(
            sim,
            X_out,
            origin=origin,
            color="k",
            msize=msize,
            plt_density=False,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xout_diff_plot.pdf")
        plt.figure()
        _plot_eig_projection(
            sim,
            X_overlap,
            origin=origin,
            color="r",
            msize=msize,
            plt_density=False,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xove_diff_plot.pdf")
    if False:
        msize = 0.5
        plt.figure()
        _plot_eig_projection(
            sim,
            X_overlap,
            origin=origin,
            color="r",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        _plot_eig_projection(
            sim,
            X_success,
            origin=origin,
            color="b",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        _plot_eig_projection(
            sim,
            X_out,
            origin=origin,
            color="k",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        plt.axis("scaled")
        plt.savefig(fname[:-4] + "_Xall_diff_plot.pdf")
    # distance vector projection
    if False:
        msize = 0.5
        orth = "max"
        plt.figure()
        _plot_dist_projection(
            sim,
            X_success,
            origin=origin,
            orth=orth,
            color="b",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        plt.savefig(fname[:-4] + "_Xsuc_distpro_plot.pdf")
        plt.figure()
        _plot_dist_projection(
            sim,
            X_out,
            origin=origin,
            orth=orth,
            color="k",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        plt.savefig(fname[:-4] + "_Xout_distpro_plot.pdf")
        plt.figure()
        _plot_dist_projection(
            sim,
            X_overlap,
            origin=origin,
            orth=orth,
            color="r",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        plt.savefig(fname[:-4] + "_Xove_distpro_plot.pdf")
    if False:
        msize = 0.5
        orth = "min"
        plt.figure()
        _plot_dist_projection(
            sim,
            X_overlap,
            origin=origin,
            orth=orth,
            color="r",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        _plot_dist_projection(
            sim,
            X_success,
            origin=origin,
            orth=orth,
            color="b",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        _plot_dist_projection(
            sim,
            X_out,
            origin=origin,
            orth=orth,
            color="k",
            msize=msize,
            plt_density=False,
            alpha=0.5,
        )
        plt.savefig(fname[:-4] + "_Xall_distpro_plot.pdf")
    if False:
        plt.figure()
        # _hist_nnb_midpoint(fname, X_success, np.concatenate((X_out, X_overlap), axis=0))
        hist_nnb(fname, X_success, np.concatenate((X_out, X_overlap), axis=0))
        plt.savefig(fname[:-4] + "_Xsuc_nnb_hist.pdf")


if __name__ == "__main__":
    # main("test_data20k.npz")
    walk_eig("test")
    # classify_points()
    # plot_file_simple("xbool_n13097_test_data10k.npz.npz", pair=[2,3], plt_density=False)
    # plot_file_eig("test_data10k.npz", "xbool_n13097_test_data10k.npz.npz", plt_density=False)
    # plt.show()
