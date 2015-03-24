from __future__ import division
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import numpy as np
from pele.potentials import HS_WCA
from basinvolume.utils import *
import time
from pele.optimize._quench import modifiedfire_cpp, lbfgs_cpp, cg_descent, steepest_descent
from matplotlib import pyplot as plt
"""
run tests in 
/scratch/sm958/Results/basinvolume_tests/n32_phi88_2D
"""

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def _hist_nnb_midpoint(fname, Xin, Xout):
    """
    histogram the distance from the midpoint of all pairs of nearest neighbours to the closest point out of the basin
    note: should check that midpoint is inside the basin!
    """
    array_dist = []
    for i in xrange(len(Xin)):
        dx = 1e100
        for j in xrange(i+1, len(Xin)):
            dx_trial = np.linalg.norm(Xin[i] - Xin[j])
            if dx_trial < dx:
                nnb_in = (i,j)
        nnb_midpoint = (Xin[nnb_in[0]] + Xin[nnb_in[1]])/2
        #find shorted distance from nnb_midpoint to points out
        dx = 1e100
        for x in Xout:
            dx_trial = np.linalg.norm(nnb_midpoint - x)
            if dx_trial < dx:
                dx = dx_trial
        array_dist.append(dx)
    np.savez(fname[:-4]+"_array_dist_midpoint", array_dist=array_dist)
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
    np.savez(fname[:-4]+"_array_dist", array_dist=array_dist)
    return array_dist

def hist_nnb(fname, Xin, Xout):
    try:
        print "loading data...",
        f = fname[:-4]+"_array_dist"
        data = np.load(f)
        array_dist= data['array_dist']
        print "done"
    except:
        array_dist = _hist_nnb(fname, Xin, Xout)
    
    plt.hist(array_dist, normed=True, bins=14)
    plt.xlabel(r'$|x_{in}-x_{out}|_{nnb}$')
    plt.ylabel(r'$p(|x_{in}-x_{out}|_{nnb})$')

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
                file_list = glob.glob(path + '/TimeSeries*')
                file_list = sorted(file_list, key = lambda x: int(x.split(".")[1]))
                series_order.append(int(dir))
                series = []
                for series_path in file_list:
                    series.extend(read_txt(series_path))
                timeseries.append(series[50000:])
    X = np.array(timeseries)
    Y = series_order
    return np.array([x for (y, x) in sorted(zip(Y, X))])

def _import_ks(explore_dir):
    """
    import spring constants
    """
    karray = [] 
    path = os.path.join(explore_dir, 'temperatures')
    f = open(path, "r")
    while True:
        k = f.readline()
        if not k: break
        karray.extend([float(k)])
    #kmax is not included because we don't have a time series for it
    #karray = np.array(karray[::-1], dtype='d')    
    return karray

#def build_histogram(explore_dir, nbins=100):
#    all_timeseries = _import_time_series(explore_dir)
#    karray = np.array(_import_ks(explore_dir))
#    hist_visits = []
#    bin_edges = np.linspace(np.amin(all_timeseries), np.amax(all_timeseries), nbins+1)
#    for i,timeseries in enumerate(all_timeseries):
#        hist = np.histogram(timeseries, bin_edges)[0]
#        hist_visits.append(hist)
##        hist_red_energy = np.vstack((hist_red_energy, bin_edges[:-1]*karray[i]))
#        print i
#    hist_visits = np.array(hist_visits)
#    hist_red_energy = np.outer(0.5*karray, bin_edges[:-1]**2)
#    assert hist_visits.shape == hist_red_energy.shape
#    assert hist_visits.shape[0] == karray.size
#    return hist_visits, hist_red_energy, karray, bin_edges


def build_histogram(explore_dir, nbins=100):
    #import PT time series
    all_timeseries = _import_time_series(explore_dir)
    karray = _import_ks(explore_dir)
    #import sphere ts
    ts_sphere = np.genfromtxt(os.path.join(explore_dir,"inner_sphere.timeseries"))
    #ts_sphere = np.genfromtxt("test_time_series_unif")
    ts_sphere = np.trim_zeros(ts_sphere) 
    #np.array([x for x in ts_sphere if x > 0]) #remove 0s
    ts_sphere = ts_sphere[:np.shape(all_timeseries)[1]]
    #print ts_sphere
    ksphere = 9.56230003699
    all_timeseries = np.vstack((ts_sphere, all_timeseries))
    karray = [ksphere] + karray
    
    hist_visits = []
    bin_edges = np.linspace(np.amin(all_timeseries), np.amax(all_timeseries), nbins+1)
    for i,timeseries in enumerate(all_timeseries):
        hist = np.histogram(timeseries, bin_edges)[0]
        hist_visits.append(hist)
#        hist_red_energy = np.vstack((hist_red_energy, 0.5 * bin_edges[:-1]*karray[i]))
        print i
    hist_visits = np.array(hist_visits)
    karray = np.array(karray)
    hist_red_energy = np.outer(0.5*karray[1:], bin_edges[:-1]**2)
    hist_red_energy = np.vstack((((24-1)*3-1)*np.log(bin_edges[:-1])+0.5*karray[0]*bin_edges[:-1]**2, hist_red_energy))
    
    assert hist_visits.shape == hist_red_energy.shape
    assert hist_visits.shape[0] == karray.size
    return hist_visits, hist_red_energy, karray, bin_edges
    
def main(explore_dir="explore_bv_jammed_packing1"):
    from histogram_reweighting.wham_potential import WhamPotential
    from histogram_reweighting import wham_utils
    
    hist_visits, hist_red_energy, karray, bin_edges = build_histogram(explore_dir, nbins=500)
    print hist_visits, hist_red_energy, karray
    nreps, nbins = hist_visits.shape
    #print hist_ts
    print np.shape(hist_visits), np.shape(hist_red_energy)
    
    whampot = WhamPotential(hist_visits, hist_red_energy)
    if True:
        X = np.random.rand( nreps + nbins )
    else:
        # estimate an initial guess for the offsets and density of states
        # so the minimizer converges more rapidly
        offsets_estimate, log_dos_estimate = wham_utils.estimate_dos(hist_visits, hist_red_energy)
        X = np.concatenate((offsets_estimate, log_dos_estimate))

    E0, grad = whampot.getEnergyGradient(X)
    rms0 = np.linalg.norm(grad) / np.sqrt(grad.size)
    
    try:
        from pele.optimize import lbfgs_cpp as quench
        if True:
            print "minimizing with pele lbfgs"
        ret = quench(X, whampot, tol=1e-3, maxstep=1e4, nsteps=10000, iprint=1)
    except ImportError:
        from wham_utils import lbfgs_scipy
        if True:
            print "minimizing with scipy lbfgs"
        ret = lbfgs_scipy(X, whampot, tol=1e-3, nsteps=10000)
    #print "quench energy", ret.energy
    
    if True:
        print "chi^2 went from %g (rms %g) to %g (rms %g) in %d iterations" % (
            E0, rms0, ret.energy, ret.rms, ret.nfev)
    
    X = ret.coords
    logn_E = X[nreps:]
    w_i_final = X[:nreps]

    if True:
        plt.plot(bin_edges[:-1], hist_visits.transpose())
        plt.show()
        
        plt.figure()
        for i in xrange(len(karray)):
            x = np.log(hist_visits[i,:]) + hist_red_energy[i,:] + w_i_final[i]
            plt.plot(bin_edges[:-1], x)
        plt.show()
            
    
    
    return logn_E, w_i_final, bin_edges
    
if __name__ == "__main__":
    import scipy
    from scipy.integrate import romb, simps, trapz, quad, nquad
    from scipy.interpolate import interp1d
    import bisect
    logn_E, w_i_final, bin_edges = main()
    import matplotlib.pyplot as plt
    ndof =  ((24-1)*3-1)
    plt.figure()
    plt.plot(bin_edges[:-1], logn_E, label=r'$\log(n_E)$')
    plt.plot(bin_edges[:-1], logn_E - ndof*np.log(bin_edges[:-1]))
    #plt.show()
#    plt.figure()
#    plt.plot(1./bin_edges[:-1], logn_E)
#    plt.plot(1./bin_edges[:-1], logn_E - ndof*np.log(bin_edges[:-1]))
#    plt.show()
    import statsmodels.api as sm
    
    logn_E -= np.amax(logn_E)
    dos = np.exp(logn_E)
    plt.figure()
    plt.plot(bin_edges[:-1], dos)
    #smooth = sm.nonparametric.lowess(dos, bin_edges[:-1], frac=0.1, it=10)
    #dos_smooth, smooth_edges = smooth[:,1], smooth[:,0]
    dosf = interp1d(bin_edges[:-1], dos, kind='cubic')
    dos_interp = dosf(bin_edges[:-1])
    
    plt.plot(bin_edges[:-1], dos_interp)
    #plt.show()
    print logn_E, w_i_final
    print "bin_edges diff", bin_edges[1] - bin_edges[0] - (bin_edges[-1] - bin_edges[-2])
    #volume non smooth
    print "raw"
    print "fake volume",np.trapz(np.ones(len(dos)),dx=bin_edges[1]-bin_edges[0])
    print "bin_edges extrema", bin_edges[0], bin_edges[-1]
    natoms = 24
    ndof = (natoms-1)*3
    rmin = 0.1
    boxv = 5.7897565913256273**3
    vmin = volume_nball(rmin, ndof)
    s = bisect.bisect(bin_edges,rmin)
    #numpy trapez
    print "np.trapez"
    A = vmin / np.trapz(dos[:s],bin_edges[:s])
    Vol = A*np.trapz(dos,bin_edges[:-1])
    print "A: {} Vol: {}".format(A, Vol)
    print "F: {}".format(-np.log(Vol) - np.log(boxv))        
    #scipy trapz
    print "scipy.trapz"
    dx=bin_edges[1]-bin_edges[0]
    A = vmin / trapz(dos[:s],dx=dx)
    Vol = A*trapz(dos, dx=dx)
    print "A: {} Vol: {}".format(A, Vol)
    print "F: {}".format(-np.log(Vol)- np.log(boxv))
    #scipy simps
    print "scipy.simps"
    A = vmin / simps(dos[:s],dx=dx)
    Vol = A*simps(dos,dx=dx)
    print "A: {} Vol: {}".format(A, Vol)
    print "F: {}".format(-np.log(Vol)- np.log(boxv))
    #scipy romb
    print "scipy.romb"
    s2 = 2**np.ceil(np.log2(s))
    dosmin = np.append(np.zeros(s2-s+1), dos[:s])
    A = vmin / simps(dosmin,dx=dx)
    dosromb = np.append(np.zeros(2**np.ceil(np.log2(dos.size))+1), dos)
    Vol = A*simps(dosromb,dx=dx)
    print "A: {} Vol: {}".format(A, Vol)
    print "F: {}".format(-np.log(Vol)- np.log(boxv))
    
    