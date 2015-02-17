from __future__ import division
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import numpy as np
from basinvolume.spheres._kmin_mcrunner import _kmin_mcrunner
from pele.potentials import HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import *
import time
from pele.optimize._quench import modifiedfire_cpp, lbfgs_cpp, cg_descent, steepest_descent
"""
run tests in 
/scratch/sm958/Results/basinvolume_tests/n32_phi88_2D
"""

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def _check_no_overlaps(coords, hs_radii, boxv):
    """check that no two particles are overlapping (using nearest image convention)"""
    no_overlap = True
    bdim = len(boxv)
    nparticles = len(coords) // bdim
    for i in xrange(nparticles):
        if no_overlap == True:
            for j in xrange(i, nparticles):
                dij = 0
                for k in xrange(bdim):
                    #use distances to nearest image convention
                    dij += np.square((coords[i*bdim+k] - coords[j*bdim+k]) -
                                      cround((coords[i*bdim+k] - coords[j*bdim+k]) / boxv[k]) * boxv[k])
                if i != j:
                    dij = np.sqrt(dij)
                    dmin = hs_radii[i]+hs_radii[j]
                    if dij - dmin <= 0:
                        no_overlap = False
                        break
        else:
            break
    return no_overlap

def get_X(fname="test_data.npz", pppn=[2,6], nconf=int(2e5)):
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    sim = _kmin_mcrunner('jammed_packing1.xydr', k=0, hmax=10, hbinsize=0.5, opt_tol=1e-5, opt_nsteps=1e5, 
                         seeds=seeds, niter=5e3, adjustf_niter=5e3, single=True, use_cell_lists=False, 
                         use_cgd=True, verbose=False)
    try:
        print "loading data...",
        data = np.load(fname)
        X_success = data['X_success']
        X_out = data['X_out']
        X_overlap = data['X_overlap']
        print "done"
    except Exception, e:
        print "failed"
        print e
        mcrunner = sim.mcrunner
        adjust_niter = sim.mc_params['adjustf_niter'] = 5e3
        
        #equilibrate
        print "Equilibrating for {}...".format(adjust_niter),
        mcrunner.run()
        print "done"
        
        #get training data
        X_success = np.empty([0,64])
        X_out = np.empty([0,64])
        X_overlap = np.empty([0,64])
        
        print "Generating training samples...",
        for _ in xrange(nconf):
            mcrunner.one_iteration()
            success = mcrunner.get_success() 
            coords = mcrunner.get_trial_coords()
            if success:
                coords = np.reshape(coords, [1,64])
                X_success = np.concatenate((X_success, coords), axis=0)
            else:
                if _check_no_overlaps(coords, sim.hs_radii, sim.boxv):
                    coords = np.reshape(coords, [1,64])
                    X_out = np.concatenate((X_out, coords), axis=0)
                else:
                    coords = np.reshape(coords, [1,64])
                    X_overlap = np.concatenate((X_overlap, coords), axis=0)
        print "done"
        np.savez(fname, X_success=X_success, X_out=X_out, X_overlap=X_overlap)
    
    return X_success, X_out, X_overlap, sim

def test_minimizer(minimizer, potential, X, origin, Etol=1e-6, dtol=1e-4, **kwargs):
    def test_same_minimum(coords, E):
        if np.abs(E-Eorigin) > Etol:
            if np.linalg.norm(coords - origin)/np.sqrt(len(coords)) > dtol:
                return False
        return True
    Eorigin = potential.getEnergy(origin)
    count=0
    nfev=0
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
    
def test1(X, potential, origin, nconf, maxstep, fname="test"):
    print "test1 nconf", nconf
    
    fire_Xbool, fire_count, fire_nfev = test_minimizer(modifiedfire_cpp, potential, X[:nconf], origin, 
                                             tol=1e-7, maxstep=maxstep, nsteps=int(1e6))
    lbfgs_Xbool, lbfgs_count, lbfgs_nfev = test_minimizer(lbfgs_cpp, potential, X[:nconf], origin,
                                             tol=1e-7, M=1, maxErise=1e-4, maxstep=maxstep/10, 
                                             nsteps=int(1e6))
    cgd_Xbool, cgd_count, cgd_nfev = test_minimizer(cg_descent, potential, X[:nconf], origin,
                                             tol=1e-7, nsteps=int(1e6))
    
    print "accuracy: fire {} lbfgs {} cgd {} ".format(fire_count/nconf, 
                                                     lbfgs_count/nconf, 
                                                     cgd_count/nconf)
    
    print "nfev: fire {:e} lbfgs {:e} cgd {:e} ".format(fire_nfev, lbfgs_nfev, cgd_nfev)
    
    np.savez("xbool_n{}_{}.npz".format(nconf, fname), X=X[:nconf], 
             fire_Xbool=fire_Xbool, lbfgs_Xbool=lbfgs_Xbool, 
             cgd_Xbool=cgd_Xbool)

def _plot_simple_projection(X, Xbool=None, color='b', pair=[0,2], plt_density=False):
    print len(X)
    if Xbool is None:
        Xbool = np.ones(len(X))
        Xpos = X
    else:
        Xpos = [x for i,x in enumerate(X) if Xbool[i]]
    x, y = [x[pair[0]] for x in Xpos], [x[pair[1]] for x in Xpos]
    if plt_density:
        plot_density(x, y)
    else:    
        plt.scatter(x, y, color=color, marker='s', s=10, edgecolor='none') 
    
def plot_file_simple(fname, array_name='fire_Xbool', pair=[7,3], plt_density=False):
    print "loading data...",
    data = np.load(fname)
    X = data['X']
    try:
        Xbool = data[array_name]
    except:
        Xbool=None
    _plot_simple_projection(X, Xbool=Xbool, pair=pair, plt_density=plt_density)

def _plot_eig_projection(sim, X, Xbool=None, origin=None, marker='s', color='b', msize=10, alpha=1, plt_density=False):
    """
    plot a projection along the largest and smallest eigenvector of the difference between origin and configuration
    """
    from pele.utils.hessian import get_sorted_eig
    print len(X)
    if Xbool is None:
        Xbool = np.ones(len(X))
        Xpos = X
    else:
        Xpos = [x for i,x in enumerate(X) if Xbool[i]]
    if origin is None:
        origin = sim.mcrunner.origin
    hess = sim.mcrunner.pot_optimizer.getEnergyGradientHessian(origin)[2]
    w, v = get_sorted_eig(hess)
    w = np.real(w)
    vmax = v[-1]
    vmin = v[0]
    x, y = [np.dot(x,vmax) for x in Xpos], [np.dot(x,vmin) for x in Xpos]
    if plt_density:
        plot_density(x, y)
    else:
        plt.scatter(x, y, color=color, marker=marker, s=msize, alpha=alpha, edgecolor='none')
    plt.xlabel(r'$\mathbf{e}_{max}$')
    plt.ylabel(r'$\mathbf{e}_{min}$')

def _plot_dist_projection(sim, X, origin=None, orth='min', marker='s', color='b', msize=10, alpha=1, plt_density=False):
    """
    plot a projection along a vector connecting the two structures and a vector perpendicular to it and an eigenvector
    """
    from pele.utils.hessian import get_sorted_eig
    print len(X)
    if origin is None:
        origin = sim.mcrunner.origin
    hess = sim.mcrunner.pot_optimizer.getEnergyGradientHessian(origin)[2]
    w, v = get_sorted_eig(hess)
    w = np.real(w)
    vmax = v[-1]
    vmin = v[0]
    if orth == 'max':
        v = vmax
    else:
        v = vmin
    x = []
    y = []
    for coords in X:
        dx = coords-origin
        dxv = dx / np.linalg.norm(dx) #distance vector
        cxv = np.ones(len(coords))
        cxv -= np.dot(cxv, dxv) * dxv #orthogonalize to 1 vector
        cxv -= np.dot(cxv, v) * v #orthogonalize to 2 vector
        cxv /= np.linalg.norm(cxv) #perpendicular unit vector
        x.append(np.dot(coords,dxv))
        y.append(np.dot(coords,cxv))
        
    if plt_density:
        plot_density(x, y)
    else:
        plt.scatter(x, y, color=color, marker=marker, s=msize, alpha=alpha, edgecolor='none')
    plt.xlabel(r'$|x-x_o|$')
    if orth == 'max':
        plt.ylabel(r'$(\mathbf{x}-\mathbf{x}_o) \times \mathbf{e}_{max}$')
    else:
        plt.ylabel(r'$(\mathbf{x}-\mathbf{x}_o) \times \mathbf{e}_{min}$')
    
def plot_file_eig(raw_fname, req_fname, array_name='fire_Xbool', plt_density=False):
    X_success, X_out, X_overlap, sim = get_X(fname=raw_fname, pppn=[3,6], nconf=int(1e5))
    print "loading data...",
    data = np.load(req_fname)
    X = data['X']
    try:
        Xbool = data[array_name]
    except:
        Xbool=None
    _plot_eig_projection(sim, X, Xbool=Xbool, plt_density=plt_density)


def plot_density(x,y):
    """
    use a one class svm to fit the density. Takes the full positive array
    """
    from sklearn import svm
    xmin, xmax = np.amin(x)-abs(np.amin(x)*0.05), np.amax(x)+abs(np.amax(x)*0.05)
    ymin, ymax = np.amin(y)-abs(np.amin(y)*0.05), np.amax(y)+abs(np.amax(y)*0.05)
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
    # fit the model
    X_train = np.column_stack((x,y))
    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=30)
    print "fitting...",
    clf.fit(X_train)
    print "done"
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title("Basin One class fit")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
    plt.axis('tight')
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.legend([a.collections[0], b1],
           ["learned frontier", "training observations"],
           loc="best", framealpha=0.5, fancybox=True)

def classify_points(fname="test_data10k.npz"):
    """
    this is a utility function to convert old format data to new format for analysis
    """
    data = np.load(fname)
    X_success = data['X_success']
    X_fail = data['X_fail']
    seeds = dict(seed_takestep=1, seed_metropolis=2)
    sim = _kmin_mcrunner('jammed_packing1.xydr', k=0, hmax=10, hbinsize=0.5, opt_tol=1e-7, seeds=seeds, niter=5e3,
                         single=True, use_cell_lists=False, verbose=False)
    X_out = []
    X_overlap = []
    for x in X_fail:
        if _check_no_overlaps(x, sim.hs_radii, sim.boxv):
            X_out.append(x)
        else:
            X_overlap.append(x)
    np.savez(fname[:-4]+"_classified", X_success=X_success, X_out=X_out, X_overlap=X_overlap)

def _walk_eig_direction(sim, stepsize=0.001):
    """
    Xbool is an array indicating wether a particuar configuration should be included
    """
    from pele.utils.hessian import get_sorted_eig
    origin = sim.mcrunner.origin
    hess = sim.mcrunner.pot_optimizer.getEnergyGradientHessian(origin)[2]
    print hess
    w, v = get_sorted_eig(hess)
    w = np.real(w)
    vmax = v[-1]/np.linalg.norm(v[-1])
    vmin = v[0]/np.linalg.norm(v[0])
    out = False
    x = np.array(sim.mcrunner.origin)
    d = 0
    backtrack_count = 0
    while out == False:
        x += vmax*stepsize
        d += stepsize
        #success = test_minimizer(cg_descent, sim.mcrunner.pot_optimizer, [x], origin, tol=1e-7, nsteps=int(1e6))[0][0]
        success = test_minimizer(modifiedfire_cpp, sim.mcrunner.pot_optimizer, [x], origin, tol=1e-7, maxstep=0.01, nsteps=int(1e6))[0][0]
        print success
        if not success and backtrack_count < 10:
            x -= vmin*stepsize
            d -= stepsize
            stepsize /= 2
            out = False
            backtrack_count += 1
        else:
            out = not success
        print d

def walk_eig(fname="test_data20k.npz"):
    sim = get_X(fname=fname, pppn=[3,6], nconf=int(1e5))[3]
    _walk_eig_direction(sim, stepsize=0.001)
    
    
def main(fname="test_data20k.npz"):
    X_success, X_out, X_overlap, sim = get_X(fname=fname, pppn=[3,6], nconf=int(1e5))
    pot = sim.mcrunner.pot_optimizer
    maxstep = sim.mc_params['opt_maxstep']
    origin = sim.mcrunner.origin
    nconf = len(X_success)
    print "nconf ",len(X_overlap)
    #align com to 0
    xmean, ymean = np.mean(origin[::2]), np.mean(origin[1::2])
    origin[::2] -= xmean
    origin[1::2] -= ymean
    
    for i,x in enumerate(X_success):
        xm, ym = np.mean(x[::2]), np.mean(x[1::2])
        X_success[i][::2] -= xm
        X_success[i][1::2] -= ym
        
    for i,x in enumerate(X_out):
        xm, ym = np.mean(x[::2]), np.mean(x[1::2])
        X_out[i][::2] -= xm
        X_out[i][1::2] -= ym
    
    for i,x in enumerate(X_overlap):
        xm, ym = np.mean(x[::2]), np.mean(x[1::2])
        X_overlap[i][::2] -= xm
        X_overlap[i][1::2] -= ym
    #Xf=X_fail
    #test1(X_success, pot, origin, nconf, maxstep, fname=fname)
    #print "false positives"
    #test1(X_fail, pot, origin, nconf, maxstep)
    #_plot_simple_projection(Xf, np.ones(len(Xf)), color='r', pair=[2,3], plt_density=False)
    if False:
        msize = 0.5
        plt.figure()
        _plot_eig_projection(sim, X_success, origin=origin, color='b', msize=msize, plt_density=False)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xsuc_plot.pdf")
        plt.figure()
        _plot_eig_projection(sim, X_out, origin=origin, color='k', msize=msize, plt_density=False)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xout_plot.pdf")
        plt.figure()
        _plot_eig_projection(sim, X_overlap, origin=origin, color='r', msize=msize, plt_density=False)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xove_plot.pdf")
    if False:
        plt.figure()
        msize=0.5
        _plot_eig_projection(sim, X_overlap, origin=origin, color='r', msize=msize, plt_density=False, alpha=0.5)
        _plot_eig_projection(sim, X_success, origin=origin, color='b', msize=msize, plt_density=False, alpha=0.5)
        _plot_eig_projection(sim, X_out, origin=origin, color='k', msize=msize, plt_density=False, alpha=0.5)
        _plot_eig_projection(sim, np.array([origin]), origin=origin, color='g', msize=10, plt_density=False, alpha=1)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xall_plot.pdf")
    
    #from here on we deal with distances from the origin
    for i,x in enumerate(X_success):
        X_success[i] -= origin
    for i,x in enumerate(X_out):
        X_out[i] -= origin
    for i,x in enumerate(X_overlap):
        X_overlap[i] -= origin
            
    if False:
        msize = 0.5
        plt.figure()
        _plot_eig_projection(sim, X_success, origin=origin, color='b', msize=msize, plt_density=False)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xsuc_diff_plot.pdf")
        plt.figure()
        _plot_eig_projection(sim, X_out, origin=origin, color='k', msize=msize, plt_density=False)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xout_diff_plot.pdf")
        plt.figure()
        _plot_eig_projection(sim, X_overlap, origin=origin, color='r', msize=msize, plt_density=False)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xove_diff_plot.pdf")
    if False:
        msize = 0.5
        plt.figure()
        _plot_eig_projection(sim, X_overlap, origin=origin, color='r', msize=msize, plt_density=False, alpha=0.5)
        _plot_eig_projection(sim, X_success, origin=origin, color='b', msize=msize, plt_density=False, alpha=0.5)
        _plot_eig_projection(sim, X_out, origin=origin, color='k', msize=msize, plt_density=False, alpha=0.5)
        plt.axis('scaled')
        plt.savefig(fname[:-4]+"_Xall_diff_plot.pdf")
    #distance vector projection
    if False:
        msize = 0.5
        orth = 'min'
        plt.figure()
        _plot_dist_projection(sim, X_success, origin=origin, orth=orth, color='b', msize=msize, plt_density=False, alpha=0.5)
        plt.savefig(fname[:-4]+"_Xsuc_distpro_plot.pdf")
        plt.figure()
        _plot_dist_projection(sim, X_out, origin=origin, orth=orth, color='k', msize=msize, plt_density=False, alpha=0.5)
        plt.savefig(fname[:-4]+"_Xout_distpro_plot.pdf")
        plt.figure()
        _plot_dist_projection(sim, X_overlap, origin=origin, orth=orth, color='r', msize=msize, plt_density=False, alpha=0.5)
        plt.savefig(fname[:-4]+"_Xove_distpro_plot.pdf")
    if True:
        msize = 0.5
        orth = 'max'
        plt.figure()
        _plot_dist_projection(sim, X_overlap, origin=origin, orth=orth, color='r', msize=msize, plt_density=False, alpha=0.5)
        _plot_dist_projection(sim, X_success, origin=origin, orth=orth, color='b', msize=msize, plt_density=False, alpha=0.5)
        _plot_dist_projection(sim, X_out, origin=origin, orth=orth, color='k', msize=msize, plt_density=False, alpha=0.5)
        plt.savefig(fname[:-4]+"_Xall_distpro_plot.pdf")

if __name__ == "__main__":
    main("test_data20k.npz")
    #walk_eig()
    #classify_points()
    #plot_file_simple("xbool_n13097_test_data10k.npz.npz", pair=[2,3], plt_density=False)
    #plot_file_eig("test_data10k.npz", "xbool_n13097_test_data10k.npz.npz", plt_density=False)
    #plt.show()    
                
            
              
                
                
                
