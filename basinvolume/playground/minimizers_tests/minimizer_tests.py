from __future__ import division
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

def get_X(fname="test_data.npz", pppn=[2,6], nconf=int(1e5)):
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    sim = _kmin_mcrunner('jammed_packing1.xydr', k=0, hmax=10, hbinsize=0.5, opt_tol=1e-7, seeds=seeds, niter=5e3,
                         single=True, use_cell_lists=False, verbose=False)
    try:
        print "loading data...",
        data = np.load(fname)
        X_success = data['X_success']
        X_fail = data['X_fail']
        print "done"
    except:
        mcrunner = sim.mcrunner
        adjust_niter = sim.mc_params['adjustf_niter'] = 5e3
        
        #equilibrate
        print "Equilibrating for {}...".format(adjust_niter),
        mcrunner.run()
        print "done"
        
        #get training data
        X_success = np.empty([0,64])
        X_fail = np.empty([0,64])
        
        print "Generating training samples...",
        for _ in xrange(nconf):
            mcrunner.one_iteration()
            success = mcrunner.get_success() 
            coords = mcrunner.get_trial_coords()
            coords = np.reshape(coords, [1,64])
            if success:
                X_success = np.concatenate((X_success, coords), axis=0)
            else:
                if _check_no_overlaps(coords, sim.hs_radii, sim.boxv):
                    X_fail = np.concatenate((X_fail, coords), axis=0)
        print "done"
        np.savez(fname, X_success=X_success, X_fail=X_fail)
    
    return X_success, X_fail, sim

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
    
    plot_projection(X, fire_Xbool)

def plot_projection(X, Xbool, pair=[0,2], plt_density=False):
    print len(X)
    Xpos = [x for i,x in enumerate(X) if Xbool[i]]
    x, y = [x[pair[0]] for x in Xpos], [x[pair[1]] for x in Xpos]
    if plt_density:
        plot_density(Xpos, pair=pair)
    else:    
        plt.scatter(x, y, marker='s', s=3, edgecolor='none') 
    
def plot_proj_file(fname, array_name='fire_Xbool', pair=[7,3], plt_density=False):
    print "loading data...",
    data = np.load(fname)
    X = data['X']
    Xbool = data[array_name]
    plot_projection(X, Xbool, pair=pair, plt_density=plt_density)

def plot_density(Xpos, pair=[0,2]):
    """
    use a one class svm to fit the density. Takes the full positive array
    """
    from sklearn import svm
    x, y = [x[pair[0]] for x in Xpos], [x[pair[1]] for x in Xpos]
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
    
def main(fname="test_data10k.npz"):
    X_success, X_fail, sim = get_X(fname=fname, pppn=[3,6], nconf=int(1e5))
    pot = sim.mcrunner.pot_optimizer
    maxstep = sim.mc_params['opt_maxstep']
    origin = sim.mcrunner.origin
    nconf = len(X_success)
    print "positives"
    test1(X_success, pot, origin, nconf, maxstep, fname=fname)
    #print "false positives"
    #test1(X_fail, pot, origin, nconf, maxstep)
    
    
   
if __name__ == "__main__":
    #main()
    plot_proj_file("xbool_n13097_test_data10k.npz.npz", pair=[5,27], plt_density=True)
    #plot_proj_file("xbool_n13097_test_data10k.npz.npz", array_name='cgd_Xbool')
    #plot_proj_file("xbool_n13097_test_data10k.npz.npz", array_name='lbfgs_Xbool')
    plt.show()    
                
            
              
                
                
                
