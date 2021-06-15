from __future__ import division
from __future__ import print_function
from builtins import str
import numpy as np
import os
import matplotlib.pyplot as plt
from pele.potentials import Harmonic
from basinvolume.spheres import _configure_mcrunner
from basinvolume.hypercube import HypercubeMCrunner
from basinvolume.utils import trymakedir, trajectory_pca, asphericity_factor, view_traceback
import time
import warnings

class _hypercube_kmin_mcrunner(_configure_mcrunner):
    """
    """
        
    def __init__(self, ndof, sidelength=1, k=0.0, stepsize=5e-1, niter=5e4, acceptance=0.2, 
                 adjustf=0.9, adjustf_niter = 5e3, adjustf_navg = 100, hmin=0, 
                 hmax=0.01, hbinsize=0.0005, record_steps_timeseries=False, 
                 record_steps_timeseries_every=[1],
                 record_trajectory=True, record_trajectory_npoints=1e4,
                 print_diffusion_only=False, single=True, 
                 seeds=None, verbose=False, workspace=None):
                
        self.temperature=1.0
        self.print_diffusion_only = print_diffusion_only
        self.record_steps_timeseries = record_steps_timeseries
        self.ndof = ndof
        self.sidelength = sidelength
        self.coords = np.zeros(self.ndof) #np.ones(self.ndof)*0.32 #CHANGE THIS: I have shifted the centre to see the effect
        if workspace is None:
            self.workspace = os.getcwd()
        else:
            self.workspace = os.path.abspath(workspace)
        
        self._set_paths()
        
        #self.mc_params = dict(k=k, temperature=temperature, )    
        kwargs = dict(sidelength=self.sidelength, k=k, acceptance=acceptance, adjustf=adjustf,
                      adjustf_niter = adjustf_niter, adjustf_navg = adjustf_navg,
                      hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                      record_steps_timeseries=record_steps_timeseries,
                      record_steps_timeseries_every=record_steps_timeseries_every,
                      record_trajectory=record_trajectory, record_trajectory_npoints=record_trajectory_npoints,
                      seeds=seeds, single=single, record_histogram=True)

        self.mc_params = dict(temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")
                
        #construct mcrunner
        #self.coords is origin, set initial configuration and origin to be the same
        potential = Harmonic(self.coords, k, bdim=self.ndof, com=False)
        self.mcrunner = HypercubeMCrunner(potential, self.coords, self.temperature, stepsize, niter,
                                          self.coords, **kwargs)
        
        self._initialise()
        
    def run(self):
        try:
            self.mcrunner.run()
            self.displ_k_min, self.var_displ_k_min = self.mcrunner.histogram.get_mean_variance()
            self._collect_trajectory()
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)
    
    def _collect_trajectory(self):
        mean_coord, var_coord = self.mcrunner.get_mean_variance_coordinate_vector()
        self.mean_coord_dist, self.var_coord_dist = np.linalg.norm(mean_coord-self.mcrunner.origin), np.sum(var_coord)
        self.trajectory = self.mcrunner.dump_trajectory(self.trajectory_path, clear=True)
        self.traj_eval, self.traj_evec = trajectory_pca(self.trajectory)
        self.asphericity = asphericity_factor(self.traj_eval)
    
    def _set_paths(self):
        dname = 'hypercube_n'+str(self.ndof)+'_l'+str(self.sidelength)
        self.base_directory = os.path.join(self.workspace,'explore_bv_'+dname)
        configfile = 'kmin_' + dname
        self.configfile = '{}/{}.config'.format(self.base_directory,configfile)
        trajectory_fname = 'kmin_trajectory_' + dname
        self.trajectory_path = '{}/{}.h5'.format(self.base_directory, trajectory_fname)
        self.diffusion_dir = os.path.join(self.base_directory, "diffusion")
        diffusion_configfname = 'diffusion_' + dname
        self.diffusion_configfname = '{}/{}'.format(self.diffusion_dir, diffusion_configfname)
    
    def _initialise(self):
        self._print_initialise()
         
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        if not self.print_diffusion_only:
            self._print_parameters()
        if self.record_steps_timeseries:
            self._print_diffusion_params()
            
    def _print_diffusion_params(self):
        trymakedir(self.diffusion_dir)
        fname = '{}.{}.config'.format(self.diffusion_configfname, int(self.mc_params['niter']))
        f = open(fname, 'w')
        self._write_sim_params(f)
        f.close()
    
    def _write_sim_params(self, f):
        """
        write simulation parameters
        """
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Explore_Jammed_Packings wrapper class input parameters\n')
        f.write('[KMIN_HYPERCUBE]\n')
        f.write('ndof: {}\n'.format(self.ndof))
        f.write('sidelength: {}\n'.format(self.sidelength))
        f.write('[KMIN_MCRUNNER]\n')

        for key, value in list(self.mc_params.items()) :
            f.write('{}: {}\n'.format(key,value))
    
    def _print_results_once(self, fname):
        """
        note that self.displ_k_min *= 1.5 to account for the limited computation time, 
        this is just an approximation 
        """
        f = open(fname,'a')
        f.write('[KMIN_MCRUNNER_STATUS]\n')
        status = self.mcrunner.get_status()
        for key, value in list(status.items()) :
            f.write('{}: {}\n'.format(key,value))
        f.write('[KMIN]\n')
        f.write('displ_k_min: {:.16f}\n'.format(self.displ_k_min))
        f.write('var_displ_k_min: {:.16f}\n'.format(self.var_displ_k_min))
        f.write('mean_coord_dist: {:.16f}\n'.format(self.mean_coord_dist))
        f.write('var_coord_dist: {:.16f}\n'.format(self.var_coord_dist))
        f.write('pca_asphericity: {:.16f}\n'.format(self.pca_asphericity))
        f.close()
        try:
            self._dump_trajectory()
        except:
            view_traceback()
    
    def _dump_diffusion_timeseries(self):
        fname = "{0}/StepsTimeSeries.{1}".format(self.diffusion_dir, int(self.mc_params['niter']))
        print("fname", fname)
        self.mcrunner.dump_steps_timeseries(fname, clear=True)
    
    def _print_results(self):
        if not self.print_diffusion_only:
            assert(hasattr(self, 'configfile'))
            self._print_results_once(self.configfile)
        if self.record_steps_timeseries:
            configfile = '{}.{}.config'.format(self.diffusion_configfname, int(self.mc_params['niter']))
            assert(os.path.isfile(configfile))
            self._print_results_once(configfile)
            self._dump_diffusion_timeseries()
    
    def _print_success_once(self, success, fname):
        """
        print whether calculation has completed successfully
        this method is overloaded her to check whether this is a 
        diffusion only calculations 
        """
        f = open(fname, 'a')
        f.write('[STATUS]\n')
        f.write('success: {}\n'.format(str(success)))
        f.close()
    
    def _print_success(self, success):
        if not self.print_diffusion_only:
            assert(hasattr(self, 'configfile'))
            self._print_success_once(success, self.configfile)
        if self.record_steps_timeseries:
            configfile = '{}.{}.config'.format(self.diffusion_configfname, int(self.mc_params['niter']))
            assert(os.path.isfile(configfile))
            self._print_success_once(success, configfile)
    
    def _import_packing_config_files(self):
        """pure virtual, must overload"""
        pass
    
if __name__ == "__main__":
    
    #pppn = [2,6,42,1806,47058,2214502422,52495396602]
    #seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    i32max = np.iinfo(np.int32).max
    seeds = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
    ndof = 100
    sim = _hypercube_kmin_mcrunner(ndof, sidelength=1, niter=1e6, k=0, seeds=seeds,
                         single=True, verbose=True, hmax=15, hbinsize=0.001)
    #record_steps_timeseries=True, record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(22)],)
    print('simulation started')
    start=time.time()
    sim.run()
    end=time.time()
    print('time elapsed', end-start)
    status = sim.mcrunner.get_status()
    print(status)
    print('d kmin: ',sim.displ_k_min)
    print('var: ',sim.var_displ_k_min)
    print('mean_coord_dist: ',sim.mean_coord_dist)
    print('var_coord_dist: ', sim.var_coord_dist)
    traj = sim.trajectory
    print(np.shape(traj))
    #sim.mcrunner.show_histogram_kmax()
#    from matplotlib import pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#    from mpl_toolkits.mplot3d import proj3d
#    fig = plt.figure(figsize=(8,8))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot(traj[:,0],traj[:,1],traj[:,2])
#    plt.show()
#    from sklearn.decomposition import PCA
#    pca = PCA(n_components=0.5)
#    pca.fit(traj)
#    print(pca.explained_variance_ratio_)
    
    plt.plot(sim.traj_eval/np.amax(sim.traj_eval))
    #print "eigenvalues", eig_val_cov
    #print "eigenvectors \n", eig_vec_cov
    print("asphericity factor", sim.asphericity)
    #bw = get_bandwidth_estimate(np.array(eig_val_cov), kernel="gaussian", method="cross_validation")
    #edges = np.linspace(np.amin(eig_val_cov), np.amax(eig_val_cov), 1000)
    #hist = get_pdf(eig_val_cov, edges, bandwidth=bw, kernel="tophat")
    #plt.plot(edges, hist, color='k', linewidth=3)
    plt.show()
              
                
                
                
