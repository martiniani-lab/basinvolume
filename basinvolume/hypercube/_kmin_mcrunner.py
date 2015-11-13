from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic
from basinvolume.spheres import _configure_mcrunner
from basinvolume.hypercube import HypercubeMCrunner
from basinvolume.utils import *
import ConfigParser
import time

class _hypercube_kmin_mcrunner(_configure_mcrunner):
    """
    """
        
    def __init__(self, ndof, sidelength=1, k=0.0, stepsize=5e-1, niter=5e4, acceptance=0.2, 
                 adjustf=0.9, adjustf_niter = 5e3, adjustf_navg = 100, hmin=0, 
                 hmax=0.01, hbinsize=0.0005, record_steps_timeseries=False, 
                 record_steps_timeseries_every=[1], print_diffusion_only=False, single=True, 
                 seeds=None, verbose=False, workspace=None):
                
        self.temperature=1.0
        self.print_diffusion_only = print_diffusion_only
        self.record_steps_timeseries = record_steps_timeseries
        self.ndof = ndof
        self.sidelength = sidelength
        self.coords = np.zeros(self.ndof)
        if workspace is None:
            self.workspace = os.getcwd()
        else:
            self.workspace = os.path.abspath(workspace)
        
        self._set_paths()
        
        #self.mc_params = dict(k=k, temperature=temperature, )    
        self.mc_params = {'k':k,'temperature':self.temperature,'niter':niter,'stepsize':stepsize,
                          'acceptance':acceptance,'adjustf':adjustf,'adjustf_niter':adjustf_niter,
                          'adjustf_navg':adjustf_navg, 'hmin':hmin,'hmax':hmax,'hbinsize':hbinsize,
                          'record_steps_timeseries':record_steps_timeseries}
        #add seeds dictionary to mc_params
        try:
            self.mc_params.update(seeds)
        except:
            print "WARNING:seeds not passed"
                
        #construct mcrunner
        #self.coords is origin, set initial configuration and origin to be the same
        potential = Harmonic(self.coords, k, bdim=self.ndof, com=False)
        self.mcrunner = HypercubeMCrunner(potential, self.coords, self.temperature, stepsize, niter, self.coords, 
                                          sidelength=self.sidelength, k=k, acceptance=acceptance, adjustf=adjustf, 
                                          adjustf_niter = adjustf_niter, adjustf_navg = adjustf_navg,
                                          hmin=hmin, hmax=hmax, hbinsize=hbinsize, 
                                          record_steps_timeseries=record_steps_timeseries, 
                                          record_steps_timeseries_every=record_steps_timeseries_every,
                                          seeds=seeds, single=single, record_histogram=True) 
        
        self._initialise()
        
    def run(self):
        try:
            self.mcrunner.run()
            self.displ_k_min, self.var_displ_k_min = self.mcrunner.histogram.get_mean_variance()
            mean_coord, var_coord = self.mcrunner.get_mean_variance_coordinate_vector()
            self.mean_coord_dist, self.var_coord_dist = np.linalg.norm(mean_coord-self.mcrunner.origin), np.sum(var_coord)
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)
    
    def _set_paths(self):
        dname = 'hypercube_n'+str(self.ndof)+'_l'+str(self.sidelength)
        self.base_directory = os.path.join(self.workspace,'explore_bv_'+dname)
        configfile = 'kmin_' + dname
        self.configfile = '{}/{}.config'.format(self.base_directory,configfile)
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
        for key, value in self.mc_params.iteritems() :
            f.write('{}: {}\n'.format(key,value))
    
    def _print_results_once(self, fname):
        """
        note that self.displ_k_min *= 1.5 to account for the limited computation time, 
        this is just an approximation 
        """
        f = open(fname,'a')
        f.write('[KMIN_MCRUNNER_STATUS]\n')
        status = self.mcrunner.get_status()
        for key, value in status.iteritems() :
            f.write('{}: {}\n'.format(key,value))
        f.write('[KMIN]\n')
        f.write('displ_k_min: {:.16f}\n'.format(self.displ_k_min))
        f.write('var_displ_k_min: {:.16f}\n'.format(self.var_displ_k_min))
        f.write('mean_coord_dist: {:.16f}\n'.format(self.mean_coord_dist))
        f.write('var_coord_dist: {:.16f}\n'.format(self.var_coord_dist))
        f.close()
        try:
            self._dump_trajectory()
        except:
            view_traceback()
    
    def _dump_trajectory(self):
        path = os.path.join(self.base_directory, 'kmin_trajectory.h5')
        traj = self.mcrunner.get_trajectory()
        write_2d_array_to_hf5(traj, 'trajectory', path)
    
    def _dump_diffusion_timeseries(self):
        fname = "{0}/StepsTimeSeries.{1}".format(self.diffusion_dir, int(self.mc_params['niter']))
        print "fname", fname
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
    
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    
    ndof = 93
    sim = _hypercube_kmin_mcrunner(ndof, sidelength=1, niter=1e8, k=0, seeds=seeds,
                         single=True, verbose=True, hmax=15, hbinsize=0.001)
    #record_steps_timeseries=True, record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(22)],)
    print 'simulation started'
    start=time.time()
    sim.run()
    end=time.time()
    print 'time elapsed', end-start
    status = sim.mcrunner.get_status()
    print status
    print 'd2 kmin: ',sim.displ_k_min
    print 'var: ',sim.var_displ_k_min
    print 'mean_coord_dist: ',sim.mean_coord_dist
    print 'var_coord_dist: ', sim.var_coord_dist
    traj = sim.mcrunner.get_trajectory()
    print np.shape(traj)
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
    mean_traj = np.mean(traj, axis=0)
    scatter_matrix = np.zeros((mean_traj.size, mean_traj.size))
    for x in traj:
        scatter_matrix += np.outer(x-mean_traj, x-mean_traj)
    cov_mat = np.cov([traj[0,:], traj[1,:], traj[2,:]])
    print scatter_matrix
    print cov_mat
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    n, bins, patches = plt.hist(eig_val_sc, 50, facecolor='green', alpha=0.75)
    plt.show()
    
              
                
                
                
