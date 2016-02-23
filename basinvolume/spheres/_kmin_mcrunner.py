from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic, HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.spheres import BV_MCrunner, _configure_mcrunner
from basinvolume.utils import *
import ConfigParser
import time

class _kmin_mcrunner(_configure_mcrunner):
    """
    this is a class that implements a kmin_mcrunner class
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    *dtol: tolerance on the rms displacement of the minimised structure with respect to the origin coordinates
    *print_diffusion_only: bool
        print diffusion data only and none of the other configuration files
    *base_directory is the path to explore_bv_* folders, set by default as cwd/explore_bv_*
    """
        
    def __init__(self, fname, k=0.0, stepsize=1e-2, niter=5e4, dtol=1e-4, eps=1., hmin=0, 
                 hmax=0.01, hbinsize=0.0005, acceptance=0.2, adjustf=0.9, adjustf_niter = 5e3, 
                 adjustf_navg = 100, opt_dtmax=1, opt_maxstep=None, opt_tol=1e-5, opt_nsteps=1e5,
                 record_steps_timeseries=False, record_steps_timeseries_every=[1], print_diffusion_only=False,
                 record_trajectory=True, record_trajectory_npoints=1e4,
                 perform_convergence_test=False, collect_minima_list=False, single=False, 
                 seeds=None, use_cell_lists=False, use_cgd=False, opt_pot_str='hs_wca',
                 packings_dir='jammed_packings', verbose=False, workspace=None, **extra_pot_kwargs):
                
        self.fname = fname
        self.temperature=1.0
        self.eps = eps
        self.print_diffusion_only = print_diffusion_only
        self.record_steps_timeseries = record_steps_timeseries
        if workspace is None:
            self.workspace = os.getcwd()
        else:
            self.workspace = os.path.abspath(workspace)
        
        self._set_paths(packings_dir)
        self._import_packing_config_files()
        self._import_packing_configuration()
        opt_maxstep = self._get_opt_maxstep(opt_maxstep)
        
        #self.mc_params = dict(k=k, temperature=temperature, )
        kwargs = dict(k=k, dtol=dtol, eps=eps, hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                      acceptance=acceptance, adjustf=adjustf, adjustf_niter=adjustf_niter,
                      adjustf_navg=adjustf_navg, opt_dtmax=opt_dtmax, opt_maxstep=opt_maxstep,
                      opt_tol=opt_tol, opt_nsteps=opt_nsteps,
                      record_steps_timeseries=record_steps_timeseries,
                      record_steps_timeseries_every=record_steps_timeseries_every,
                      record_trajectory=record_trajectory, record_trajectory_npoints=record_trajectory_npoints,
                      perform_convergence_test=perform_convergence_test, collect_minima_list=collect_minima_list,
                      seeds=seeds, use_cell_lists=use_cell_lists, record_histogram=True, single=single,
                      use_cgd=use_cgd, use_periodic=True, use_frozen=False, opt_pot_str=opt_pot_str,
                      **extra_pot_kwargs)

        self.mc_params = dict(temperature=self.temperature,niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)

        # add seeds dictionary to mc_params
        try:
            self.mc_params.update(seeds)
        except:
            print "WARNING:seeds not passed"
        
        self._requench_coords(dtol, opt_maxstep, verbose, opt_pot_str=opt_pot_str, **extra_pot_kwargs)
        
        # construct mcrunner
        # self.coords is origin, set initial configuration and origin to be the same
        # harmonic potential with fixed centre of mass
        potential = Harmonic(self.coords, k, bdim=self.bdim, com=True)
        self.mcrunner = BV_MCrunner(potential, self.coords, self.temperature, stepsize, niter, self.coords, 
                                    self.hs_radii, self.boxv, self.sca, rattlers=self.rattlers, **kwargs)
        
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
    
    def _collect_trajectory(self, fix_com=True):
        mean_coord, var_coord = self.mcrunner.get_mean_variance_coordinate_vector()
        self.mean_coord_dist, self.var_coord_dist = get_dist_com(mean_coord, self.mcrunner.origin, self.bdim), np.sum(var_coord)
        self.trajectory = self.mcrunner.dump_trajectory(self.trajectory_path, clear=True)
        if fix_com:
            for i,coords in enumerate(self.trajectory):
                self.trajectory[i] = get_dist_vec_com(coords, self.mcrunner.origin, self.mcrunner.bdim)        
        self.traj_eval, self.traj_evec = trajectory_pca(self.trajectory)
        self.pca_asphericity = asphericity_factor(self.traj_eval)
    
    def _set_paths(self, packings_dir):
        dname = self.fname
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        self.base_directory = os.path.join(self.workspace,'explore_bv_'+str(dname))
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(self.workspace,packings_dir)
        self.packings_dir = packings_dir
        self.configpath = os.path.join(packings_dir,'{}.config'.format(dname))
        self.findk_configpath = os.path.join(self.base_directory,'findk_'+dname+'.config')
        configfile = 'kmin_' + dname
        self.configfile = '{}/{}.config'.format(self.base_directory,configfile)
        trajectory_fname = 'kmin_trajectory_' + dname
        self.trajectory_path = '{}/{}.h5'.format(self.base_directory, trajectory_fname)
        self.diffusion_dir = os.path.join(self.base_directory, "diffusion")
        diffusion_configfname = 'diffusion_' + dname
        self.diffusion_configfname = '{}/{}'.format(self.diffusion_dir, diffusion_configfname)
    
    def _import_packing_config_files(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('JAMMED_PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('JAMMED_PACKING','packing_fraction')
        self.sca = configf.getfloat('JAMMED_PACKING','sca')
    
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
        f.write('[KMIN_IMPORTED_JAMMED_PACKING]\n')
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {}\n'.format(self.imp_packing_frac))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndim))
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        assert(self.sca >0)
        f.write('sca: {:.16f}\n'.format(self.sca))
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
        f.write('displ_k_min: {:.16f}\n'.format(self.displ_k_min * 1.25)) #note 1.25
        f.write('var_displ_k_min: {:.16f}\n'.format(self.var_displ_k_min))
        f.write('mean_coord_dist: {:.16f}\n'.format(self.mean_coord_dist))
        f.write('var_coord_dist: {:.16f}\n'.format(self.var_coord_dist))
        f.write('pca_asphericity: {:.16f}\n'.format(self.pca_asphericity))
        f.close()
    
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
    
if __name__ == "__main__":
    
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    extra_pot_kwargs = dict(pow=3, a=1)
    opt_pot_str = 'hs_wca' #'inverse_power_stillinger'
    sim = _kmin_mcrunner('jammed_packing0.xydr', niter=1e4, k=0, opt_tol=1e-4, seeds=seeds,
                         record_steps_timeseries=True, record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(22)],
                         single=True, use_cell_lists=True, verbose=True, use_cgd=True,
                         hmax=20, hbinsize=0.05, opt_nsteps=1e6,
                         opt_pot_str=opt_pot_str, **extra_pot_kwargs)
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
    #sim.mcrunner.show_histogram_kmax()
    
    
        
                
            
              
                
                
                
