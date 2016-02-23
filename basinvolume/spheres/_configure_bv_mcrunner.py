from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.spheres import BV_MCrunner, _configure_mcrunner
from basinvolume.utils import *
import ConfigParser
import time
import cPickle as pickle

class configure_bv_mcrunner(_configure_mcrunner):
    """
    this is an abstract class that implements the basic components of a configure bv_mcrunner class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    *dtol: tolerance on the rms displacement of the minimised structure with respect to the origin coordinates
    """
    
    def __init__(self, rank, nprocs):
        self.rank = rank
        self.nprocs = nprocs
    
    def __call__(self, fname, k=1.0, temperature=1.0, stepsize=1e-1, niter=2e4, dtol=1e-4, eps=1., hmin=0, 
                 hmax=100, hbinsize=1, acceptance=0.2, adjustf=0.9, adjustf_niter = 5e3, adjustf_navg = 100, 
                 pt_eq_niter=0, ts_niter=None, ts_freq=1, opt_dtmax=1, opt_maxstep=None, 
                 opt_tol=1e-5, opt_nsteps=1e5, perform_convergence_test=False, collect_minima_list=False, 
                 single=False, seeds=None, use_cell_lists=False, use_cgd=False, record_histogram = False,
                 packings_dir='jammed_packings', base_dir=None, verbose = False,
                 opt_pot_str='hs_wca', **extra_pot_kwargs):
                
        self.fname = fname
        self._set_paths(base_dir, packings_dir)
        self._import_packing_config_files()
        self._import_packing_configuration()
        hbinsize = self._get_histogram_bin(k)
        opt_maxstep = self._get_opt_maxstep(opt_maxstep)
                
        #set parameters
        #self.mc_params = dict(k=k, temperature=temperature, )
        kwargs = dict(k=k, dtol=dtol, eps=eps, hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                      acceptance=acceptance, adjustf=adjustf, adjustf_niter=adjustf_niter, adjustf_navg=adjustf_navg,
                      pt_eq_niter=pt_eq_niter, ts_niter=ts_niter, ts_freq=ts_freq,
                      opt_dtmax=opt_dtmax, opt_maxstep=opt_maxstep, opt_tol=opt_tol, opt_nsteps=opt_nsteps,
                      perform_convergence_test=perform_convergence_test, record_histogram=record_histogram,
                      collect_minima_list=collect_minima_list, seeds=seeds, use_cell_lists=use_cell_lists,
                      single=single, use_periodic=True, use_frozen=False, use_cgd=use_cgd, record_trajectory=False,
                      opt_pot_str=opt_pot_str)
        kwargs.update(extra_pot_kwargs)

        self.mc_params = dict(temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)
        #add seeds dictionary to mc_params
        try:
            self.mc_params.update(seeds)
        except:
            print "WARNING:seeds not passed"
        
        self._initialise()
        self._requench_coords(dtol, opt_maxstep, verbose)
        
        #construct mcrunner
        #self.coords is origin, set initial configuration and origin to be the same
        #harmonic potential with fixed centre of mass
        potential = Harmonic(self.coords, k, bdim=self.bdim, com=True)
        mcrunner = BV_MCrunner(potential, self.coords, temperature, stepsize, niter, self.coords,
                               self.hs_radii, self.boxv, self.sca, rattlers=self.rattlers, **kwargs)
        
        return mcrunner 
    
    def _set_paths(self, base_dir, packings_dir):
        """
        set base_directory, packings_directory and configpaths, configfile
        """
        dname = self.fname
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        
        if base_dir is None:
            base_directory = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
            assert(os.path.exists(base_directory))
        else:
            if not os.path.isabs(base_dir):
                base_directory = os.path.join(os.getcwd(),packings_dir)
        self.base_directory = base_directory
        
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        
        self.packing_configpath = os.path.join(packings_dir,'{}.config'.format(dname))
        self.findk_configpath = os.path.join(self.base_directory,'findk_'+dname+'.config')  
        self.kmin_configpath = os.path.join(self.base_directory,'kmin_'+dname+'.config')
        self.configfile = '{}/explore_{}.config'.format(self.base_directory, dname)
    
    def _get_histogram_bin(self, k):
        """automatically estimate size of histogram"""
        hmax = self.displ_k_min*k #self.displ_k_max*self.kmax
        hbinsize= hmax * 0.0001 
        return hbinsize
    
    def _initialise(self):
        """initialisation function"""
        #change directory only at the end of initialise
        self._print_initialise()
        os.chdir(self.base_directory)
        
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        if self.rank == 0:
            self._print_parameters()
    
    def _write_sim_params(self, f):
        """
        write simulation parameters
        """    
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Explore_Jammed_Packings wrapper class input parameters\n')
        f.write('[IMPORTED_JAMMED_PACKING]\n')
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
        f.write('[MCRUNNER]\n')
        for key, value in self.mc_params.iteritems() :
            f.write('{}: {}\n'.format(key,value))
        f.write('[STATUS]\n')
        for i in xrange(self.nprocs):
            f.write('success_rank{}: {}\n'.format(str(i), "False"))
    
    def _import_packing_config_files(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.packing_configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('JAMMED_PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('JAMMED_PACKING','packing_fraction')
        self.sca = configf.getfloat('JAMMED_PACKING','sca')
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK','kmax')
        self.prob_kmax = configf.getfloat('FINDK','prob')
        self.displ_k_max = configf.getfloat('FINDK','displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK','var_displ_k_max')
        configf.read(str(self.kmin_configpath))
        self.displ_k_min = configf.getfloat('KMIN','displ_k_min')
        self.var_displ_k_min = configf.getfloat('KMIN','var_displ_k_min')
    
    def print_success_all(self, success):
        """
        print whether calculation has completed successfully
        """
        assert(hasattr(self, 'configfile'))
        if self.rank == 0:
            configf = ConfigParser.ConfigParser()
            configf.read(str(self.configfile))
            for i in xrange(self.nprocs):
                configf.set('STATUS', 'success_rank{}'.format(str(i)), success)
            configf.write(open(str(self.configfile),'w'))        
    
if __name__ == "__main__":
        
    #first 7 primary pseudo perfect numbers
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    
    sim = configure_bv_mcrunner(0, 1)
    extra_pot_kwargs = dict(pow=3, a=1)
    opt_pot_str = 'hs_wca' #'inverse_power_stillinger'
    mcrunner = sim('jammed_packing0.xyzdr', seeds=seeds, use_cell_lists=True, verbose=True,
                   opt_pot_str=opt_pot_str, **extra_pot_kwargs)
    print 'simulation started'
    start=time.time()
    mcrunner.run()
    end=time.time()
    print 'time elapsed', end-start
    status = mcrunner.get_status()
    print status
    mcrunner.dump_minima_list('minima_list.db')
    mcrunner.show_histogram()
    
        
                
            
              
                
                
                
