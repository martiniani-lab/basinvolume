from __future__ import division
import numpy as np
import os
from mcpele.monte_carlo import NullPotential
from basinvolume.spheres import _configure_mcrunner
from basinvolume.utils import *
from basinvolume.mbar_spheres import BVInnerSphereMCrunner
import ConfigParser
import time
import warnings

def _subtract_com(x, ndim=3):
    x = x.reshape(-1, ndim)
    com = x.mean(0)
    return (x - com[np.newaxis, :]).ravel()

class _config_innersphere_mcrunner(_configure_mcrunner):
    """this is a class that implements a mcrunner that samples the inner sphere of a basin
    
    when niter=None, niter is set equal to exact number of PT niter
    """
        
    def __init__(self, fname, niter=None, dtol=1e-4, eps=1., hmin=0, hmax=0.01, hbinsize=0.0005, 
                 opt_dtmax=1, opt_maxstep=None, opt_tol=1e-5, opt_nsteps=1e5,
                 perform_convergence_test=False, collect_minima_list=False,
                 seeds=None, use_cell_lists=False, use_cgd=False, record_histogram=False, 
                 packings_dir='jammed_packings', verbose=False, opt_pot_str='hs_wca',
                 **extra_pot_kwargs):
                
        self.fname = fname
        self.temperature=1.0
        self.eps = eps
        
        self._set_paths(packings_dir)
        self._import_packing_config_files()
        self._import_packing_configuration()
        self.k = 1.0 / self.u2_k0
        self.stepsize = 1./np.sqrt(self.k)
        self.ref_radius = self.stepsize / 2
        if niter is not None:
            self.niter = niter
        opt_maxstep = self._get_opt_maxstep(opt_maxstep)
        
        #self.mc_params = dict(k=k, temperature=temperature, )    
        kwargs = dict(dtol=dtol, eps=eps, hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                      opt_dtmax=opt_dtmax, opt_maxstep=opt_maxstep, opt_tol=opt_tol, opt_nsteps=opt_nsteps,
                      perform_convergence_test=perform_convergence_test, collect_minima_list=collect_minima_list,
                      seeds=seeds, use_cell_lists=use_cell_lists, record_histogram=record_histogram,
                      use_cgd=use_cgd, use_periodic=True, use_frozen=False, opt_pot_str=opt_pot_str,
                      **extra_pot_kwargs)

        self.mc_params = dict(k=self.k, temperature=self.temperature, niter=self.niter, stepsize=self.stepsize)
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")
        
        self._requench_coords(dtol, opt_maxstep, verbose, opt_pot_str=opt_pot_str, **extra_pot_kwargs)
        
        #construct mcrunner
        self.coords = _subtract_com(self.coords, ndim=self.bdim)
        potential = NullPotential()
        self.mcrunner_gaussian = BVInnerSphereMCrunner(potential, self.coords, self.temperature, self.stepsize, self.niter,
                                                       self.coords, self.hs_radii, self.boxv, self.sca,
                                                       rattlers=self.rattlers, gaussian_step=True, **kwargs)
        self.mcrunner_ballpick = BVInnerSphereMCrunner(potential, self.coords, self.temperature, self.ref_radius,
                                                       self.niter, self.coords, self.hs_radii, self.boxv, self.sca,
                                                       rattlers=self.rattlers, gaussian_step=False, **kwargs)
        self._initialise()

    def run(self):
        try:
            self.mcrunner_gaussian.run()
            self.mcrunner_ballpick.run()
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)
    
    def _set_paths(self, packings_dir):
        dname = self.fname
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        self.base_directory = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        self.configpath = os.path.join(packings_dir,'{}.config'.format(dname))
        self.findk_configpath = os.path.join(self.base_directory,'findk_'+dname+'.config')
        configfile = 'innersphere_' + dname
        self.configfile = '{}/{}.config'.format(self.base_directory,configfile)  
    
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
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK','kmax')
        self.prob_kmax = configf.getfloat('FINDK','prob')
        self.displ_k_max = configf.getfloat('FINDK','displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK','var_displ_k_max')
        #import mean displacement of replica with largest k
        path = os.path.join(self.base_directory, '0/hist_mean')
        fileHandle = open (path, "r")
        lineList = fileHandle.readlines()
        fileHandle.close()
        niter, u2, var, std_err = lineList[-1].split()
        self.u2_k0 = float(u2)
        self.var_k0 = float(var)
        self.niter = int(niter) + 1
    
    def _initialise(self):
        self._print_initialise()
         
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        self._print_parameters()
    
    def _write_sim_params(self, f):
        """
        write simulation parameters
        """
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Explore_Jammed_Packings wrapper class input parameters\n')
        f.write('[INNERSPHERE_IMPORTED_JAMMED_PACKING]\n')
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
        f.write('[INNERSPHERE_MCRUNNER]\n')
        for key, value in self.mc_params.iteritems() :
            f.write('{}: {}\n'.format(key,value))

    def _print_results(self):
        """
        note that self.displ_k_min *= 1.5 to account for the limited computation time, 
        this is just an approximation 
        """
        fname = self.configfile
        f = open(fname, 'a')
        f.write('[INNERSPHERE_GAUSSIAN_MCRUNNER_STATUS]\n')
        status = self.mcrunner_gaussian.get_status()
        for key, value in status.iteritems():
            f.write('{}: {:.16f}\n'.format(key, value))
        f.write('[INNERSPHERE_BALLPICK_MCRUNNER_STATUS]\n')
        status = self.mcrunner_ballpick.get_status()
        for key, value in status.iteritems():
            f.write('{}: {:.16f}\n'.format(key, value))
        f.close()
        path = os.path.join(self.base_directory, "inner_sphere.timeseries")
        self.mcrunner_gaussian.dump_timeseries(path, clear=False)
    
if __name__ == "__main__":
    
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    
    sim = _config_innersphere_mcrunner('jammed_packing1.xyzdr', niter=1e5, opt_tol=1e-4, seeds=seeds, 
                                use_cell_lists=False, verbose=False, use_cgd=True, opt_nsteps=1e5)
    print 'simulation started'
    start = time.time()
    sim.run()
    end = time.time()
    print 'time elapsed', end - start
    print "gaussian"
    status = sim.mcrunner_gaussian.get_status()
    print status
    print 'stepsize: ', sim.mcrunner_gaussian.get_stepsize()
    print "ballpick"
    status = sim.mcrunner_ballpick.get_status()
    print status
    print 'stepsize: ', sim.mcrunner_ballpick.get_stepsize()
    sim.mcrunner_gaussian.show_histogram_analytical()
    
    
        
                
            
              
                
                
                
