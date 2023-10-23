from __future__ import division
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import str
import time
import warnings
import os
import configparser
import numpy as np

from basinvolume.geometry.point_sampling import HyperElemInnerSphereMCrunner
from basinvolume.spheres import _configure_mcrunner
from basinvolume.utils import *
from mcpele.monte_carlo import NullPotential


class _oracle_hyperelem_innersphere_mcrunner(_configure_mcrunner):
    """this is a class that implements a mcrunner that samples the inner sphere of a basin
    
    when niter=None, niter is set equal to exact number of PT niter
    """
        
    def __init__(self, base_dir, niter=None, hmin=0, hmax=0.01, hbinsize=0.0005, 
                 seeds=None, record_histogram=False, verbose=False):
                
        self.temperature=1.0
        
        self._set_paths(base_dir)
        self._import_packing_config_files()
        self.k = 1.0 / self.u2_k0
        self.stepsize = 1./np.sqrt(self.k)
        self.ref_radius = self.stepsize / 2
        self.coords = np.zeros(self.ndof)
        if niter is not None:
            self.niter = niter
        
        #self.mc_params = dict(k=k, temperature=temperature, )    
        kwargs = dict(hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                      seeds=seeds, record_histogram=record_histogram,
                      geom_params=self.geom_params, geometry=self.geometry)

        self.mc_params = dict(k=self.k, temperature=self.temperature,
                              niter= self.niter, stepsize=self.stepsize)
        self.mc_params.update(kwargs)
        #add seeds dictionary to mc_params
        if seeds is None:
            warnings.warn("seeds not passed")
        
        #construct mcrunner
        potential = NullPotential()
        self.mcrunner_gaussian = HyperElemInnerSphereMCrunner(potential, self.coords, self.temperature, self.stepsize,
                                                              self.niter, self.coords, gaussian_step=True, **kwargs)
        self.mcrunner_ballpick = HyperElemInnerSphereMCrunner(potential, self.coords, self.temperature, self.ref_radius,
                                                              self.niter, self.coords, gaussian_step=False, **kwargs)
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
    
    def _set_paths(self, base_dir):
        """
        set base_directory, packings_directory and configpaths, configfile
        """
        assert 'oracle' in base_dir and 'hyper' in base_dir and 'explore_bv' in base_dir
        if not os.path.isabs(base_dir):
            base_directory = os.path.join(os.getcwd(), base_dir)
            assert (os.path.exists(base_directory))
        else:
            base_directory = base_dir
        self.base_directory = base_directory

        base_name = os.path.basename(os.path.normpath(self.base_directory))
        dname = str(base_name).replace('explore_bv_','')
        self.findk_configpath = os.path.join(self.base_directory,'findk_'+dname+'.config')  
        configfile = 'innersphere_' + dname
        self.configfile = '{}/{}.config'.format(self.base_directory,configfile)
    
    def _import_packing_config_files(self):
        configf = configparser.ConfigParser()
        configf.read(str(self.findk_configpath))
        self.ndof = configf.getfloat('FINDK_HYPERELEM','ndof')
        self.geometry = configf.get('FINDK_HYPERELEM', 'geometry')
        geom_params = configf.get('FINDK_HYPERELEM', 'geom_params')
        self.geom_params = np.array([float(x) for x in geom_params.split()])
        self.kmax = configf.getfloat('FINDK','kmax')
        self.prob_kmax = configf.getfloat('FINDK','prob')
        self.displ_k_max = configf.getfloat('FINDK','displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK','var_displ_k_max')
        #import mean displacement of replica with second largest k
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
        f.write('[INNERSPHERE_HYPERELEM]\n')
        f.write('ndof: {}\n'.format(self.ndof))
        f.write('geometry: {}\n'.format(self.geometry))
        f.write('geom_params: ')
        for val in self.geom_params:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        f.write('[INNERSPHERE_MCRUNNER]\n')
        for key, value in list(self.mc_params.items()) :
            f.write('{}: {}\n'.format(key,value))
    
    def _print_results(self):
        """
        note that self.displ_k_min *= 1.5 to account for the limited computation time, 
        this is just an approximation 
        """
        fname = self.configfile
        f = open(fname,'a')
        f.write('[INNERSPHERE_GAUSSIAN_MCRUNNER_STATUS]\n')
        status = self.mcrunner_gaussian.get_status()
        for key, value in list(status.items()) :
            f.write('{}: {:.16f}\n'.format(key,value))
        f.write('[INNERSPHERE_BALLPICK_MCRUNNER_STATUS]\n')
        status = self.mcrunner_ballpick.get_status()
        for key, value in list(status.items()):
            f.write('{}: {:.16f}\n'.format(key, value))
        f.close()
        path = os.path.join(self.base_directory, "inner_sphere.timeseries")
        self.mcrunner_gaussian.dump_timeseries(path, clear=False)
    
if __name__ == "__main__":
    
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1], seed_oracle=pppn[2])
    
    sim = _oracle_hyperelem_innersphere_mcrunner('explore_bv_oracle_hypersphere_n2_r1.0', niter=1e6,
                                                 seeds=seeds, verbose=False)
    print('simulation started')
    start=time.time()
    sim.run()
    end=time.time()
    print('time elapsed', end-start)
    print("gaussian")
    status = sim.mcrunner_gaussian.get_status()
    print(status)
    print('stepsize: ',sim.mcrunner_gaussian.get_stepsize())
    print("ballpick")
    status = sim.mcrunner_ballpick.get_status()
    print(status)
    print('stepsize: ', sim.mcrunner_ballpick.get_stepsize())
    sim.mcrunner_gaussian.show_histogram_analytical()
    
    
        
                
            
              
                
                
                
