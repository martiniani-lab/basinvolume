from __future__ import division

import numpy as np
import os
import time
import warnings
import os
import ConfigParser
import numpy as np

from basinvolume.geometry.point_sampling import HyperElemFindkMCrunner
from basinvolume.spheres import _configure_mcrunner
from basinvolume.utils import trymakedir
from basinvolume.utils import view_traceback
from mcpele.monte_carlo import NullPotential
from basinvolume.geometry.point_sampling.utils import _append_geom_params


class _hyperelem_findk_mcrunner(_configure_mcrunner):
    """
    """
        
    def __init__(self, ndof, geometry="cube", geom_params=[1.], k=100.0, niter=1e6, avgcount=1e4,
                 ktarget=0.9, knavg=1000, ktol=0.025, hmin=0, hmax=0.01, 
                 hbinsize=0.0005, seeds=None, verbose=False, workspace=None):
                
        self.temperature=1.0
        self.ndof = ndof
        self.geometry = geometry
        self.geom_params = geom_params
        self.coords = np.zeros(self.ndof)
        if workspace is None:
            self.workspace = os.getcwd()
        else:
            self.workspace = os.path.abspath(workspace)
        
        self._set_paths()
        stepsize = np.sqrt(1.0 / k) #stepsize plays the role of the standard deviation
        
        #self.mc_params = dict(k=k, temperature=temperature, )    
        kwargs = dict(geometry=geometry, geom_params=self.geom_params, ktarget=ktarget,
                      knavg=knavg, ktol=ktol, avgcount=avgcount,
                      hmin=hmin, hmax=hmax, hbinsize=hbinsize, seeds=seeds)

        self.mc_params = dict(k=k, temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")
                
        #self.coords is origin, set initial configuration and origin to be the same
        potential = NullPotential()
        #####       
        self.mcrunner = HyperElemFindkMCrunner(potential, self.coords, self.temperature, stepsize, niter,
                                               self.coords, **kwargs)
        
        self._initialise()
        
    def run(self):
        try:
            self.mcrunner.run()
            self.kmax = self.mcrunner.get_k()
            self.prob = self.mcrunner.findk.get_prob()
            self.displ_k_max, self.var_displ_k_max = self.mcrunner.findk.get_mean_variance()
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)
    
    def _set_paths(self):
        dname = 'hyper{}_n'.format(self.geometry) + str(self.ndof)
        dname = _append_geom_params(dname, self.geometry, self.geom_params)
        self.base_directory = os.path.join(self.workspace,'explore_bv_'+dname)
        configfile = 'findk_' + dname
        self.configfile = '{}/{}.config'.format(self.base_directory,configfile)
    
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
        f.write('[FINDK_HYPERELEM]\n')
        f.write('ndof: {}\n'.format(self.ndof))
        f.write('geometry: {}\n'.format(self.geometry))
        f.write('geom_params: ')
        for val in self.geom_params:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        f.write('[FINDK_MCRUNNER]\n')
        for key, value in self.mc_params.iteritems() :
            f.write('{}: {}\n'.format(key,value))
    
    def _print_results(self):
        fname = self.configfile
        f = open(fname, 'a')
        f.write('[FINDK_MCRUNNER_STATUS]\n')
        status = self.mcrunner.get_status()
        for key, value in status.iteritems():
            f.write('{}: {}\n'.format(key, value))
        f.write('[FINDK]\n')
        f.write('kmax: {:.16f}\n'.format(self.kmax))
        f.write('prob: {:.16f}\n'.format(self.prob))
        f.write('displ_k_max: {:.16f}\n'.format(self.displ_k_max))
        f.write('var_displ_k_max: {:.16f}\n'.format(self.var_displ_k_max))
        f.close()
    
    def _import_packing_config_files(self):
        """pure virtual, must overload"""
        pass
    
if __name__ == "__main__":
    
    #sim = _findk_mcrunner('jammed_packing0.xydr')
    pppn = [2, 6, 42, 1806, 47058, 2214502422, 52495396602]
    seeds = dict(seed_takestep=1158925890, seed_oracle=pppn[2])
    ndof = 10
    sim = _hyperelem_findk_mcrunner(ndof, geometry="sphere_exp_decay",
                                    geom_params=[1., 0.1], avgcount=1e4, k=50,
                                    ktarget=0.9, knavg=1e3, seeds=seeds, verbose=True)
    print 'simulation started'
    start=time.time() 
    sim.run()
    end=time.time()
    print 'time elapsed', end-start
    status = sim.mcrunner.get_status()
    print status
    print "self.kmax:", sim.kmax
    print "self.prob:", sim.prob
    print "self.displ_k_max:", sim.displ_k_max
    print "self.var_displ_k_max:", sim.var_displ_k_max
    #print "Nd/k: ", sim.nparticles * sim.bdim / sim.kmax
    print "(N-1)d/k", sim.ndof / sim.kmax
    #sim.mcrunner.show_histogram()
    print "entries in histogram:", sim.mcrunner.get_entries()
    
