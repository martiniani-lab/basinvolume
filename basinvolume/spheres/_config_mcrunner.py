from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic, HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import *
import warnings

class _configure_mcrunner(object):
    """
    this is an abstract class that implements the basic components of a _configure_mcrunner class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    """
    __metaclass__ = abc.ABCMeta
     
    @abc.abstractmethod
    def _set_paths(self, *args, **kwargs):
        """
        set base_directory, packings_directory and configpaths
        """
    
    def _get_opt_maxstep(self, opt_maxstep):
        """returns opt max step"""
        if opt_maxstep is None:
            opt_maxstep = self.boxv[0]*0.01
        return opt_maxstep
    
    @abc.abstractmethod
    def _initialise(self):
        """initialisation function"""
    
    @abc.abstractmethod
    def _print_initialise(self):
        """"print initialise"""
    
    def _print_success(self, success):
        """
        print whether calculation has completed successfully
        """
        assert(hasattr(self, 'configfile'))
        fname = self.configfile
        f = open(fname,'a')
        f.write('[STATUS]\n')
        f.write('success: {}\n'.format(str(success)))
        f.close()
    
    def _print_parameters(self):
        """writes the simulation parameters"""
        f = self._open_param_stream()
        self._write_sim_params(f)
        self._write_code_version(f)
        f.close()
    
    def _open_param_stream(self):
        """
        returns a stream where to write the parameters
        """
        assert(hasattr(self, 'configfile'))
        fname = self.configfile
        f = open(fname,'w')
        return f
    
    @abc.abstractmethod
    def _write_sim_params(self, f):
        """
        write simulation parameters
        """
    
    def _write_code_version(self, f):
        """print software version"""
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
    
    def _requench_coords(self, dtol, opt_maxstep, verbose):
        """re-quench origin to avoid rounding errors"""
        pot_optimizer = HS_WCA(use_periodic=True, eps=self.eps, sca=self.sca, radii=self.hs_radii, ndim=self.bdim, boxvec=self.boxv)
        res = modifiedfire_cpp(self.coords, pot_optimizer, maxstep=opt_maxstep, nsteps=1e6, tol=1e-9)
        if not res.success:
            assert(False)
        elif res.nfev > 1:
            warnings.warn('Configuration has moved on re-quenching, this should not happen')
             
        drms= np.sqrt(np.dot(self.coords,self.coords)/self.ndim) - np.sqrt(np.dot(res.coords, res.coords)/self.ndim)
        assert(drms <= dtol)
        self.coords = res.coords
        
        if verbose:
            print 'results from quench \n'
            print res
            hess = pot_optimizer.getHessian(self.coords)
            w, v = np.linalg.eig(hess)
            w = np.real(w)
            print 'eigenvalues'
            print sorted(w)
    
    @abc.abstractmethod 
    def _import_packing_config_files(self):
        """import packings configuration file"""
        
    def _import_packing_configuration(self):
        """imports the coordinates, data relative to the shape of the particles and
        whether the particles are rattlers or not. Note that self.rattlers returned 
        here is of size self.ndim but in generate_jammed_packings is of size self.nparticles.
        This should be run in initialise()
        """
        path = os.path.join(self.packings_dir,self.fname)
        if self.bdim == 2:
            self.coords, hs_diameters, self.rattlers = read_xydr(path)
        else:
            self.coords, hs_diameters, self.rattlers = read_xyzdr(path)
        self.hs_radii = hs_diameters/2
                
            
              
                
                
                
