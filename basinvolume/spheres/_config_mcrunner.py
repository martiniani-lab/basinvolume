from __future__ import division
import numpy as np
import os
import abc
from pele.potentials import HS_WCA, InversePowerStillinger
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import get_git_version, get_python_version, get_cython_version, full_coordinates
from basinvolume.utils import read_xydfr, read_xyzdfr, reduce_coordinates, read_xydr, read_xyzdr
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
            #opt_maxstep = self.boxv[0] * 0.01
            opt_maxstep = self.sca * np.amin(self.red_radii)
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
        f = open(fname, 'a')
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
        f = open(fname, 'w')
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
    
    def _requench_coords(self, dtol, opt_maxstep, verbose, gtol=1e-7, frozen=False,
                         opt_pot_str='hs_wca', **extra_pot_kwargs):
        """re-quench origin to avoid rounding errors"""
        quench = lambda red_coords, pot_optmizer : modifiedfire_cpp(red_coords, pot_optimizer,
                                                                    maxstep=opt_maxstep, nsteps=1e6,
                                                                    tol=gtol)
        if opt_pot_str.lower() == 'hs_wca':
            if frozen:
                pot_optimizer = HS_WCA(use_periodic=False, reference_coords=self.coords, eps=self.eps,
                                       sca=self.sca, radii=self.hs_radii, use_frozen=True,
                                       frozen_atoms=self.frozen, ndim=self.bdim)
                res = quench(self.red_coords, pot_optimizer)
                new_coords = full_coordinates(res.coords, self.coords, self.frozen, self.bdim)
                self.red_coords = np.array(res.coords)
            else:
                pot_optimizer = HS_WCA(use_periodic=True, eps=self.eps, sca=self.sca, radii=self.hs_radii,
                                       ndim=self.bdim, boxvec=self.boxv)
                res = quench(self.red_coords, pot_optimizer)
                new_coords = res.coords
        elif opt_pot_str.lower() == 'inverse_power_stillinger':
            pow = extra_pot_kwargs['pow']
            a = extra_pot_kwargs['a']
            pot_optimizer = InversePowerStillinger(pow, a=a, ndim=self.bdim, boxvec=self.boxv)
            res = quench(self.red_coords, pot_optimizer)
            new_coords = res.coords
        else:
            raise NotImplementedError

        if not res.success:
            assert(False)
        elif res.nfev > 1:
            warnings.warn('Configuration has moved on re-quenching, this should not happen')
             
        drms = np.sqrt(np.dot(self.coords - new_coords, self.coords - new_coords) / self.ndim)
        assert(drms <= dtol)
        self.coords = np.array(new_coords)
        
        if verbose:
            print 'results from quench \n'
            print res
            hess = pot_optimizer.getHessian(res.coords)
            w, v = np.linalg.eig(hess)
            w = np.real(w)
            print 'eigenvalues'
            print sorted(w)
    
    @abc.abstractmethod 
    def _import_packing_config_files(self):
        """import packings configuration file"""
        
    def _import_packing_configuration(self, frozen=False):
        """imports the coordinates, data relative to the shape of the particles and
        whether the particles are rattlers or not. Note that self.rattlers returned 
        here is of size self.ndim but in generate_jammed_packings is of size self.nparticles.
        This should be run in initialise()
        """
        path = os.path.join(self.packings_dir,self.fname)
        if frozen:
            if self.bdim == 2:
                self.coords, hs_diameters, self.frozen, self.rattlers = read_xydfr(path)
            elif self.bdim == 3:
                self.coords, hs_diameters, self.frozen, self.rattlers = read_xyzdfr(path)
            else:
                raise NotImplementedError("bdim={} not implemented".format(self.bdim))
            self.hs_radii = np.array(hs_diameters / 2)
            self.red_coords = reduce_coordinates(self.coords, self.frozen, self.bdim)
            self.red_radii = np.delete(self.hs_radii.copy(), self.frozen)
            self.red_rattlers = reduce_coordinates(self.rattlers, self.frozen, self.bdim)
        else:
            if self.bdim == 2:
                self.coords, hs_diameters, self.rattlers = read_xydr(path)
            elif self.bdim == 3:
                self.coords, hs_diameters, self.rattlers = read_xyzdr(path)
            else:
                raise NotImplementedError("bdim={} not implemented".format(self.bdim))
            self.hs_radii = hs_diameters / 2.0
            self.red_coords = self.coords
            self.red_radii = self.hs_radii
            self.red_rattlers = self.rattlers
                
            
              
                
                
                
