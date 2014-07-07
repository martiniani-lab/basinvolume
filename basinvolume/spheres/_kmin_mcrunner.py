from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic, HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.spheres import BV_MCrunner
from basinvolume.utils import *
import ConfigParser
import time

class _kmin_mcrunner(object):
    """
    this is an abstract class that implements the basic components of a k0_mcrunner class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    *dtol: tolerance on the rms displacement of the minimised structure with respect to the origin coordinates
    """
        
    def __init__(self, fname, k=0.0, stepsize=1e-2, niter=5e4, dtol=1e-4, eps=1., hmin=0, 
                 hmax=10, hbinsize=0.1, acceptance=0.2, adjustf=0.9, adjustf_niter = 5e3, adjustf_navg = 100, 
                 opt_dtmax=1, opt_maxstep=None, opt_tol=1e-3, opt_nsteps=1e4, packings_dir='jammed_packings', verbose=False):
        dname = fname
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        self.base_directory = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        self.configpath = os.path.join(packings_dir,'jammed_packings.config')
        self.fname = fname
        #self.mc_params = dict(k=k, temperature=temperature, )
        self.temperature=1.0
        self.eps = eps
        
        self._import_packing_config_file()
        self._import_packing_configuration()
        
        #automatically set opt max step
        if opt_maxstep is None:
            opt_maxstep = self.boxv[0]*0.1
            
        self.mc_params = {'k':k,'temperature':self.temperature,'niter':niter,'stepsize':stepsize,'dtol':dtol,'eps':eps,'hmin':hmin,'hmax':hmax,
                      'hbinsize':hbinsize,'acceptance':acceptance,'adjustf':adjustf,'adjustf_niter':adjustf_niter,'adjustf_navg':adjustf_navg,
                      'opt_dtmax':opt_dtmax,'opt_maxstep':opt_maxstep,'opt_tol':opt_tol,'opt_nsteps':opt_nsteps}
                    
        #re-quench origin to avoid rounding errors
        pot_optimizer = HS_WCA(self.eps, self.sca, self.hs_radii, boxvec=self.boxv)
        res = modifiedfire_cpp(self.coords, pot_optimizer, maxstep=(self.boxv[0]*0.1), nsteps=1e6, tol=1e-9)
        if not res.success:
            assert(False)
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
        
        #construct mcrunner
        #self.coords is origin, set initial configuration and origin to be the same
        #harmonic potential with fixed centre of mass
        
        potential = Harmonic(self.coords, k, bdim=self.bdim, com=True)
        self.mcrunner = BV_MCrunner(potential, self.coords, self.temperature, stepsize, niter, self.coords, self.hs_radii, self.boxv, self.sca,
                               rattlers=self.rattlers, k=k, dtol=dtol, eps=eps, hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                               acceptance=acceptance, adjustf=adjustf, adjustf_niter = adjustf_niter, adjustf_navg = adjustf_navg, 
                               opt_dtmax=opt_dtmax, opt_maxstep=opt_maxstep, opt_tol=opt_tol, opt_nsteps=opt_nsteps) 
        
        self._print_initialise()
        
    def run(self):
        self.mcrunner.run()
        self.displ_k_min, self.var_displ_k_min = self.mcrunner.histogram.get_mean_variance()
        self._print_results()
        
    def _import_packing_config_file(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert(self.bdim==2 or self.bdim==3) #currently PBC only implemented for 2d and 3d case
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('JAMMED_PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('JAMMED_PACKING','packing_fraction')
        self.sca = configf.getfloat('JAMMED_PACKING','sca')
        
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
        
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        self._print_parameters()
    
    def _print_parameters(self):
        """writes the simulation parameters"""
        dname = 'kmin_' + self.fname 
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        fname = '{}/{}.config'.format(self.base_directory,dname)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Explore_Jammed_Packings wrapper class input parameters\n')
        f.write('[KMIN_IMPORTED_JAMMED_PACKING]\n')
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {}\n'.format(self.imp_packing_frac))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndim))
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{} '.format(val))
        f.write('\n')
        assert(self.sca >0)
        f.write('sca: {}\n'.format(self.sca))
        f.write('[KMIN_MCRUNNER]\n')
        for key, value in self.mc_params.iteritems() :
            f.write('{}: {}\n'.format(key,value)) 
        f.close()
    
    def _print_results(self):
        dname = 'kmin_' + self.fname
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        fname = '{}/{}.config'.format(self.base_directory,dname)
        f = open(fname,'a')
        f.write('[KMIN_MCRUNNER_STATUS]\n')
        status = self.mcrunner.get_status()
        for key, value in status.iteritems() :
            f.write('{}: {}\n'.format(key,value))
        f.write('[KMIN]\n')
        f.write('displ_k_min: {}\n'.format(self.displ_k_min))
        f.write('var_displ_k_min: {}\n'.format(self.var_displ_k_min))
        f.close()
    
if __name__ == "__main__":
    
    sim = _kmin_mcrunner('jammed_packing0.xyzdr', verbose=True)
    print 'simulation started'
    start=time.time()
    #pickle.dump(sim, open('testpickle.pickle',"wb"), pickle.HIGHEST_PROTOCOL)
    #sim = pickle.load(open('testpickle.pickle', "rb"))
    #mcrunner = sim('jammed_packing0.xyzdr')
    sim.run()
    end=time.time()
    print end-start
    status = sim.mcrunner.get_status()
    print status
    print 'd2 kmin: ',sim.displ_k_min
    print 'var: ',sim.var_displ_k_min
    sim.mcrunner.show_histogram()
    
    
        
                
            
              
                
                
                
