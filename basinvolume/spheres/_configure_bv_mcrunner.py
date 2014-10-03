from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic, HS_WCA, HS_WCAPeriodicCellLists
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.spheres import BV_MCrunner
from basinvolume.utils import *
import ConfigParser
import time
import cPickle as pickle

class configure_bv_mcrunner(object):
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
        
    def __call__(self, fname, k=1.0, temperature=1.0, stepsize=1e-1, niter=2e4, dtol=1e-4, eps=1., hmin=0, 
                 hmax=100, hbinsize=1, acceptance=0.2, adjustf=0.9, adjustf_niter = 5e3, adjustf_navg = 100, 
                 pt_eq_niter=0, ts_niter=None, ts_freq=1, opt_dtmax=1, opt_maxstep=None, 
                 opt_tol=1e-7, opt_nsteps=1e5, perform_convergence_test=False, collect_minima_list=False, 
                 seeds=None, use_cell_lists=False, record_histogram = False, 
                 packings_dir='jammed_packings', base_dir=None, verbose = False):
        
        self.fname = fname
        dname = fname
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
        
        self.packing_configpath = os.path.join(packings_dir,'jammed_packings.config')
        self.findk_configpath = os.path.join(self.base_directory,'findk_'+dname+'.config')  
        self.kmin_configpath = os.path.join(self.base_directory,'kmin_'+dname+'.config')
        
        self._import_config_files()
        self._import_packing_configuration()
        
        #automatically estimate size of histogram
        hmax = self.displ_k_min*k #self.displ_k_max*self.kmax
        hbinsize= hmax * 0.0001 
        
        #automatically set opt max step
        if opt_maxstep is None:
            opt_maxstep = self.boxv[0]*0.1
        
        #self.mc_params = dict(k=k, temperature=temperature, )
        self.eps = eps
        self.mc_params = {'k':k,'temperature':temperature,'niter':niter,'stepsize':stepsize,'dtol':dtol,'eps':self.eps,
                          'hmin':hmin,'hmax':hmax,'hbinsize':hbinsize,'acceptance':acceptance,'adjustf':adjustf,
                          'adjustf_niter':adjustf_niter,'adjustf_navg':adjustf_navg,'pt_eq_niter':pt_eq_niter,
                          'ts_niter':ts_niter, 'ts_freq':ts_freq,'opt_dtmax':opt_dtmax,'opt_maxstep':opt_maxstep,
                          'opt_tol':opt_tol,'opt_nsteps':opt_nsteps,'perform_convergence_test':perform_convergence_test, 
                          'collect_minima_list':collect_minima_list, 'record_histogram':record_histogram}
        
        #add seeds dictionary to mc_params
        try:
            self.mc_params.update(seeds)
        except:
            print "WARNING:seeds not passed"
        
        self._initialise()
        
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
        potential = Harmonic(self.coords, k, bdim=self.bdim,com=True)
        mcrunner = BV_MCrunner(potential, self.coords, temperature, stepsize, niter, self.coords, self.hs_radii, self.boxv, self.sca,
                               rattlers=self.rattlers, k=k, dtol=dtol, eps=eps, hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                               acceptance=acceptance, adjustf=adjustf, adjustf_niter = adjustf_niter, adjustf_navg = adjustf_navg, 
                               pt_eq_niter=pt_eq_niter, ts_niter=ts_niter, ts_freq=ts_freq,
                               opt_dtmax=opt_dtmax, opt_maxstep=opt_maxstep, opt_tol=opt_tol, opt_nsteps=opt_nsteps,
                               perform_convergence_test=perform_convergence_test, record_histogram=record_histogram, 
                               collect_minima_list=collect_minima_list, seeds=seeds, use_cell_lists=use_cell_lists)
        
        return mcrunner 
        
    def _initialise(self):
        """initialisation function"""
        #change directory only at the end of initialise
        self._print_initialise()
        os.chdir(self.base_directory)
        
    def _import_config_files(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.packing_configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert(self.bdim==2 or self.bdim==3) #currently PBC only implemented for 3d case
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
        dname = self.fname 
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        fname = '{}/explore_{}.config'.format(self.base_directory,dname)
        f = open(fname,'w')
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
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version())) 
        f.close()
    
if __name__ == "__main__":
    
    
    #first 7 primary pseudo perfect numbers
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    
    sim = configure_bv_mcrunner()
    mcrunner = sim('jammed_packing0.xyzdr', seeds=seeds, use_cell_lists=True, verbose=True)
    print 'simulation started'
    start=time.time()
    mcrunner.run()
    end=time.time()
    print 'time elapsed', end-start
    status = mcrunner.get_status()
    print status
    mcrunner.dump_minima_list('minima_list.db')
    mcrunner.show_histogram()
    
        
                
            
              
                
                
                
