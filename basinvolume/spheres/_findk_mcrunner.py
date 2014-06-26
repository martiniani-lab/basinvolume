from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic
from basinvolume.spheres import Findk_MCrunner
from basinvolume.utils import trymakedir, read_xyzdr, read_xydr
import ConfigParser
import time

class _findk_mcrunner(object):
    """
    this is a class that implements configure_findk_mcrunner class,
    *k: harmonic spring constant
    *ktarget: target acceptance associated to kmax
    *kfactor: the factor by which k is decreased at each iteration, it must be in (0,1)
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates 
    """
     #niter=1e8   
     #avgcount=1e5
     ####k=1e2, niter=1e8, avgcount=1e5, dtol=1e-4, eps=1., ktarget=0.85, kfactor=0.6, knavg=1000, ktol=0.05,
                 ##opt_dtmax=1, opt_maxstep=None, opt_tol=1e-4, opt_nsteps=1e4, packings_dir='jammed_packings'
    def __init__(self, fname, k=1e2, niter=1e8, avgcount=1e5, dtol=1e-4, eps=1., ktarget=0.85, kfactor=0.6, knavg=1000, ktol=0.05,
                 opt_dtmax=1, opt_maxstep=None, opt_tol=1e-4, opt_nsteps=1e4, packings_dir='jammed_packings'):
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
        
        self._import_packing_config_file()
        self._import_packing_configuration()
        
        #automatically set opt max step
        if opt_maxstep is None:
            opt_maxstep = self.boxv[0]*0.1
        
        self.mc_params = {'k':k,'temperature':self.temperature,'niter':niter,'avgcount':avgcount,'dtol':dtol,'eps':eps, 'ktarget':ktarget, 
                          'kfactor':kfactor, 'knavg':knavg, 'ktol':ktol, 'opt_dtmax':opt_dtmax,'opt_maxstep':opt_maxstep,
                          'opt_tol':opt_tol,'opt_nsteps':opt_nsteps}
        
        #self.coords is origin, set initial configuration and origin to be the same
        potential = Harmonic(self.coords,0,bdim=self.bdim,com=False) #set the potential to 0, the potential is completely fictitious here (there's no energy test),
        #k is entirely controlled by the stepsize 
        stepsize = np.sqrt(1.0/k) #stepsize plays the role of the standard deviation
        #stepsize = np.sqrt(self.ndim/k)  #####################
        #####       
        self.mcrunner = Findk_MCrunner(potential, self.coords, self.temperature, stepsize, niter, self.coords, self.hs_radii, self.boxv, self.sca,
                                  rattlers=self.rattlers, avgcount=avgcount, dtol=dtol, eps=eps, ktarget=ktarget, kfactor=kfactor, knavg=knavg, 
                                  ktol=ktol, opt_dtmax=opt_dtmax, opt_maxstep=opt_maxstep, opt_tol=opt_tol, opt_nsteps=opt_nsteps) 
        self._print_initialise()
    
    def run(self):
        self.mcrunner.run()
        self.kmax = self.mcrunner.get_k()
        self.prob = self.mcrunner.findk.get_prob()
        self.displ_k_max, self.var_displ_k_max = self.mcrunner.findk.get_mean_variance()
        self._print_results()        
    
    def _import_packing_config_file(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert(self.bdim==2 or self.bdim==3) #currently PBC only implemented for 2d-3d case
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
        dname = 'findk_' + self.fname 
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        fname = '{}/{}.config'.format(self.base_directory,dname)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Explore_Jammed_Packings wrapper class input parameters\n')
        f.write('[FINDK_IMPORTED_JAMMED_PACKING]\n')
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
        f.write('[FINDK_MCRUNNER]\n')
        for key, value in self.mc_params.iteritems() :
            f.write('{}: {}\n'.format(key,value)) 
        f.close()
    
    def _print_results(self):
        dname = 'findk_' + self.fname
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        fname = '{}/{}.config'.format(self.base_directory,dname)
        f = open(fname,'a')
        f.write('[FINDK_MCRUNNER_STATUS]\n')
        status = self.mcrunner.get_status()
        for key, value in status.iteritems() :
            f.write('{}: {}\n'.format(key,value))
        f.write('[FINDK]\n')
        f.write('kmax: {}\n'.format(self.kmax))
        f.write('prob: {}\n'.format(self.prob))
        f.write('displ_k_max: {}\n'.format(self.displ_k_max))
        f.write('var_displ_k_max: {}\n'.format(self.var_displ_k_max))
        f.close()
    
if __name__ == "__main__":
    
    #sim = _findk_mcrunner('jammed_packing0.xydr')
    sim = _findk_mcrunner('jammed_packing0.xyzdr')
    print 'simulation started'
    start=time.time() 
    sim.run()
    end=time.time()
    print "time ",end-start
    print "self.kmax: ", sim.kmax
    print "self.prob: ", sim.prob
    print "self.displ_k_max: ", sim.displ_k_max
    print "self.var_displ_k_max: ", sim.var_displ_k_max
    print "Nd/k: ", sim.nparticles*sim.bdim/sim.kmax
    print "(N-1)d/k", (sim.nparticles-1)*sim.bdim/sim.kmax
    sim.mcrunner.show_histogram()
    print "entries in histogram: ",sim.mcrunner.get_entries()
    