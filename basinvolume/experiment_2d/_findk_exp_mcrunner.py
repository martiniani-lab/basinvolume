from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic, HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.spheres import Findk_MCrunner, _configure_mcrunner
from basinvolume.utils import trymakedir, read_xyzdr, read_xydr
from basinvolume.utils import get_git_version, get_python_version, get_cython_version
import ConfigParser
import time
import copy

class _findk_exp_mcrunner(_configure_mcrunner):
    """
    this is a class that implements configure_findk_mcrunner class,
    *k: harmonic spring constant
    *ktarget: target acceptance associated to kmax
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates
    """
    def __init__(self, fname, k=150, niter=1e8, avgcount=1e4, dtol=1e-4, eps=1., ktarget=0.9,
                 knavg=1000, ktol=0.025, opt_dtmax=1, opt_maxstep=None, opt_tol=1e-5,
                 opt_nsteps=1e5, perform_convergence_test=False, collect_minima_list=False,
                 seeds=None, use_cell_lists=False, use_cgd=False, packings_dir='jammed_packings', verbose=False):

        self.temperature=1.0
        self.eps = eps
        self.fname = fname

        self._set_paths(packings_dir)
        self._import_packing_config_files()
        self._import_packing_configuration(frozen=True)
        opt_maxstep = self._get_opt_maxstep(opt_maxstep)

        #select rcontainer to correspond to frozen particle furthest away
        rcontainer = 0
        for i in xrange(len(self.hs_radii)):
            r2=0
            for j in xrange(self.bdim):
                r2 += self.coords[i*self.bdim+j] * self.coords[i*self.bdim+j]
            if r2 > (rcontainer*rcontainer):
                rcontainer = np.sqrt(r2)
                index = i
                if verbose:
                    print "new rcontainer",rcontainer
        #rcontainer -= self.hs_radii[index] #subtract radius of furthest most particle from rcontainer

        #self.mc_params = dict(k=k, temperature=temperature, )
        self.mc_params = {'k':k,'temperature':self.temperature,'niter':niter,'avgcount':avgcount,'dtol':dtol,'eps':self.eps,
                          'ktarget':ktarget, 'knavg':knavg, 'ktol':ktol, 'opt_dtmax':opt_dtmax,'opt_maxstep':opt_maxstep,
                          'opt_tol':opt_tol,'opt_nsteps':opt_nsteps, 'perform_convergence_test':perform_convergence_test,
                          'collect_minima_list':collect_minima_list, 'rcontainer':rcontainer, 'use_cgd':use_cgd}
        #add seeds dictionary to mc_params
        try:
            self.mc_params.update(seeds)
        except:
            print "WARNING:seeds not passed"

        self._requench_coords(dtol, opt_maxstep, verbose, frozen=True)

        #self.coords is origin, set initial configuration and origin to be the same
        potential = Harmonic(self.red_coords,0,bdim=self.bdim,com=False) #set the potential to 0, the potential is completely fictitious here (there's no energy test),
        #k is entirely controlled by the stepsize
        stepsize = np.sqrt(1.0/k) #stepsize plays the role of the standard deviation
        #stepsize = np.sqrt(self.ndim/k)  #####################
        #####
        self.mcrunner = Findk_MCrunner(potential, self.coords, self.temperature, stepsize, niter, self.coords,
                                       self.hs_radii, self.boxv, self.sca, rattlers=self.rattlers, avgcount=avgcount,
                                       dtol=dtol, eps=eps, ktarget=ktarget, knavg=knavg, ktol=ktol,
                                       opt_dtmax=opt_dtmax, opt_maxstep=opt_maxstep, opt_tol=opt_tol,
                                       opt_nsteps=opt_nsteps, perform_convergence_test=perform_convergence_test,
                                       collect_minima_list=collect_minima_list, seeds=seeds, use_cell_lists=use_cell_lists,
                                       use_periodic=False, use_frozen=True, frozen_atoms=self.frozen, rcontainer=rcontainer,
                                       use_cgd=use_cgd)
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
            self._print_success(False)

    def _set_paths(self, packings_dir):
        dname = os.path.splitext(self.fname)[0]
        self.base_directory = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        self.configpath = os.path.join(packings_dir,'{}.config'.format(dname))
        configfile = 'findk_' + dname
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
        self.mobile_particle_radius = configf.getfloat('JAMMED_PACKING','mobile_particle_radius')
        self.frozen_particle_radius = configf.getfloat('JAMMED_PACKING','mobile_particle_radius')

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
        f.write('[FINDK_IMPORTED_JAMMED_PACKING]\n')
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
        f.write('[FINDK_MCRUNNER]\n')
        for key, value in self.mc_params.iteritems() :
            f.write('{}: {}\n'.format(key,value))

    def _print_results(self):
        fname = self.configfile
        f = open(fname,'a')
        f.write('[FINDK_MCRUNNER_STATUS]\n')
        status = self.mcrunner.get_status()
        for key, value in status.iteritems() :
            f.write('{}: {}\n'.format(key,value))
        f.write('[FINDK]\n')
        f.write('kmax: {:.16f}\n'.format(self.kmax))
        f.write('prob: {:.16f}\n'.format(self.prob))
        f.write('displ_k_max: {:.16f}\n'.format(self.displ_k_max))
        f.write('var_displ_k_max: {:.16f}\n'.format(self.var_displ_k_max))
        f.close()

if __name__ == "__main__":

    #sim = _findk_mcrunner('jammed_packing0.xydr')
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[1])

    sim = _findk_exp_mcrunner('jammed_packing1.xydfr', seeds=seeds, use_cell_lists=False, verbose=True, use_cgd=True)
    print 'simulation started'
    start=time.time()
    sim.run()
    frac = sim.mcrunner.conftest2.get_failed_quench_frac()
    print "failed quench frac",frac
    end=time.time()
    print 'time elapsed', end-start
    print "self.kmax: ", sim.kmax
    print "self.prob: ", sim.prob
    print "self.displ_k_max: ", sim.displ_k_max
    print "self.var_displ_k_max: ", sim.var_displ_k_max
    #print "Nd/k: ", sim.nparticles*sim.bdim/sim.kmax
    print "(N-1)d/k", (sim.nparticles-1)*sim.bdim/sim.kmax
    sim.mcrunner.show_histogram()
    print "entries in histogram: ",sim.mcrunner.get_entries()
