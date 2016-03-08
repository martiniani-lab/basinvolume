from __future__ import division
import numpy as np
import abc
from pele.potentials import HS_WCA
from pele.potentials import InversePowerStillingerCut
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import *
import ConfigParser
import re
import argparse
try:
    import pylab
except:
    pass

class _Generate_Jammed_Packing(object):
    """
    this is an abstract class that implements the basic components of a generate packing class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *method to generate packing, this could be for example direct sampling, 
    sequential sampling,quench or LSA
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, packing_frac=0.65, packings_dir='packings'):
        self.packing_frac = packing_frac
        self.base_directory = os.path.join(os.getcwd(),'jammed_packings')
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        self.iteration = 0
        self.sca = -1
        self.eps = 1.
    
    def _import_single_packing_config_file(self, fname):
        dname = fname
        if dname.endswith('.xyzd'):
            dname = dname[:-5]
        elif dname.endswith('.xyd'):
            dname = dname[:-4]
        self.configpath = os.path.join(self.packings_dir, dname+'.config')
        self._import_packing_config_file()
          
    def _import_packing_config_file(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.configpath))
        self.nparticles = configf.getint('PACKING','nparticles')
        self.bdim = configf.getint('PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('PACKING','packing_fraction')
    
    @abc.abstractmethod
    def _initialise(self):
        """initialisation function"""
        self.configpath = os.path.join(self.packings_dir,'packings.config')
        assert(os.path.isfile(self.configpath))
        self._import_packing_config_file()
    
    @abc.abstractmethod
    def _import_packing_configuration(self, fname):
        """imports the coordinates and data relative to the shape of the particles
            this should be run in initialise()
        """
    @abc.abstractmethod
    def _generate_packing_coords(self):
        """function that generates the packing"""
    
    @abc.abstractmethod
    def _write_opengl_input(self, n):
        """writes a opengl input file, n is the unique identifier of the structure"""
        
    @abc.abstractmethod
    def _dump_configuration(self, n):
        """writes a configuration file, e.g .xyzd, n is the unique identifier of the structure"""
            
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
    
    def _print_parameters(self, n):
        """writes the simulation parameters"""
        fname = '{}/jammed_packing{}.config'.format(self.base_directory,n)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Generate_Jammed_Packings base class input parameters\n')
        f.write('[JAMMED_PACKING]\n')
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {}\n'.format(self.packing_frac))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndim))
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        assert(self.sca > 0)
        f.write('sca: {:.16f}\n'.format(self.sca))
        f.write('\n')
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
        f.close()
        
    def _print(self, n):
        """dump configuration and opengl input to packings directory
            n is the unique identifier of the structure
        """
        self._print_parameters(n)
        self._dump_configuration(n)
        self._write_opengl_input(n)
    
    @abc.abstractmethod
    def _histogram_eigenvalues(self):
        """ method to plot eigenvalues histograms
        """
    
    @abc.abstractmethod
    def one_iteration(self,fname):
        """perform one iteration
        """
    def run(self):
        """run generate packings"""
        self._initialise()
        for fname in os.listdir(self.packings_dir):
            if ('xyzd' in fname) or ('xyd' in fname):
                print "\n",fname
                self.one_iteration(fname)
        self._histogram_eigenvalues()
            
class HS_Generate_Jammed_Packing(_Generate_Jammed_Packing):
    """
    *this class generates packings and identifies rattlers by computing the hessian eigenvalues for each particle
    *in the equilibrium jammed structure. A .xyzdr file is produced that contains the 3 system coordinates, the particle 
    * diameter and if not it's a rattler (0 if a rattler, 1 otherwise)
    *PARAMETERS
    *hs_radii: array with the radii of the particles, if none sample particle sizes from a normal distribution
    *mu: average particle size, passable to normal distribution
    *sig: standard deviaton of normal distribution from which to sample particles
    *sca: determines % by which the hs is inflated
    *eps: LJ interaction energy of WCA part of the HS potential
    """    
    def __init__(self, packing_frac=0.7, rattler_eval_tol=1.,
        packings_dir='packings', use_cell_lists=False, show=False,
        opt_pot_str='hs_wca', **extra_pot_kwargs):
        super(HS_Generate_Jammed_Packing,self).__init__(packing_frac=packing_frac, packings_dir=packings_dir)
        
        self.opt_pot_str = opt_pot_str
        self.extra_pot_kwargs = extra_pot_kwargs
        self.use_cell_lists = use_cell_lists
        ##constants#
        self.rattler_eval_tol = rattler_eval_tol
        ############
        #histogram variables#
        self.block_evalues = []
        self.whole_evalues = []
        self.nbins = 1000
        self.nbins_low = 6000
        self.low_range = (-0.00001,0.00001)
        self.show = show
    
    def _initialise(self):
        self._print_initialise()
    
    def one_iteration(self,fname):
        """perform one iteration
        """
        self._import_single_packing_config_file(fname)
        self.rattlers = np.empty(self.nparticles,dtype='d')
        self.rattlers_draw = np.empty(self.nparticles,dtype='d')
        
        self._import_packing_configuration(fname)
        self.max_nrattlers = int(self.nparticles*0.1)
        
        #assert that largest soft particle is not > 1/2 of smallest box size
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv) / 2:
            print "WARNING: max soft diameter >= 1/2 box side!"
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv):
            raise Exception("WARNING: particle does not fit the box")
        
        ###potential needs to be called because self.coords is an input argument of HS_WCAPeriodicCellLists
        rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca) #rcut set to largest particle diameter
        if self.use_cell_lists:
            if np.amin(self.boxv) // rcut <= 3:
                self.use_cell_lists = False
        if self.opt_pot_str.lower() == "hs_wca":
            if self.use_cell_lists:
                self.potential = HS_WCA(use_periodic=True,
                    use_cell_lists=True, eps=self.eps, sca=self.sca,
                    radii=self.hs_radii, boxvec=self.boxv,
                    reference_coords=self.coords, rcut=rcut,
                    ndim=self.bdim, ncellx_scale=1.0)
            else:
                self.potential = HS_WCA(use_periodic=True, eps=self.eps,
                    sca=self.sca, radii=self.hs_radii, boxvec=self.boxv,
                    ndim=self.bdim)
        elif self.opt_pot_str.lower() == "inverse_power_stillinger":
            pow = self.extra_pot_kwargs['pow']
            rcut = self.extra_pot_kwargs["rcut"]
            self.potential = InversePowerStillingerCut(pow,
                self.stillinger_a_radii, ndim=self.bdim,
                boxvec=self.boxv, rcut=rcut, use_cell_lists=True)
        else:
            raise NotImplementedError
        
        success = self._generate_packing_coords() #returns false if saddle
        
        if success:
            self._find_rattlers()
            #strips the integer unique identifier out of fname
            n = int(re.search(r'\d+',fname).group())
            self._print(n)
        
        self.iteration+=1
    
    def _find_rattlers(self):
        hess = self.potential.getHessian(self.coords)
        for i in xrange(self.nparticles):
            i1 = self.bdim*i
            hess_block = hess[i1:i1+self.bdim,i1:i1+self.bdim]
            w, v = np.linalg.eig(hess_block)
            w = np.real(w)
            self.rattlers[i] = np.amin(np.absolute(w))
            self.rattlers_draw[i] = float(self.rattlers[i] >= self.rattler_eval_tol)
            if self.rattlers_draw[i] < 1:
                print 'zero eigenvalue, particle {}'.format(i)
                print w
#            rattler = np.less_equal(np.absolute(w),self.rattler_eval_tol)
#            #self.rattlers[i] = float(not True in rattler)
#            if True in rattler:
#                self.rattlers[i] = 0.
#                print 'zero eigenvalue, particle {}'.format(i)
#                print w
#            else:
#                self.rattlers[i] = 1.
    
    
    def _generate_packing_coords(self):
        """
        perform quench and run tests
        """
        success = self._generate_packing_coords_iteration(tol=1e-7)
        return success
    
    def _generate_packing_coords_iteration(self, tol=1e-7, iprint=-1):
        """quenches the imported structure using FIRE"""
        fire_maxstep = np.amin(self.hs_radii)*self.sca
        res = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep, nsteps=1e6, tol=tol, iprint=iprint)
        if not res.success:
            print 'quench failed'
            return False
        
        self.coords = res.coords
        self.energy = res.energy
        
        #test that on ri-minimisation the structure does not change
        res2 = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep, nsteps=1e6, tol=tol)
        if res2.nfev > 1:
            print 'quench failed (structure changed at second minimisation)'
            return False
        
        #asserts that none of the hard sphere is overlapping
        no_overlap = self._check_no_overlaps()
        if not no_overlap:
            print 'overlap found'
            return False 
        
        #analyse packing, assert that the whole system has only 3 0'evalues + a 0 evalue for each rattler 0 evalue
        hess = self.potential.getHessian(self.coords)
        ratt0evals= []
        nratls = 0
        for i in xrange(self.nparticles):
            i1 = self.bdim*i
            hess_block = hess[i1:i1+self.bdim,i1:i1+self.bdim]
            w, v = np.linalg.eig(hess_block)
            w = np.real(w)
            if np.any(w < self.rattler_eval_tol):
                nratls += 1
            ratt0evals.extend([x for x in w if abs(x) < self.rattler_eval_tol]) #append to array of zero evalues due to rattlers
            self.block_evalues.extend(w) 
        
        w, v = np.linalg.eig(hess)
        w = np.real(w)
        full0evals = [x for x in w if abs(x) < 1e-6]
        if len(full0evals) - len(ratt0evals) > self.bdim:
            print 'hessian 0s mismatch rattlers 0s'
            return False
        self.whole_evalues.extend(w)
          
        print "nrattlers: {}".format(nratls)
        if nratls > self.max_nrattlers:
            print '{} rattlers constitute more than 10% of the system'.format(nratls)
            return False 

#        if the mismatch test works correctly there is no need to test for negative eigenvalues 
#        because the negative eigenvalue that makes the test fail might (and probably will) belong 
#        to a rattler, in which case who cares. If it does not belong to a rattler then the mismatch 
#        test will fail anyway first, so I am going to remove this test.
#        #check that there isn't any significantly negative evalue
#        if np.any(w < -2.5e-7):
#            print 'e: {} eigenvalue < -2.5e-7'.format(np.amin(w))
#            return False
        
        return True
    
    def _get_particles_volume(self):
        """returns volume of n=self.bdim dimensional sphere"""
        volumes = volume_nball(self.hs_radii,self.bdim)
        vtot = np.sum(volumes)
        return vtot
    
    def _import_packing_configuration(self, fname):
        path = os.path.join(self.packings_dir, fname)
        if self.bdim == 2:
            self.coords, hs_diameters = read_xyd(path)
        elif self.bdim == 3:
            self.coords, hs_diameters = read_xyzd(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        self.hs_radii = hs_diameters/2
        self._compute_sca()
    
    def _compute_sca(self):
        ##test##
        vol_part = self._get_particles_volume()
        vol_box = np.prod(self.boxv)
        phi = vol_part/vol_box #instanteneous pack frac
        assert(phi - self.imp_packing_frac < 1e-4)
        ##endtest##
        ###r_soft = r_hs*(1+sca)
        self.sca = np.power(self.packing_frac/self.imp_packing_frac,1./self.bdim) - 1
        
    def _check_no_overlaps(self):
        """check that no two particles are overlapping (using nearest image convention)"""
        no_overlap = True
        for i in xrange(self.nparticles):
            if no_overlap == True:
                for j in xrange(self.nparticles):
                    dij = 0
                    for k in xrange(self.bdim):
                        #use distances to nearest image convention
                        dij += np.square((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) -
                                          cround((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
                    if i != j:
                        dij = np.sqrt(dij)
                        dmin = self.hs_radii[i]+self.hs_radii[j]
                        if dij - dmin <= 0:
                            print 'invalid configuration'
                            print 'atoms {} {} are overlapping'.format(i,j)
                            print 'real distance {}'.format(dij)
                            print 'min distance {}'.format(dmin)
                            no_overlap = False
                            break
            else:
                break
        return no_overlap
            
    def _correct_coords(self):
        """this function returns the nearest images in the central box, useful for dumping the configurations"""
        coords = self.coords.copy()
        put_in_box(coords,self.boxv)
        return coords
    
    def _dump_configuration(self,n):
        """write coordinates to file .xyzdr"""
        coords = self._correct_coords()
        directory = self.base_directory
        if self.bdim == 2:
            fname = "{0}/jammed_packing{1}.xydr".format(directory,n)
            f = open(fname,'w')
            for i in xrange(self.nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                          self.hs_radii[i]*2,self.rattlers[i]))
        elif self.bdim == 3:
            fname = "{0}/jammed_packing{1}.xyzdr".format(directory,n)
            f = open(fname,'w')
            for i in xrange(self.nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                          coords[i*self.bdim+2],self.hs_radii[i]*2,self.rattlers[i]))
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        f.close()
    
    def _write_opengl_input(self,n):
        """write opengl input file"""
        coords = self._correct_coords()
        boxv = self.boxv
        colour = 14
        directory = self.base_directory
        fname = "{0}/jammed_packing{1}.dat".format(directory,n)
        f = open(fname,'w')
        f.write('{}\n'.format(self.nparticles))
        if self.bdim == 2:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2, - np.amax(self.hs_radii)))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(np.amax(self.hs_radii)*2))
            for i in xrange(self.nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(0.0))
                f.write('{}\t'.format(self.hs_radii[i]*2*(1.+self.sca)))
                f.write('{}\n'.format(colour-int(self.rattlers_draw[i])))
        elif self.bdim == 3:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2,-boxv[2]/2))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(boxv[2]))
            for i in xrange(self.nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(self.hs_radii[i]*2*(1.+self.sca)))
                f.write('{}\n'.format(colour-int(self.rattlers_draw[i])))
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        f.close()
    
    def _histogram_eigenvalues(self):
        #self.eigenvalues = np.array(self.eigenvalues,dtype='d')
        self.block_evalues = np.real(self.block_evalues)
        pylab.figure()
        self.block_histogram, bins = np.histogram(self.block_evalues ,bins=self.nbins)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.block_histogram, align='center', width=width)
        pylab.savefig(os.path.join(self.base_directory,'blocks_histogram.eps'))
        if self.show:
            pylab.show()
        pylab.figure()
        self.block_histogram_low, bins = np.histogram(self.block_evalues ,bins=self.nbins_low, range=self.low_range)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.block_histogram_low, align='center', width=width)
        pylab.savefig(os.path.join(self.base_directory, 'blocks_histogram_low{}.eps'.format(self.low_range[1])) )
        if self.show:
            pylab.show()
        
        self.whole_evalues = np.real(self.whole_evalues)
        pylab.figure()
        self.whole_histogram, bins = np.histogram(self.whole_evalues ,bins=self.nbins)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.whole_histogram, align='center', width=width)
        pylab.savefig(os.path.join(self.base_directory,'whole_histogram.eps'))
        if self.show:
            pylab.show()
        pylab.figure()
        self.whole_histogram_low, bins = np.histogram(self.whole_evalues ,bins=self.nbins_low, range=self.low_range)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.whole_histogram_low, align='center', width=width)
        pylab.savefig(os.path.join(self.base_directory,'whole_histogram_low{}.eps'.format(self.low_range[1])))
        if self.show:
            pylab.show()

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="generate 2/3-D hard disks/spheres packings")
    parser.add_argument("-p","--density", type=float, help="target packing fraction",default=0.7)
    parser.add_argument("-e","--etol", type=float, help="tolerance on particles eigenvalues, if eval < etol particle will be considered a rattler",default=1.0)
    parser.add_argument("--nocell", action='store_false', help="don't use cell lists, default: True",default=True)
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="packings")
    parser.add_argument("--show", action='store_true', help="show histograms", default=False)
    # potential arguments
    parser.add_argument("--opt-pot", type=str, help="optmizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')
    args = parser.parse_args()
    print args
    
    # potential type
    opt_pot_str = args.opt_pot
    extra_pot_kwargs = dict()
    if opt_pot_str == 'hs_wca':
        pass
    elif opt_pot_str == 'inverse_power_stillinger':
        extra_pot_kwargs.update(dict(pow=3, rcut=1.5))
        print 'setting inverse_power_stillinger parameters: ', extra_pot_kwargs
    else:
        raise NotImplementedError
    
    sim = HS_Generate_Jammed_Packing(packing_frac=args.density, rattler_eval_tol=args.etol, packings_dir=args.packingsdir,
                                     use_cell_lists=args.nocell, show=args.show)
    sim.run()
    
    
        
                
            
              
                
                
                
