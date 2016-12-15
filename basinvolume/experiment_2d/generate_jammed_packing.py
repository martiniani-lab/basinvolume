from __future__ import division
import numpy as np
import os
import pyvoro
from pele.potentials import HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import trymakedir, volume_nball, get_git_version, get_cython_version
from basinvolume.utils import get_python_version, read_xydf, read_xyzdf,reduce_coordinates, full_coordinates
from basinvolume.spheres import _Generate_Jammed_Packing
import ConfigParser
import re
import argparse
import sys
try:
    import pylab
except:
    pass

class HS_Exp_Generate_Jammed_Packing(_Generate_Jammed_Packing):
    """
    *this class generates packings and identifies rattlers by computing the hessian eigenvalues for each particle
    *in the equilibrium jammed structure. A .xyzdr file is produced that contains the 3 system coordinates, the particle 
    * diameter and if not it's a rattler (0 if a rattler, 1 otherwise)
    *PARAMETERS
    *expand_sca: #amount by which scaling factor is multiplied to over-inflate
    *hs_radii: array with the radii of the particles, if none sample particle sizes from a normal distribution
    *mu: average particle size, passable to normal distribution
    *sig: standard deviaton of normal distribution from which to sample particles
    *sca: determines % by which the hs is inflated
    *eps: LJ interaction energy of WCA part of the HS potential
    """    
    def __init__(self, packing_frac=0.65, rattler_eval_tol=1., packings_dir='packings', show=False):
        super(HS_Exp_Generate_Jammed_Packing,self).__init__(packing_frac=packing_frac, packings_dir=packings_dir)                                                        
        
        ##constants#
        self.rattler_eval_tol = rattler_eval_tol
        ############
        #histogram variables#
        self.block_evalues = []
        self.whole_evalues = []
        self.nbins = 1000
        self.nbins_low = 500
        self.low_range = (-1,1)
        self.show = show
           
    def _initialise(self):
        self._print_initialise()
    
    def one_iteration(self,fname):
        """perform one iteration
        """
        #import configuration
        self._import_single_packing_config_file(fname)
        self._import_packing_configuration(fname)
        self._compute_sca()
        self.max_nrattlers = int(self.nparticles*0.1)

        #assert that largest soft particle is not > 1/2 of smallest box size
        if np.amax(self.hs_radii)*2*(1+self.sca) >= np.amin(self.boxv)/2:
            print "WARNING: max soft diameter >= 1/2 box side!"

        assert(len(self.coords)/self.bdim == len(self.hs_radii))

        #rcut = np.amax(self.hs_radii)*2
        #use_cell_lists=True, boxvec=self.boxv
        self.potential = HS_WCA(reference_coords=self.coords, eps=self.eps, sca=self.sca,
                                radii=self.hs_radii, use_frozen=True, frozen_atoms=self.frozen,
                                ndim=self.bdim, use_cell_lists=False, distance_method='cartesian')

        success = self._generate_packing_coords() #returns false if saddle

        if success:
            self._find_rattlers()
            #strips the integer unique identifier out of fname
            n = int(re.search(r'\d+',fname).group())
            self._print(n)

        self.iteration+=1

    def _find_rattlers(self):
        self.rattlers = np.empty(self.nparticles,dtype='d')
        self.rattlers_draw = np.empty(self.nparticles,dtype='d')
        redcoords = reduce_coordinates(self.coords, self.frozen, self.bdim)
        hess = self.potential.getHessian(redcoords)
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
        #set the eigenvalue of frozen particles to 0 as if they were rattlers
        for i in sorted(self.frozen, reverse=False):
            self.rattlers = np.insert(self.rattlers, i, 0)
            self.rattlers_draw = np.insert(self.rattlers_draw, i, 0)
    
    def _generate_packing_coords(self):
        """
        perform quench and run tests
        """
        success = self._generate_packing_coords_iteration(tol=1e-9)
        return success
    
    def _generate_packing_coords_iteration(self, tol=1e-9):
        """quenches the imported structure using FIRE"""
        redcoords = reduce_coordinates(self.coords, self.frozen, self.bdim)
        red_radii = np.delete(self.hs_radii, self.frozen)
        assert(len(redcoords) == self.ndim)
        maxstep = np.amin(red_radii) * self.sca
        res = modifiedfire_cpp(redcoords, self.potential, maxstep=maxstep, nsteps=1e6, tol=tol)
        if not res.success:
            print 'quench failed'
            return False
        
        new_redcoords = res.coords.copy()
        self.energy = res.energy
        
        #test that on re-minimisation the structure does not change
        res2 = modifiedfire_cpp(new_redcoords, self.potential, maxstep=maxstep, nsteps=1e6, tol=tol)
        if res2.nfev > 1:
            print 'quench failed (structure changed at second minimisation)'
            return False
        
        #check that no particle has moved more than its own diameter
        dvec = np.power(new_redcoords - redcoords,2)
        for i in xrange(0,np.size(dvec), self.bdim):
            if np.sqrt(np.sum(dvec[i:i+self.bdim])) > red_radii[int(i/self.bdim)]:
                print "quench rejected, particle has moved more than its own radius"
                return False
        
        self.coords = full_coordinates(new_redcoords, self.coords, self.frozen, self.bdim)
        #asserts that none of the hard sphere is overlapping
        no_overlap = self._check_no_overlaps()
        if not no_overlap:
            print 'overlap found'
            return False 
        
        #analyse packing, assert that the whole system has only 3 0'evalues + a 0 evalue for each rattler 0 evalue
        hess = self.potential.getHessian(redcoords)
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
            
        #assert that the number of 0 block eigenvalues mathesh the full hessian 0 eigenvalues
        w, v = np.linalg.eig(hess)
        w = np.real(w)
        full0evals = [x for x in w if abs(x) < self.rattler_eval_tol]
        if len(full0evals) - len(ratt0evals) > self.bdim:
            print 'hessian 0s mismatch rattlers 0s'
            #do not return false because for lower packings fraction the number of low freequency modes increases significantly
            pass
        self.whole_evalues.extend(w)
        
        print "nrattlers: {}".format(nratls)
        if nratls > self.max_nrattlers:
            print '{} rattlers constitute more than 10% of the system'.format(nratls)
            return False 
        
        #check that there isn't any significantly negative evalue
        if np.any(w < -2e-7):
            print 'eigevalue < -2e-7'
            return False
        
        return True
    
    def _check_no_overlaps(self):
        """check that no two particles are overlapping (using nearest image convention)"""
        no_overlap = True
        for i in xrange(self.nparticles):
            if no_overlap == True:
                for j in xrange(i, self.nparticles):
                    dij = 0
                    for k in xrange(self.bdim):
                        #use distances to nearest image convention
                        dij += np.square(self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k])
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
            
    def _import_packing_configuration(self, fname):
        path = os.path.join(self.packings_dir,fname)
        if self.bdim == 2:
            self.coords, hs_diameters, self.frozen = read_xydf(path)
        elif self.bdim == 3:
            self.coords, hs_diameters, self.frozen = read_xyzdf(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        self.hs_radii = hs_diameters/2
    
    def _import_single_packing_config_file(self, fname):
        dname = fname
        if dname.endswith('.xyzdf'):
            dname = dname[:-6]
        elif dname.endswith('.xydf'):
            dname = dname[:-5]
        self.configpath = os.path.join(self.packings_dir, dname+'.config')
        self._import_packing_config_file()
        #self.packing_frac = np.power(1+self.imp_sca,self.bdim)*self.imp_packing_frac
    
    def _import_packing_config_file(self):
        configf = ConfigParser.ConfigParser()
        configf.read(self.configpath)
        self.nparticles = configf.getint('PACKING','nparticles')
        self.bdim = configf.getint('PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('PACKING','packing_fraction')
        self.imp_sca = (configf.getfloat('PACKING','deflation') - 1)
        self.mobile_particle_radius = configf.getfloat('EXPERIMENTAL_DATA_EXTRACTION','mobile_particle_radius')
        self.frozen_particle_radius = configf.getfloat('EXPERIMENTAL_DATA_EXTRACTION','mobile_particle_radius')
    
    def _compute_sca(self):
        ##test##
        vparticle = self._get_particles_volume()
        vcavity = self._get_voronoi_mobile_area()
        assert(0 < vparticle < vcavity)
        self.vcavity = vcavity
        phi = vparticle/vcavity
        assert(phi - self.imp_packing_frac < 1e-4)
        ##endtest##
        ###r_soft = r_hs*(1+sca)
        self.sca = np.power(self.packing_frac/self.imp_packing_frac,1./self.bdim) - 1
    
    def _get_particles_volume(self):
        """returns volume of n=self.bdim dimensional sphere for mobile particles """
        volume = 0.
        for i in xrange(len(self.hs_radii)):
            if i not in self.frozen:
                volume += volume_nball(self.hs_radii[i],self.bdim)
        return volume
    
    def _get_voronoi_mobile_area(self):
        """
        Voronoi tesselates the packing and adds up the areas of the mobile particles. This should
        give some decent estimate of the volume fraction for the current packing
        """
        #get coordinates
        coords = self.coords.reshape(-1,self.bdim).tolist()
        #get box limits
        limits = []
        for i in xrange(self.bdim):
            limits.append([-self.boxv[i]/2,self.boxv[i]/2])
        #compute dispersion (max distance between two points that might be adjacent)
        dispersion = np.amax(self.hs_radii) * 2
        #get radii and compute mean and standard deviation
        radii = self.hs_radii.tolist()
        #tesselate packing
        if self.bdim == 2:
            cells = pyvoro.compute_2d_voronoi(coords,limits, dispersion, radii=radii)
        elif self.bdim == 3:
            cells = pyvoro.compute_voronoi(coords,limits, dispersion, radii=radii)
        else:
            raise NotImplementedError("pyvoro bdim={} not implemented".format(self.bdim))
        assert(len(cells) == int(len(self.coords)/self.bdim))
        #compute free volume
        vcavity = 0.
        vtot = 0.
        assert(len(cells) == len(self.hs_radii))
        for i,cell in enumerate(cells):
            assert(cell['original'] == coords[i])
            vtot += cell['volume']
            if i not in self.frozen:
                vcavity += cell['volume']
        #test that sum of voronoi areas is within some precision from the exact area
        assert(abs(vtot - np.product(self.boxv)) < 1e-3)
        assert(0 < vcavity < vtot)
        return vcavity
        
    def _dump_configuration(self,n):
        """write coordinates to file .xyzdr"""
        coords = self.coords
        directory = self.base_directory
        nparticles = len(self.hs_radii)
        if self.bdim == 2:
            fname = "{0}/jammed_packing{1}.xydfr".format(directory,n)
            f = open(fname,'w')
            for i in xrange(nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                          self.hs_radii[i]*2, int(i in self.frozen), self.rattlers[i]))
        elif self.bdim == 3:
            fname = "{0}/jammed_packing{1}.xyzdfr".format(directory,n)
            f = open(fname,'w')
            for i in xrange(nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                          coords[i*self.bdim+2],self.hs_radii[i]*2,int(i in self.frozen),
                                                                          self.rattlers[i]))
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        f.close()
    
    def _print(self, n):
        """dump configuration and opengl input to packings directory"""
        self._print_parameters(n)
        self._dump_configuration(n)
        self._write_opengl_input(n)
    
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
        assert(self.sca >0)
        f.write('sca: {:.16f}\n'.format(self.sca))
        f.write('vcavity: {:.16f}\n'.format(self.vcavity))
        f.write('mobile_particle_radius: {:.16f}\n'.format(self.mobile_particle_radius))
        f.write('frozen_particle_radius: {:.16f}\n'.format(self.frozen_particle_radius))
        f.write('\n')
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
        f.close()
    
    def _write_opengl_input(self,n):
        """write opengl input file"""
        coords = self.coords
        boxv = self.boxv
        colour = 14
        nparticles = len(self.hs_radii)
        directory = self.base_directory
        fname = "{0}/jammed_packing{1}.dat".format(directory,n)
        f = open(fname,'w')
        f.write('{}\n'.format(nparticles))
        if self.bdim == 2:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2, - np.amax(self.hs_radii)))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(np.amax(self.hs_radii)*2))
            for i in xrange(nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(0.0))
                f.write('{}\t'.format(self.hs_radii[i]*2*(1.+self.sca)))
                if i in self.frozen:
                    f.write('{}\n'.format(colour + 1))
                elif bool(self.rattlers_draw[i]):
                    f.write('{}\n'.format(colour + 2))
                else:
                    f.write('{}\n'.format(colour))
        elif self.bdim == 3:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2,-boxv[2]/2))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(boxv[2]))
            for i in xrange(nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(self.hs_radii[i]*2*(1.+self.sca)))
                if i in self.frozen:
                    f.write('{}\n'.format(colour + 1))
                elif bool(self.rattlers_draw[i]):
                    f.write('{}\n'.format(colour + 2))
                else:
                    f.write('{}\n'.format(colour))
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
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="packings")
    parser.add_argument("--show", action='store_true', help="show histograms", default=False)
    args = parser.parse_args()
    print args

    sim = HS_Exp_Generate_Jammed_Packing(packing_frac=args.density, rattler_eval_tol=args.etol, packings_dir=args.packingsdir,
                                         show=args.show)
    sim.run()
