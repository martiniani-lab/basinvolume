from __future__ import division
import numpy as np
import os
from pele.potentials import HS_WCAFrozen
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import trymakedir, volume_nball, get_git_version, get_cython_version
from basinvolume.utils import get_python_version, read_xydf, read_xyzdf,reduce_coordinates, full_coordinates
from basinvolume.spheres import _Generate_Jammed_Packing
import ConfigParser
import re
import argparse
import sys

class HS_Exp_Generate_Jammed_Packing(_Generate_Jammed_Packing):
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
    def __init__(self, rattler_eval_tol=1.,packings_dir='packings'):
        super(HS_Exp_Generate_Jammed_Packing,self).__init__(packing_frac=0, packings_dir=packings_dir)                                                        
        
        ##constants#
        self.rattler_eval_tol = rattler_eval_tol 
        ############
        #HACK
        self.configpath = os.path.join(self.packings_dir, 'packing1.config')
        self._import_packing_config_file()
        
    def _initialise(self):
        self._print_initialise()
    
    def one_iteration(self,fname):
        """perform one iteration
        """
        #import configuration
        self._import_exp_packing_config_file(fname)
        self._import_packing_configuration(fname)
                
        #assert that largest soft particle is not > 1/2 of smallest box size
        if np.amax(self.hs_radii)*2*(1+self.sca) >= np.amin(self.boxv)/2:
            print "WARNING: max soft diameter >= 1/2 box side!"
        
        #initialise needs to import at least one configuration to compute sca
        if self.iteration == 0:
            self._initialise()
#        print "coords", self.coords
#        print "frozen", self.frozen
#        print "hsradii", self.hs_radii
#        print "sca", self.sca
#        print "eps", self.eps
#        print "ndim", self.bdim
        assert(len(self.coords)/self.bdim == len(self.hs_radii))
        ###potential needs to be called because self.coords is an input argument of HS_WCAPeriodicCellLists
        self.potential = HS_WCAFrozen(self.coords, self.frozen, self.eps, self.sca, self.hs_radii, ndim=self.bdim)
        
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
        assert(len(redcoords) == self.ndim)
        maxstep = np.amin(self.hs_radii)
        res = modifiedfire_cpp(redcoords,self.potential, maxstep=maxstep, nsteps=1e6, tol=tol)
        if not res.success:
            print 'quench failed'
            return False
        
        redcoords = res.coords.copy()
        self.energy = res.energy
        
        #test that on re-minimisation the structure does not change
        res2 = modifiedfire_cpp(redcoords, self.potential, maxstep=maxstep, nsteps=1e6, tol=tol)
        if res2.nfev > 1:
            print 'quench failed (structure changed at second minimisation)'
            return False
        
        self.coords = full_coordinates(redcoords, self.coords, self.frozen, self.bdim)
        #asserts that none of the hard sphere is overlapping
        no_overlap = self._check_overlaps()
        if not no_overlap:
            print 'overlap found'
            return False 
        
        #analyse packing, assert that the whole system has only 3 0'evalues + a 0 evalue for each rattler 0 evalue
        hess = self.potential.getHessian(redcoords)
        ratt0evals= []
        for i in xrange(self.nparticles):
            i1 = self.bdim*i
            hess_block = hess[i1:i1+self.bdim,i1:i1+self.bdim]
            w, v = np.linalg.eig(hess_block)
            w = np.real(w)
            ratt0evals.extend([x for x in w if abs(x) < self.rattler_eval_tol]) #append to array of zero evalues due to rattlers
        
        #assert that the number of 0 block eigenvalues mathesh the full hessian 0 eigenvalues
        w, v = np.linalg.eig(hess)
        w = np.real(w)
        full0evals = [x for x in w if abs(x) < self.rattler_eval_tol]
        if len(full0evals) - len(ratt0evals) > self.bdim:
            print 'hessian 0s mismatch rattlers 0s'
            return False
        
        #check that there isn't any significantly negative evalue
        if np.any(w) < -0.01:
            print 'eigevalue < -0.01'
            return False
        
        return True
    
    def _check_overlaps(self):
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
        else:
            self.coords, hs_diameters, self.frozen = read_xyzdf(path)
        self.hs_radii = hs_diameters/2
    
    def _import_exp_packing_config_file(self, fname):
        dname = fname
        if dname.endswith('.xyzdf'):
            dname = dname[:-6]
        elif dname.endswith('.xydf'):
            dname = dname[:-5]
        self.configpath = os.path.join(self.packings_dir, dname+'.config')
        self._import_packing_config_file()
        self.packing_frac = np.power(1+self.sca,self.bdim)*self.imp_packing_frac
    
    def _import_packing_config_file(self):
        configf = ConfigParser.ConfigParser()
        configf.read(self.configpath)
        self.nparticles = configf.getint('PACKING','nparticles')
        self.bdim = configf.getint('PACKING','boxdim')
        assert(self.bdim==2 or self.bdim==3) #currently PBC only implemented for 3d case
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('PACKING','packing_fraction')
        self.sca = (configf.getfloat('PACKING','deflation') - 1)*1.1 #fudge
        
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
        else:
            fname = "{0}/jammed_packing{1}.xyzdfr".format(directory,n)
            f = open(fname,'w')
            for i in xrange(nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                          coords[i*self.bdim+2],self.hs_radii[i]*2,int(i in self.frozen),
                                                                          self.rattlers[i]))
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
        else:
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
        f.close()
                    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="generate 2/3-D hard disks/spheres packings")
    parser.add_argument("-e","--etol", type=float, help="tolerance on particles eigenvalues, if eval < etol particle will be considered a rattler",default=1.0)
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="packings")
    args = parser.parse_args()
    print args
    
    sim = HS_Exp_Generate_Jammed_Packing(rattler_eval_tol=args.etol, packings_dir=args.packingsdir)
    sim.run()
    
    
        
                
            
              
                
                
                
