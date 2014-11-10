from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import HS_WCA
from pele.storage import Minimum
from basinvolume.utils import *
from basinvolume.gui import HSWCASystem
import ConfigParser
import time
import re
import pylab
from pele.gui import run_gui
import argparse

class analyse_jammed_packings(object):
    """
    this is an abstract class that implements the basic components of a configure bv_mcrunner class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    """
        
    def __init__(self, etol=0.1, packings_dir='jammed_packings', hist_show=False):
        self.base_directory = os.path.join(os.getcwd(),'analyse_jammed_packings')
        self.packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.configpath = os.path.join(packings_dir,'jammed_packings.config')
        self._import_packing_config_file()
        self.eps=1.
        self.etol = 0.1
        self.show = hist_show
        self.iteration = 0
        self.block_evalues = []
        self.whole_evalues = []
        self.nbins = 1000
        self.nbins_low = 500
        self.low_range = (-1,1)
        
    def _initialise(self):
        """initialisation function"""
        self._import_packing_config_file()
        #change directory only at the end of initialise
        self._print_initialise()
        os.chdir(self.base_directory)
        
    def _import_packing_config_file(self):
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
        
    def _import_packing_configuration(self,fname):
        """imports the coordinates and data relative to the shinitape of the particles
            this should be run in initialise()
        """
        path = os.path.join(self.packings_dir,fname)
        if self.bdim == 2:
            self.coords, hs_diameters, self.rattlers = read_xydr(path)
        elif self.bdim == 3:
            self.coords, hs_diameters, self.rattlers = read_xyzdr(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        self.hs_radii = hs_diameters/2
      
    def analyse_hessian(self,fname):
        """compute hessian and its eigenvalues
        """
        self.hess_block = np.zeros((self.bdim,self.bdim));        
        hess = self.potential.getHessian(self.coords)
        
        #analyse packing, assert that the whole system has only 3 0'evalues + a 0 evalue for each rattler 0 evalue
        ratt0evals= []
        for i in xrange(self.nparticles):
            i1 = self.bdim*i
            hess_block = hess[i1:i1+self.bdim,i1:i1+self.bdim]
            w, v = np.linalg.eig(hess_block)
            w = np.real(w)
            if np.any(np.absolute(w) < self.etol):
                print 'zero eigenvalue, particle {}'.format(i)
                print w
                assert(self.rattlers[i1]==0)
            self.block_evalues.extend(w)
            ratt0evals.extend([x for x in w if abs(x) < self.etol]) #append to array of zero evalues due to rattlers
        
        w, v = np.linalg.eig(hess)
        w = np.real(w)
        full0evals = [x for x in w if abs(x) < self.etol]
        if len(full0evals) - len(ratt0evals) > self.bdim:
            print "configuration is a saddle: more than 3 + bloc0's eigenvalues"
        self.whole_evalues.extend(w)
        
        #check that there isn't any significantly negative evalue
        if np.any(w) < -0.1:
            print "configuration is a saddle, it has strongly negative evalue"
        
        self.iteration+=1
        print "\n"
    
    def one_iteration(self, fname):
        self._import_packing_configuration(fname)
        if self.iteration is 0:
            self._initialise()
            self.system = HSWCASystem(self.eps, self.sca, self.hs_radii, self.boxv, bdim=self.bdim)
            self.potential = self.system.get_potential()
            self.db = self.system.create_database()
        self.analyse_hessian(fname)
        coords = self.coords.copy()
        np.array(put_in_box(coords,self.boxv))
        self.db.addMinimum(self.potential.getEnergy(self.coords), coords)
#        m = Minimum(self.potential.getEnergy(self.coords), coords)
#        m.user_data = dict(rattlers=self.rattlers[::self.bdim])
#        self.db.session.add(m)
#        self.db.session.commit()
        
        
    
    def run(self):
        """run generate packings"""
        for fname in os.listdir(self.packings_dir):
            if 'xyzdr' in fname or 'xydr' in fname:
                print fname
                self.one_iteration(fname)
        self._histogram_eigenvalues()
    
    def _histogram_eigenvalues(self):
        #self.eigenvalues = np.array(self.eigenvalues,dtype='d')
        self.block_evalues = np.real(self.block_evalues)
        pylab.figure()
        self.block_histogram, bins = np.histogram(self.block_evalues ,bins=self.nbins)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.block_histogram, align='center', width=width)
        pylab.savefig('blocks_histogram.eps')
        if self.show:
            pylab.show()
        pylab.figure()
        self.block_histogram_low, bins = np.histogram(self.block_evalues ,bins=self.nbins_low, range=self.low_range)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.block_histogram_low, align='center', width=width)
        pylab.savefig('blocks_histogram_low{}.eps'.format(self.low_range[1]))
        if self.show:
            pylab.show()
        
        self.whole_evalues = np.real(self.whole_evalues)
        pylab.figure()
        self.whole_histogram, bins = np.histogram(self.whole_evalues ,bins=self.nbins)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.whole_histogram, align='center', width=width)
        pylab.savefig('whole_histogram.eps')
        if self.show:
            pylab.show()
        pylab.figure()
        self.whole_histogram_low, bins = np.histogram(self.whole_evalues ,bins=self.nbins_low, range=self.low_range)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        pylab.bar(center, self.whole_histogram_low, align='center', width=width)
        pylab.savefig('whole_histogram_low{}.eps'.format(self.low_range[1]))
        if self.show:
            pylab.show()
    
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        self._print_parameters()
    
    def _print(self, n):
        """print eigenvalues"""
    
    def _print_parameters(self):
        """writes the simulation parameters"""
        fname = '{}/analyse_jammed_packing.config'.format(self.base_directory)
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
            f.write('{} '.format(val))
        f.write('\n')
        assert(self.sca >0)
        f.write('sca: {}\n'.format(self.sca))
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
        f.close()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="analyse hard disks/spheres packings")
    parser.add_argument("-e","--etol", type=float, help="tolerance on particles eigenvalues, if eval < etol particle will be considered a rattler",default=1.0)
    parser.add_argument("--show", action='store_true', help="show histograms",default=False)
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="jammed_packings")
    args = parser.parse_args()
    print args
    
    analyse = analyse_jammed_packings(etol=args.etol, packings_dir=args.packingsdir, hist_show=args.show)
    start=time.time()
    analyse.run()
    end=time.time()
    print "time elapsed",end-start
    run_gui(analyse.system, analyse.db)
    
    
        
                
            
              
                
                
                
