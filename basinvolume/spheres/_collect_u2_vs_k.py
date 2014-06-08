from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import Harmonic
from basinvolume.spheres import Findk_MCrunner
from basinvolume.utils import trymakedir, read_xyzdr, read_xydr
import ConfigParser
import time
from basinvolume.post_processing import F_Basin_From_MC_Data, F_Basin_From_MC_Data_Free_COM, Gauss_Lobatto_abscissas
import pylab as plt
import argparse

class _collect_u2_vs_k(object):
    """
    this is a class that implements _collect_u2_vs_k class 
    """
        
    def __call__(self, fname='explore_bv_jammed_packing0', base_dir='analysis', explore_dir='explore_bv_', packings_dir='jammed_packings'):
               
        self.fname = fname
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        if not os.path.isabs(explore_dir):
            explore_dir = os.path.join(os.getcwd(),explore_dir+fname)
        self.explore_dir = explore_dir
        self.base_directory = self.explore_dir + '/' + base_dir
        
        self.packing_configpath = os.path.join(packings_dir,'jammed_packings.config')
        self.findk_configpath = os.path.join(self.explore_dir,'findk_'+fname+'.config')  
        self.kmin_configpath = os.path.join(self.explore_dir,'kmin_'+fname+'.config')
        
        self._import_config_files()
        self.run()
    
    def run(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        self._import_ks()
        self._import_u2()
        self._print_u2_vs_k()
        self._compute_volume()
        self._plot_data()
    
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
        configf.read(str(self.findk_configpath))tennis
        self.kmax = configf.getfloat('FINDK','kmax')
        self.prob_kmax = configf.getfloat('FINDK','prob')
        self.displ_k_max = configf.getfloat('FINDK','displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK','var_displ_k_max')
        configf.read(str(self.kmin_configpath))
        self.kmin = configf.getfloat('KMIN_MCRUNNER','k')
        self.displ_k_min = configf.getfloat('KMIN','displ_k_min')
        self.var_displ_k_min = configf.getfloat('KMIN','var_displ_k_min')
    
    def _import_ks(self):
        """
        must run before import u2
        """
        karray = [] 
        path = os.path.join(self.explore_dir,'temperatures')
        f = open(path, "r")
        while True:
            k = f.readline()
            if not k: break
            karray.extend([float(k)])
        karray.extend([self.kmax])
        self.karray = np.array(karray,dtype='d')

    def _import_u2_reverse(self):
        n = len(self.karray)
        self.u2_array = [0 for _ in xrange(n)]
        self.var_array = [0 for _ in xrange(n)] 
        for subdir, dirs, files in os.walk(self.explore_dir):
            for dir in dirs:
                if dir.isdigit():
                    path = os.path.join(self.explore_dir,dir+'/hist_mean')
                    fileHandle = open (path,"r")
                    lineList = fileHandle.readlines()
                    fileHandle.close()
                    niter, u2, var = lineList[-1].split()
                    self.u2_array[int(dir)] = u2
                    self.var_array[int(dir)] = var
        #extend with kmin
        self.u2_array.extend([self.displ_k_max])
        self.var_array.extend([self.var_displ_k_max])
        print self.u2_array
        
         
    def _print_u2_vs_k(self):
        """writes <u2> and variance vs """
        dname = 'u2_vs_k'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#{:>15}\t{:>15}\n'.format('<u2>','var(<u2>)'))
        for i in xrange(len(self.u2_array)):
            f.write('{:>15.15e}\t{:>15.15e}\n'.format(self.u2_array[i],self.var_array[i])) 
        f.close()
        
    def _compute_volume(self):
        """
        numerical volume obtained 
        """
        
        self.F0, self.sigF0, self.farray, self.sigfarray = F_Basin_From_MC_Data(self.bdim, self.nparticles, self.karray,\
                                                                                self.u2_array, np.prod(self.boxv),\
                                                                                self.prob_kmax).get_free_energy_F0(self.var_array)
        
        self.F0unc, self.sigF0unc, self.farrayunc, self.sigfarrayunc= F_Basin_From_MC_Data_Free_COM(self.bdim, self.nparticles, self.karray,\
                                                                                self.u2_array, np.prod(self.boxv),\
                                                                                self.prob_kmax).get_free_energy_F0(self.var_array)
        self.tarray = Gauss_Lobatto_abscissas(len(self.u2_array))()
        rF0 = self.F0 + self.nparticles*np.log(np.prod(self.boxv))
        rF0unc = self.F0unc + self.nparticles*np.log(np.prod(self.boxv))
        print 'rF0 {} rF0unc {}'.format(rF0, rF0unc)

    def _plot_data(self):
        cont_karray = np.linspace(self.kmin, self.kmax, 100)
        u2_array_app = (cont_karray + (self.nparticles*self.bdim)/self.displ_k_min) / (self.nparticles*self.bdim)
        u2_array_app = 1.0/u2_array_app
        
        plt.figure()
        plt.errorbar(self.tarray,self.farray,yerr=self.sigfarray)
        plt.xlabel('t')
        plt.ylabel('integrand')
        plt.savefig('')
        plt.figure(self.base_directory+'/integrand.eps')
        plt.plot(cont_karray,u2_array_app,'-')
        plt.errorbar(self.karray,self.u2_array,yerr=np.sqrt(self.var_array),marker='s',linestyle='')
        plt.xlabel('k')
        plt.ylabel('<u2>')
        plt.ylim(bottom=0)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.figure(self.base_directory+'/u2_vs_k.eps')
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="analyze PT data from thermodynamic integration")
    #parser.add_argument("nparticles", type=int, help="number of particles")
    parser.add_argument("-f","--fname", type=str, help="specify packing to analyze",default=None)
    parser.add_argument("-d","--fdir", type=str, help="directory containing file, if not absolute path by default: fdir+fname",default='explore_bv_')
    parser.add_argument("-w","--workdir", type=str, help="directory containing PT data (all) must be absolute, default chwdir",default=os.getcwd())
    args = parser.parse_args()
    print args
    
    fname = args.fname
    fdir = args.fdir
    wdir = args.w
    assert(os.path.isabs(wdir))
    
    if not os.path.isabs(fdir):
        fdir = os.path.join(wdir,fdir+fname)
    
    sim = _collect_u2_vs_k()
    
    if (fname != None):
        sim(fname=fname,explore_dir=fdir)
    else :
        for subdir, dirs, files in os.walk(wdir):
            for dir in dirs:
                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
                    path = os.path.join(wdir,dir)
                    sim(explore_dir=path)
                    
            
    
