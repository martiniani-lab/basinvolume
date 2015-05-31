from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from itertools import cycle
    from basinvolume.utils import *
    import scipy
    from scipy.stats import t
    from scipy.interpolate import spline
    from itertools import chain
except ImportError as err:
    print err

#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
plt.rcParams.update({'font.size': 16})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
##########################################################
"""
for plotting a linear fit with intervals of confidence see http://nbviewer.ipython.org/url/bagrow.com/dsv/LEC10_notes_2014-02-13.ipynb
"""
def _read_nparticles(folder):
    nparticles = ""
    for char in folder[1:]:
        if char == "_":
            break
        nparticles+=char
    nparticles = int(nparticles)
    return nparticles

def studentTCI(y, yerr, alpha=0.95):
    """
    return lower and upper bound of confidence interval
    see https://en.wikipedia.org/wiki/Confidence_interval Thoretical example
    """
    y = np.array(y)
    yerr = np.array(yerr)
    ts = t.interval(np.array([alpha]*len(y)), len(y)-1)
    return y + alpha*ts[0]*yerr, y + alpha*ts[1]*yerr 

def _sort_pair(x,y):
    """
    sorts x and moves elements of y accordingly
    """
    xc = np.array(x)
    points = zip(xc,y)
    sorted_points = sorted(points, key=lambda x: x[0])
    new_x = np.array([point[0] for point in sorted_points])
    new_y = np.array([point[1] for point in sorted_points])
    return new_x, new_y

class EntropyData(object):
    def __init__(self, nparticles, path, 
                 apf_file="entropy_APF", apf_file_title="ENTROPY_APF",
                 kd_file="entropy_kernel_density", kd_file_title="LOG_OMEGA_KERNEL_DENSITY",
                 lo_file="entropy_LogOmega", lo_file_title="ENTROPY_LOG_OMEGA",
                 loml_file="entropy_ML_LogOmega", loml_file_title="LOG_OMEGA_ML",
                 gen_gaussian_param_names = [("mu","mu_error"), ("alpha","alpha_error"), ("zeta","zeta_error")]):
        self.nparticles = nparticles
        self.path = path
        self.apf = Bunch(file=apf_file, title=apf_file_title, entropy=[0.,0.])
        self.kd = Bunch(file=kd_file, title=kd_file_title, entropy=[0.,0.])
        parameters = dict(mu=[0.,0.],alpha=[0.,0.],zeta=[0.,0.])
        self.lo = Bunch(file=lo_file, title=lo_file_title, entropy=[0.,0.], parameters=parameters)        #parameters are mu,alpha,zeta
        self.loml = Bunch(file=loml_file, title=loml_file_title, entropy=[0.,0.], parameters=parameters)  #parameters are mu,alpha,zeta
        self.gen_gaussian_param_names = gen_gaussian_param_names
    
    def add_all_entropy(self):
        self.add_entropy(self.apf)
        self.add_entropy(self.kd)
        self.add_entropy_with_parameters(self.lo)
        self.add_entropy_with_parameters(self.loml)
    
    def add_entropy(self, member):
        """
        member should be the appropriate class member
        n : packing number
        example
        -------
        data = EntropyData()
        data.add_entropy(S, ds, data.apf)
        """
        configf = ConfigParser.ConfigParser()
        fpath = os.path.join(self.path, member.file)
        if os.path.isfile(fpath):
            configf.read(fpath)
            try:
                member.entropy[0] = configf.getfloat(member.title,'S_star')
                member.entropy[1] = configf.getfloat(member.title,'error_S_star')
            except Exception, e:
                print e
         
    def add_entropy_with_parameters(self, member):
        """
        member should be the appropriate class member
        parameters: list ot tuples
            (value, error)
        n : packing number
        example
        -------
        data = EntropyData()
        data.add_entropy(S, ds, data.apf)
        """
        configf = ConfigParser.ConfigParser()
        fpath = os.path.join(self.path, member.file)
        if os.path.isfile(fpath):
            configf.read(fpath)
            try:
                member.entropy[0] = configf.getfloat(member.title,'S_star')
                member.entropy[1] = configf.getfloat(member.title,'error_S_star')
            except Exception, e:
                print e
            for name in self.gen_gaussian_param_names:
                try:
                    val = configf.getfloat(member.title, name[0])
                    member.parameters[name[0]][0] = val
                    errval = configf.getfloat(member.title, name[1])
                    member.parameters[name[0]][1] = errval
                except Exception:
                    pass

class plot_entropy(object):
    def __init__(self, workdir=None, analysis_folder="entropy_analysis_all", Nrange=(0,128)):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir
        
        self.Nrange = Nrange
        self.analysis_folder = analysis_folder
        self.entropy_data = []
        markers = ["bo", "r^", "gs", "kx", "c+"]
        self.markercycler = cycle(markers)
        
        subdirs = get_immediate_subdirectories(workdir)
        for folder in subdirs:
            if folder[1].isdigit():
                path = os.path.join(workdir,folder, self.analysis_folder)
                if os.path.isdir(path):
                    n = _read_nparticles(folder)
                    if self.Nrange[0] <= n <= self.Nrange[1]:
                        data = EntropyData(n, path)
                        data.add_all_entropy()
                        self.entropy_data.append(data)
        self.apf_entropy = list(chain.from_iterable((data.nparticles, data.apf.entropy[0], data.apf.entropy[1]) for data in self.entropy_data))
        self.kd_entropy = list(chain.from_iterable((data.nparticles, data.kd.entropy[0], data.kd.entropy[1]) for data in self.entropy_data))
        self.lo_entropy = list(chain.from_iterable((data.nparticles, data.lo.entropy[0], data.lo.entropy[1]) for data in self.entropy_data))
        self.loml_entropy = list(chain.from_iterable((data.nparticles, data.loml.entropy[0]) for data in self.entropy_data)) 
        self.all_entropies_err = [(self.apf_entropy,r"$\sum p \ln p$","apf"), (self.kd_entropy,r"$\log \Omega_{KDE}$","kde"), 
                             (self.lo_entropy,r"$\log \Omega_G$", "logomega")]
        self.all_entropies = [(self.loml_entropy,r"$\log \Omega_{GML}$","logomegaml")]
        
        self.lo_parameters = list(chain.from_iterable((data.nparticles, 
                                                       data.lo.parameters['mu'][0], data.lo.parameters['mu'][1], 
                                                       data.lo.parameters['alpha'][0], data.lo.parameters['alpha'][1], 
                                                       data.lo.parameters['zeta'][0], data.lo.parameters['zeta'][1]) for data in self.entropy_data))
        self.loml_parameters = list(chain.from_iterable((data.nparticles,
                                                         data.loml.parameters['mu'][0], data.loml.parameters['alpha'][0], 
                                                         data.loml.parameters['zeta'][0]) for data in self.entropy_data))
          
    def _ploterr(self, entropy_array, xlabel=r"$N$", ylabel=r"$S$", raw=True, title=None, show=False):
        nparticles = np.array(entropy_array[0::3])
        nmax = np.amax(nparticles)
        trialx = np.linspace(0,nmax,1000)
        fig = plt.figure()
        #extensive
        ax = fig.add_subplot(111)
        y = np.array(entropy_array[1::3]) - log_factorial(np.array(nparticles))
        yerr = np.array(entropy_array[2::3])
        xa,y = _sort_pair(nparticles,y)
        x, yerr = _sort_pair(nparticles,yerr)
        #print ylabel, len(x), len(y)
        #print x,"\n", y
        fit = np.polyfit(x, y, 1, w=1./np.array(yerr))
        ynew = trialx * fit[0] + fit[1]        
        ax.errorbar(x, y, yerr=yerr, fmt='bo', ms=9, label=r"$S^\star -\log N!$")
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #raw
        if raw:
            y = np.array(entropy_array[1::3])
            x,y = _sort_pair(nparticles,y)
            fit = np.polyfit(x, y, 1, w=1./np.array(yerr))
            ynew = trialx * fit[0] + fit[1] 
            ax.errorbar(x, y, yerr=yerr, fmt='r^', ms=9, label=r"$S^\star$")
            ax.plot(trialx,ynew,'r--', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(frameon=False, loc=2)
        if title:
            plt.title(title)
        if show:
            plt.show()
        return ax
            
    def _plot(self, entropy_array, xlabel=r"$N$", ylabel=r"$S$", raw=True, title=None, show=False):
        nparticles = entropy_array[::2]
        nmax = np.amax(nparticles)
        trialx = np.linspace(0,nmax,1000)
        fig = plt.figure()
        #extensive
        ax = fig.add_subplot(111)
        y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
        fit = np.polyfit(nparticles, y, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.errorbar(nparticles, y, fmt='bo', ms=9, label=r"$S^\star -\log N!$")
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #raw
        if raw:
            y = np.array(entropy_array[1::2])
            fit = np.polyfit(nparticles, y, 1)
            ynew = trialx * fit[0] + fit[1] 
            ax.errorbar(nparticles, y, fmt='r^', ms=9, label=r"$S^\star$")
            ax.plot(trialx,ynew,'r--', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(frameon=False, loc=2)
        if title:
            plt.title(title)
        if show:
            plt.show()
        return ax
    
    def plot_single(self, show=False, savefig=True):
        for item in self.all_entropies_err:
            entropy_array, label, plot_label = item
            self._ploterr(entropy_array, ylabel=label)
            if savefig:
                plt.savefig('plot_{}.pdf'.format(plot_label))
        for item in self.all_entropies:
            entropy_array, label, plot_label = item
            self._plot(entropy_array, ylabel=label)
            if savefig:
                plt.savefig('plot_{}.pdf'.format(plot_label))
        if show:
            plt.show()
    
    def plot_all(self, xlabel=r"$N$", ylabel=r"$S^\star -\log N!$", title=None, show=False, savefig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for item in self.all_entropies_err:
            entropy_array, label, plot_label = item
            nparticles = np.array(entropy_array[0::3])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0,nmax,1000)
            #extensive
            y = np.array(entropy_array[1::3]) - log_factorial(np.array(nparticles))
            yerr = np.array(entropy_array[2::3])
            xa,y = _sort_pair(nparticles,y)
            x, yerr = _sort_pair(nparticles,yerr)
            fit = np.polyfit(x, y, 1, w=1./np.array(yerr))
            ynew = trialx * fit[0] + fit[1]
            m = next(self.markercycler)
            ax.errorbar(x, y, yerr=yerr, fmt=m, ms=8, label=label)
            ax.plot(trialx,ynew,m[0]+'--', linewidth=2)
        #loop over entropies without an associated error
        for item in self.all_entropies:
            entropy_array, label, plot_label = item
            nparticles = np.array(entropy_array[0::2])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0,nmax,1000)
            #extensive
            y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
            x,y = _sort_pair(nparticles,y)
            fit = np.polyfit(x, y, 1)
            ynew = trialx * fit[0] + fit[1]
            m = next(self.markercycler)
            ax.errorbar(x, y, fmt=m, ms=8, mew=2, label=label)
            ax.plot(trialx,ynew,m[0]+'--', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(frameon=False, loc=2)
        if title:
            plt.title(title)
        if show:
            plt.show()
        if savefig:
            plt.savefig('compare_all.pdf')
        return ax
    
    def plot_lo_param(self, xlabel=r"$N$", show=False, savefig=False):
        nparticles = np.array(self.lo_parameters[::7])
        nparticlesml = np.array(self.loml_parameters[::4])
        nmax = np.amax(np.append(nparticles,nparticlesml))
        nmin = np.amin(np.append(nparticles,nparticlesml))
        trialx = np.linspace(0,nmax,1000)
        #mu
        mu = np.array(self.lo_parameters[1::7])
        mu_err = np.array(self.lo_parameters[2::7])
        xa, mu = _sort_pair(nparticles,mu)
        x, mu_err = _sort_pair(nparticles,mu_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(x, mu, yerr=mu_err, fmt='bo', ms=9, label=r"$\mu$")        
        fit = np.polyfit(x, mu, 1, w=1./np.array(mu_err))
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #muML
        mu = np.array(self.loml_parameters[1::4])
        x,mu = _sort_pair(nparticlesml,mu)
        ax.errorbar(x, mu, fmt='r^', ms=9, label=r"$\mu_{ML}$")        
        fit = np.polyfit(x, mu, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'r--', linewidth=2)
        ax.legend(frameon=False, loc=2)
        plt.xlabel(xlabel) #plot
        plt.ylabel(r"$\mu$")
        if savefig:
            plt.savefig('lo_mu.pdf')
        #alpha
        alpha = np.array(self.lo_parameters[3::7])
        alpha_err = np.array(self.lo_parameters[4::7])
        xa,alpha = _sort_pair(nparticles,alpha)
        x, alpha_err = _sort_pair(nparticles,alpha_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(x, alpha, yerr=alpha_err, fmt='bo', ms=9, label=r"$\alpha$")        
        fit = np.polyfit(x, alpha, 1, w=1./np.array(alpha_err))
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #alphaML
        alpha = np.array(self.loml_parameters[2::4])
        x,alpha = _sort_pair(nparticlesml, alpha)
        ax.errorbar(x, alpha, fmt='r^', ms=9, label=r"$\alpha_{ML}$")        
        fit = np.polyfit(x, alpha, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'r--', linewidth=2)
        ax.legend(frameon=False, loc=2)
        plt.xlabel(xlabel)
        plt.ylabel(r"$\alpha$")
        if savefig:
            plt.savefig('lo_alpha.pdf')
        #zeta
        trialx = np.linspace(0,1/nmin,1000)
        zeta = np.array(self.lo_parameters[5::7])
        zeta_err = np.array(self.lo_parameters[6::7])
        xa,zeta = _sort_pair(nparticles, zeta)
        x, zeta_err = _sort_pair(nparticles, zeta_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        szeta = 2-zeta        
        ax.errorbar(1./x, szeta, yerr=zeta_err, fmt='bo', ms=8, label=r"$2-\zeta$")
        fit = np.polyfit(1./x, szeta, 1, w=1./np.array(zeta_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #zetaML
        zeta = np.array(self.loml_parameters[3::4])
        x, zeta = _sort_pair(nparticlesml, zeta)
        szeta = 2-zeta
        ax.errorbar(1./x, szeta, fmt='r^', ms=8, label=r"$2-\zeta_{ML}$")        
        fit = np.polyfit(1./x, szeta, 1, w=1./np.array(zeta_err))
        ynew = trialx * fit[0] + fit[1] 
        plt.xlabel(xlabel)
        plt.ylabel(r"$\zeta_{ML}$")
        ax.legend(frameon=False, loc=2)
        ax.plot(trialx,ynew,'r--', linewidth=2)
        plt.xlabel(r"1/N")
        plt.ylabel(r"$2-\zeta$")
        ax.legend(frameon=False, loc=2)
        if show:
            plt.show()
        if savefig:
            plt.savefig('lo_zeta.pdf')
            
    def plot_compare_apf2D(self, ax=None, show=False, savefig=True):
        """
        plot a comparison to the data provided by D. Asenjo for 2D packings
        for data with soft to hard ration 1.12
        """
        fpath = os.path.join(self.workdir, 'apf_prl_data/N_mean_alpha_beta_dense.dat')
        if not os.path.isfile(fpath):
            raise Exception("{} not a file".format(fpath))
        dat_dense = np.loadtxt(fpath)
        f_ex_dense = 3.39558433477
        kmax_dense = np.array([40000.0, 50000.0, 90000.0, 180000.0, 400000.0])
        f0_dense = dat_dense[-1,1] / 128. - f_ex_dense
        #plot all
        entropy_array, label, plot_label = self.all_entropies_err[0] 
        ax = self._ploterr(entropy_array, title="comparison to PRL 2D data", raw=False, show=False)
        trialx = np.linspace(0,128,1000)
        #d+1/d-1 equation
        ynew = trialx * 1./2 
        ax.plot(trialx, ynew,'r--', linewidth=2, label=r'$\frac{d-1}{d+1}f(\phi)N$')
        ynew = trialx * 1./3 
        ax.plot(trialx, ynew,'r--', linewidth=2)
        #plot apf_prl
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        x = dat_dense[:,0]
        y_apf = dat_dense[:,1] - dat_dense[:,0] * f_ex_dense - dat_dense[:,0] * np.log(dat_dense[:,0]) + \
        dat_dense[:,0] - np.log(dat_dense[:,0]) + np.log(2.*np.pi/kmax_dense)
        y_lo = dat_dense[:,7] + dat_dense[:,1] - dat_dense[:,0] * f_ex_dense - (dat_dense[:,0] * np.log(dat_dense[:,0])) + \
        dat_dense[:,0] - np.log(dat_dense[:,0]) + np.log(2.*np.pi/kmax_dense)
        #2D apf
        ax.plot(x, y_apf, 'b*', markersize=15, label=r"PRL(\sum p \ln p)_{2D}")
        fit = np.polyfit(x, y_apf, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #2D LogOmega
#        ax.plot(x, y_lo, 'g*', markersize=15, label=r"PRL(\ln \Omega_G)_{2D}")
#        fit = np.polyfit(x, y_lo, 1)
#        ynew = trialx * fit[0] + fit[1] 
#        ax.plot(trialx,ynew,'g--', linewidth=2)
        ax.legend(frameon=False, loc=2)
        if show:
            plt.show()
        if savefig:
            plt.savefig('compare_apf_prl.pdf')
    
                
if __name__ == "__main__":
    show=True
    savefig=False
    pe = plot_entropy(analysis_folder="entropy_analysis_all")
    #pe.plot_single(show=True, savefig=True)
    pe.plot_compare_apf2D(show=show,savefig=savefig)
    #pe.plot_all(show=show,savefig=savefig)
    #pe.plot_lo_param(show=show,savefig=savefig)


