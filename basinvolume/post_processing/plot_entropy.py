from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from itertools import cycle
    from basinvolume.utils import log_factorial
    from scipy.stats import t
    from scipy.interpolate import spline
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

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

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
    sorted_points = sorted(points)
    new_x = np.array([point[0] for point in sorted_points])
    new_y = np.array([point[1] for point in sorted_points])
    return new_x, new_y

class plot_entropy(object):
    def __init__(self, workdir=None):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir
    
        self.analysis_folder = "entropy_analysis_all"
        self.apf_file = ["entropy_AFP", "ENTROPY_APF"]
        self.kd_file = ["entropy_kernel_density", "LOG_OMEGA_KERNEL_DENSITY"]
        self.lo_file = ["entropy_LogOmega", "ENTROPY_LOG_OMEGA"]
        self.loml_file = ["entropy_ML_LogOmega", "LOG_OMEGA_ML"]
        self.apf_entropy = []
        self.kd_entropy = []
        self.lo_entropy = []
        self.lo_parameters = [] #mu,alpha,zeta
        self.loml_entropy = []
        self.loml_parameters = [] #mu,alpha,zeta
        markers = ["bo", "r^", "gs", "kx", "c+"]
        self.markercycler = cycle(markers)
        
        subdirs = get_immediate_subdirectories(workdir)
        for folder in subdirs:
            if folder[1].isdigit():
                path = os.path.join(workdir,folder,self.analysis_folder)
                if os.path.isdir(path):
                    configf = ConfigParser.ConfigParser()
                    n = _read_nparticles(folder)
                    #AFP
                    fpath = os.path.join(path,self.apf_file[0])
                    if os.path.isfile(fpath):
                        configf.read(fpath)
                        self.apf_entropy.append(n)
                        self.apf_entropy.append(configf.getfloat(self.apf_file[1],'S_star'))
                        self.apf_entropy.append(configf.getfloat(self.apf_file[1],'error_S_star'))
                    #kernel density log omega
                    fpath = os.path.join(path,self.kd_file[0])
                    if os.path.isfile(fpath):
                        configf.read(fpath)
                        self.kd_entropy.append(n)
                        self.kd_entropy.append(configf.getfloat(self.kd_file[1],'S_star'))
                        self.kd_entropy.append(configf.getfloat(self.kd_file[1],'error_S_star'))
                    #log omega
                    fpath = os.path.join(path,self.lo_file[0])
                    if os.path.isfile(fpath):
                        configf.read(fpath)
                        self.lo_entropy.append(n)
                        self.lo_entropy.append(configf.getfloat(self.lo_file[1],'S_star'))
                        self.lo_entropy.append(configf.getfloat(self.lo_file[1],'error_S_star'))
                        self.lo_parameters.append(n)
                        self.lo_parameters.append(configf.getfloat(self.lo_file[1],'mu'))
                        self.lo_parameters.append(configf.getfloat(self.lo_file[1],'mu_error'))
                        self.lo_parameters.append(configf.getfloat(self.lo_file[1],'alpha'))
                        self.lo_parameters.append(configf.getfloat(self.lo_file[1],'alpha_error'))
                        self.lo_parameters.append(configf.getfloat(self.lo_file[1],'zeta'))
                        self.lo_parameters.append(configf.getfloat(self.lo_file[1],'zeta_error'))
                    #log omega ML
                    fpath = os.path.join(path,self.loml_file[0])
                    if os.path.isfile(fpath):
                        configf.read(fpath)
                        self.loml_entropy.append(n)
                        self.loml_entropy.append(configf.getfloat(self.loml_file[1],'S_star'))
                        self.loml_parameters.append(n)
                        self.loml_parameters.append(configf.getfloat(self.loml_file[1],'mu'))
                        self.loml_parameters.append(configf.getfloat(self.loml_file[1],'alpha'))
                        self.loml_parameters.append(configf.getfloat(self.loml_file[1],'zeta'))
        
        self.all_entropies_err = [(self.apf_entropy,r"\sum p \ln p"), (self.kd_entropy,r"\ln \Omega_{KDE}"), 
                             (self.lo_entropy,r"\ln \Omega_G")]
        self.all_entropies = [(self.loml_entropy,r"\ln \Omega_GML")]
        
    def _ploterr(self, entropy_array, xlabel="N", ylabel="S", title=None, show=False):
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
        fit = np.polyfit(x, y, 1, w=1./np.array(yerr))
        ynew = trialx * fit[0] + fit[1] 
        ax.errorbar(x, y, yerr=yerr, fmt='bo', ms=8, label=r"S-logN")
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #raw
        y = np.array(entropy_array[1::3])
        x,y = _sort_pair(nparticles,y)
        fit = np.polyfit(x, y, 1, w=1./np.array(yerr))
        ynew = trialx * fit[0] + fit[1] 
        ax.errorbar(x, y, yerr=yerr, fmt='r^', ms=8, label=r"S")
        ax.plot(trialx,ynew,'r--', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(frameon=False, loc=2)
        if title:
            plt.title(title)
        if show:
            plt.show()
            
    def _plot(self, entropy_array, xlabel="N", ylabel="S", title=None, show=False):
        nparticles = entropy_array[::2]
        nmax = np.amax(nparticles)
        trialx = np.linspace(0,nmax,1000)
        fig = plt.figure()
        #extensive
        ax = fig.add_subplot(111)
        y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
        fit = np.polyfit(nparticles, y, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.errorbar(nparticles, y, fmt='bo', ms=8, label=r"S-logN")
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #raw
        y = np.array(entropy_array[1::2])
        fit = np.polyfit(nparticles, y, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.errorbar(nparticles, y, fmt='r^', ms=8, label=r"S")
        ax.plot(trialx,ynew,'r--', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(frameon=False, loc=2)
        if title:
            plt.title(title)
        if show:
            plt.show()
    
    def plot_single(self, show=False):
        for item in self.all_entropies_err:
            entropy_array, label = item
            self._ploterr(entropy_array, ylabel=label)
        for item in self.all_entropies:
            entropy_array, label = item
            self._plot(entropy_array, ylabel=label)
        if show:
            plt.show()
    
    def plot_all(self, xlabel="N", ylabel="S-logN", title=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for item in self.all_entropies_err:
            entropy_array, label = item
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
            entropy_array, label = item
            nparticles = np.array(entropy_array[0::2])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0,nmax,1000)
            #extensive
            y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
            xa,y = _sort_pair(nparticles,y)
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
    
    def plot_lo_param(self, xlabel="N", show=False):
        nparticles = np.array(self.lo_parameters[::7])
        nparticlesml = np.array(self.loml_parameters[::4])
        nmax = np.amax(nparticles)
        nmin = np.amin(nparticles)
        trialx = np.linspace(0,nmax,1000)
        #mu
        mu = np.array(self.lo_parameters[1::7])
        mu_err = np.array(self.lo_parameters[2::7])
        xa,mu = _sort_pair(nparticles,mu)
        x, mu_err = _sort_pair(nparticles,mu_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(x, mu, yerr=mu_err, fmt='bo', ms=8, label=r"$\mu$")        
        fit = np.polyfit(x, mu, 1, w=1./np.array(mu_err))
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #muML
        mu = np.array(self.loml_parameters[1::4])
        xa,mu = _sort_pair(nparticlesml,mu)
        ax.errorbar(x, mu, fmt='r^', ms=8, label=r"$\mu_{ML}$")        
        fit = np.polyfit(x, mu, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'r--', linewidth=2)
        ax.legend(frameon=False, loc=2)
        plt.xlabel(xlabel) #plot
        plt.ylabel(r"$\mu$")
        #alpha
        alpha = np.array(self.lo_parameters[3::7])
        alpha_err = np.array(self.lo_parameters[4::7])
        xa,alpha = _sort_pair(nparticles,alpha)
        x, alpha_err = _sort_pair(nparticles,alpha_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(x, alpha, yerr=alpha_err, fmt='bo', ms=8, label=r"$\alpha$")        
        fit = np.polyfit(x, alpha, 1, w=1./np.array(alpha_err))
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #alphaML
        alpha = np.array(self.loml_parameters[2::4])
        xa,alpha = _sort_pair(nparticlesml,alpha)
        ax.errorbar(x, alpha, fmt='r^', ms=8, label=r"$\alpha_{ML}$")        
        fit = np.polyfit(x, alpha, 1)
        ynew = trialx * fit[0] + fit[1] 
        ax.plot(trialx,ynew,'r--', linewidth=2)
        ax.legend(frameon=False, loc=2)
        plt.xlabel(xlabel)
        plt.ylabel(r"$\alpha$")
        #zeta
        trialx = np.linspace(0,1/nmin,1000)
        zeta = np.array(self.lo_parameters[5::7])
        zeta_err = np.array(self.lo_parameters[6::7])
        xa,zeta = _sort_pair(nparticles,alpha)
        x, zeta = _sort_pair(nparticles,alpha_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        ax.errorbar(1./x, 2-zeta, yerr=zeta_err, fmt='bo', ms=8, label=r"$2-\zeta$")
        fit = np.polyfit(1./x, 2-zeta, 1, w=1./np.array(zeta_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx,ynew,'b--', linewidth=2)
        #zetaML
        zeta = np.array(self.loml_parameters[3::4])
        xa,zeta = _sort_pair(nparticlesml,zeta)
        ax.errorbar(1./x, 2-zeta, fmt='r^', ms=8, label=r"$2-\zeta_{ML}$")        
        fit = np.polyfit(1./x, 2-zeta, 1)
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
                            
                
                
                
if __name__ == "__main__":
    pe = plot_entropy()
    pe.plot_single(show=True)
    #pe.plot_all(show=True)
    #pe.plot_lo_param(show=True)


