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

def plot_entropy(workdir=None):
    if not workdir:
        workdir = os.getcwd()
    if not os.path.isabs(workdir):
        workdir = os.path.abspath(workdir)
    analysis_folder = "entropy_analysis_all"
    apf_file = ["entropy_AFP", "ENTROPY_APF"]
    kd_file = ["entropy_kernel_density", "LOG_OMEGA_KERNEL_DENSITY"]
    lo_file = ["entropy_LogOmega", "ENTROPY_LOG_OMEGA"]
    loml_file = ["entropy_ML_LogOmega", "LOG_OMEGA_ML"]
    apf_entropy = []
    kd_entropy = []
    lo_entropy = []
    lo_parameters = [] #mu,alpha,zeta
    loml_entropy = []
    loml_parameters = [] #mu,alpha,zeta
    
    subdirs = get_immediate_subdirectories(workdir)
    for folder in subdirs:
        if folder[1].isdigit():
            path = os.path.join(workdir,folder,analysis_folder)
            if os.path.isdir(path):
                configf = ConfigParser.ConfigParser()
                n = _read_nparticles(folder)
                #AFP
                fpath = os.path.join(path,apf_file[0])
                if os.path.isfile(fpath):
                    configf.read(fpath)
                    apf_entropy.append(n)
                    apf_entropy.append(configf.getfloat(apf_file[1],'S_star'))
                    apf_entropy.append(configf.getfloat(apf_file[1],'error_S_star'))
                #kernel density log omega
                fpath = os.path.join(path,kd_file[0])
                if os.path.isfile(fpath):
                    configf.read(fpath)
                    kd_entropy.append(n)
                    kd_entropy.append(configf.getfloat(kd_file[1],'S_star'))
                    kd_entropy.append(configf.getfloat(kd_file[1],'error_S_star'))
                #log omega
                fpath = os.path.join(path,lo_file[0])
                if os.path.isfile(fpath):
                    configf.read(fpath)
                    lo_entropy.append(n)
                    lo_entropy.append(configf.getfloat(lo_file[1],'S_star'))
                    lo_entropy.append(configf.getfloat(lo_file[1],'error_S_star'))
                    lo_parameters.append(configf.getfloat(lo_file[1],'mu'))
                    lo_parameters.append(configf.getfloat(lo_file[1],'mu_error'))
                    lo_parameters.append(configf.getfloat(lo_file[1],'alpha'))
                    lo_parameters.append(configf.getfloat(lo_file[1],'alpha_error'))
                    lo_parameters.append(configf.getfloat(lo_file[1],'zeta'))
                    lo_parameters.append(configf.getfloat(lo_file[1],'zeta_error'))
                #log omega ML
                fpath = os.path.join(path,loml_file[0])
                if os.path.isfile(fpath):
                    configf.read(fpath)
                    loml_entropy.append(n)
                    loml_entropy.append(configf.getfloat(loml_file[1],'S_star'))
                    loml_parameters.append(configf.getfloat(loml_file[1],'mu'))
                    loml_parameters.append(configf.getfloat(loml_file[1],'alpha'))
                    loml_parameters.append(configf.getfloat(loml_file[1],'zeta'))
    
    all_entropies_err = [(apf_entropy,r"\sum p \ln p"), (kd_entropy,r"\ln \Omega_{KDE}"), (lo_entropy,r"\ln \Omega_G")]
    all_entropies = [(loml_entropy,r"\ln \Omega_GML")]
    markers = ["bo", "r^", "gs", "kx", "c+"]
    markercycler = cycle(markers)
    
    def _ploterr(entropy_array, xlabel="N", ylabel="S", title=None, show=False):
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
            
    def _plot(entropy_array, xlabel="N", ylabel="S", title=None, show=False):
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
    
    def _plot_all(xlabel="N", ylabel="S-logN", title=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for item in all_entropies_err:
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
            m = next(markercycler)
            ax.errorbar(x, y, yerr=yerr, fmt=m, ms=8, label=label)
            ax.plot(trialx,ynew,m[0]+'--', linewidth=2)
        #loop over entropies without an associated error
        for item in all_entropies:
            entropy_array, label = item
            nparticles = np.array(entropy_array[0::2])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0,nmax,1000)
            #extensive
            y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
            xa,y = _sort_pair(nparticles,y)
            fit = np.polyfit(x, y, 1)
            ynew = trialx * fit[0] + fit[1]
            m = next(markercycler)
            ax.errorbar(x, y, fmt=m, ms=8, mew=2, label=label)
            ax.plot(trialx,ynew,m[0]+'--', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(frameon=False, loc=2)
        if title:
            plt.title(title)
        if show:
            plt.show()
    
#    _ploterr(apf_entropy, show=False, ylabel="$\sum p \ln p$")
#    _ploterr(kd_entropy, show=False, ylabel="$\ln \Omega_{KDE}$")
#    _ploterr(lo_entropy, show=False, ylabel="$\ln \Omega_{G}$")
#    _plot(loml_entropy, show=False, ylabel="$\ln \Omega_{GML}$")
    _plot_all(show=True)
    
                    
                
if __name__ == "__main__":
    plot_entropy()


