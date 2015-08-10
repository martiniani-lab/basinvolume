from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import re
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from itertools import cycle
    from basinvolume.utils import *
    import scipy
    from scipy.stats import t
    from scipy.interpolate import splrep, splev, interp1d
    from scipy.integrate import simps
    import glob
    from basinvolume.post_processing import PackingData, PackingDataSet, BasinAnalysis
    from scipy.optimize import curve_fit
except ImportError as err:
    print err
#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
plt.rcParams.update({'font.size': 18})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
##########################################################
####SET COLOUR MAP######                                                               
def get_color_cycle():
    cm = plt.get_cmap('Set2')
    color_cycle=cycle([cm(1. * i / 7) for i in xrange(7)])
    return color_cycle
########################
#####################LINE STYLE CYCLER####################                             
lines = ["-","--","-.", ":", "_"]
linecycler = cycle(lines)
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

class MBARPackingData(PackingData):
    def __init__(self, name, configpath, configpath_packing, packing_path=None):
        super(MBARPackingData, self).__init__(name, configpath, configpath_packing, packing_path=packing_path)
        self.log_gr = None
        self.log_gr_ratio = None
        self.gr_ratio = None
        self.dos = None
        
    def import_dos_data(self, path, log_gr_file = "log_gr.csv", log_gr_ratio_file = "log_gr_ratio.csv", 
                        gr_ratio_file = "gr_ratio.csv", dos_file = "dos.csv"): 
        """
        I have removed mean, that can be added to the set
        """
        #log_gr
        fpath = os.path.join(path, log_gr_file)
        if os.path.isfile(fpath):
            x, xerr, y, yerr, fit = read_csv_xy(fpath)
            self.log_gr = np.transpose(np.array([x, xerr, y, yerr, fit]))
        #log_gr_ratio
        fpath = os.path.join(path, log_gr_ratio_file)
        if os.path.isfile(fpath):
            x, xerr, y, yerr, fit = read_csv_xy(fpath)
            y -= np.amax(y)
            self.log_gr_ratio = np.transpose(np.array([x, xerr, y, yerr, fit]))
        #gr_ratio
        fpath = os.path.join(path, gr_ratio_file)
        if os.path.isfile(fpath):
            x, xerr, y, yerr, fit = read_csv_xy(fpath)
            self.gr_ratio = np.transpose(np.array([x, xerr, y, yerr, fit]))
        #dos
        fpath = os.path.join(path, dos_file)
        if os.path.isfile(fpath):
            x, xerr, y, yerr, fit = read_csv_xy(fpath)
            self.dos = np.transpose(np.array([x, xerr, y, yerr, fit]))

class MBARPackingDataSet(PackingDataSet):
    def __init__(self, set_path):
        super(MBARPackingDataSet, self).__init__(set_path)
        str_values = re.findall('\d+', self.set_name)
        try:
            self.hs_poly =  float('0.'+str_values[4][1:])
        except Exception:
            print "can't pick up polydispersity, setting to 0. Folder name: ", self.set_name
            self.hs_poly = 0.
        if 'fcc' in self.set_name:
            self.structural_label = 'fcc'
        elif 'disordered' in self.set_name:
            self.structural_label = 'disordered'
        else:
            self.structural_label = None
            print "cannot recognise structural label (fcc or disordered), set to None"
        self.log_gr_data = []
        self.log_gr_ratio_data = [] 
        self.gr_ratio_data = []
        self.dos_data = []
        self.dos_mean = []
        self.log_gr_mean = []
        self.log_gr_ratio_mean = []
        self.gr_ratio_mean = []
        self.dos_moments = []
        
    def add_data_all(self, packing_data):
        """
        packing data is a list of PackingData objects
        """
        self.packing_data.extend(packing_data)
        for data in packing_data:
            #the reason why they must all be true is because we are interested in the relation among these variables
            if (data.F is not None and data.P is not None and data.Z is not None and data.boo is not None and
                data.log_gr is not None and data.log_gr_ratio is not None and data.gr_ratio is not None and
                data.dos is not None):
                self.free_energies.append(data.F)
                self.free_energies_err.append(data.Ferr)
                self.pressures.append(data.P)
                self.contacts.append(data.Z)
                self.boos.append(data.boo)
                self.log_gr_data.append(data.log_gr)
                self.log_gr_ratio_data.append(data.log_gr_ratio) 
                self.gr_ratio_data.append(data.gr_ratio)
                self.dos_data.append(data.dos)

    def _compute_mean(self):
        def get_mean(data):
            #pick the smallest range that works for all curves
            xmin, xmax = data[0][0,0], data[0][-1,0]
            for arr in data:
                nxmin, nxmax = arr[0,0], arr[-1,0]
                if nxmin > xmin:
                    xmin = nxmin
                if nxmax < xmax:
                    xmax = nxmax
            xref = np.linspace(xmin, xmax, len(data[0][:,0]))
            all = []
            for arr in data:
                x, y = arr[:,0], arr[:,2]
                assert x.size == xref.size, 'x array size mismatches'
                #tcky = splev(x, splrep(x, y, s=0), der=0)
                f = interp1d(x, y, bounds_error=False)
                all.append(f(xref))
            all = np.array(all)
            wsum = np.sum(all, axis=0)
            keep = np.nonzero(np.asarray(np.isfinite(wsum), dtype='i')) # * np.array([np.abs(x) > 1e-16 for x in wsum]), dtype='i')
            all = all[:, keep[0]]
            xref = xref[keep[0]]
            assert xref.size == all.shape[1]
            mu = np.mean(all, axis=0)
            std = np.std(all, axis=0)
            return np.transpose(np.array([xref, np.zeros(xref.size), mu, std, np.zeros(xref.size)]))
        
        self.log_gr_mean = get_mean(self.log_gr_data)
        self.log_gr_ratio_mean = get_mean(self.log_gr_ratio_data)  
        self.dos_mean = get_mean(self.dos_data)
        for i,arr in enumerate(self.gr_ratio_data):
            x, y = arr[:,0], arr[:,2]
            j = next(xp[0] for xp in enumerate(x) if xp[1] > 0.1)
            y /= np.mean(y[:j])
            self.gr_ratio_data[i][:,2] = y
        self.gr_ratio_mean = get_mean(self.gr_ratio_data)
    
    def _compute_dos_moments(self):
        def normalize_dist(y, x):
            area = simps(y, x)
            y = np.array(y) / area
            area = simps(y, x)
            y = np.array(y) / area
            return y
        for arr in self.dos_data:
            x, y = arr[:,0], arr[:,2]
            y = normalize_dist(y, x)
            mode =  x[np.argmax(y)]
            mean = np.average(x, weights=y)
            var = np.average((x-mean)**2, weights=y)
            std = np.sqrt(var)
            skewness = np.average((x-mean)**3, weights=y) / std**3
            kurtosis = np.average((x-mean)**4, weights=y) / var**2
            self.dos_moments.append([mode, mean, var, skewness, kurtosis])

    def compute_mean_and_moments(self):
        if len(self.log_gr_data) > 0 and len(self.log_gr_ratio_data) > 0 \
        and len(self.gr_ratio_data) > 0 and len(self.dos_data) > 0:
            self._compute_mean()
            self._compute_dos_moments()

class MBARBasinAnalysis(BasinAnalysis):
    def __init__(self, workspace=None, packings_dir='packings', jammed_packings_dir='jammed_packings', 
                 analysis_dir='analysis', volume_file="mbar_volume_data", pressure_file="pressure_data", 
                 zboo_file="glob_boo", volume_title = "VOLUME_MBAR"):
        super(MBARBasinAnalysis, self).__init__(workspace=workspace, packings_dir=packings_dir, 
                                                jammed_packings_dir=jammed_packings_dir, analysis_dir=analysis_dir, 
                                                volume_file=volume_file, pressure_file=pressure_file, 
                                                zboo_file=zboo_file, volume_title=volume_title)
        print self.volume_title
    def _collect_data_single_all(self, set_path):
        pd_list = []
        packing_dataset = MBARPackingDataSet(set_path)
        for fname in os.listdir(os.path.join(set_path, self.jammed_packings_dir)):
            if 'xyzd' in fname or 'xyd' in fname:
                dname = self._get_dname(fname)
                dname_packing = self._get_dname_packing(fname)
                base_directory_path = os.path.join(set_path, 'explore_bv_' + str(dname))
                if os.path.isdir(base_directory_path):
                    configpath = os.path.join(set_path, self.jammed_packings_dir, dname + '.config')
                    configpath_packing = os.path.join(set_path, self.packings_dir, dname_packing + ".config")
                    pd = MBARPackingData(str(dname), configpath, configpath_packing)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.volume_file)
                    pd.import_volume_data(path, title=self.volume_title)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.pressure_file)
                    pd.import_pressure_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.zboo_file)
                    pd.import_structural_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir)
                    pd.import_dos_data(path)                   
                    pd_list.append(pd)
        packing_dataset.add_data_all(pd_list)
        packing_dataset.compute_mean_and_moments()    
        return packing_dataset

class plot_mbar_data(object):
    def __init__(self, packing_datasets, figdir="figures"):
        if not os.path.isabs(figdir):
            figdir = os.path.join(os.getcwd(), figdir)
        trymakedir(figdir)
        self.packing_datasets = packing_datasets
    
    def __call__(self):
        if False:
            color_cycle = get_color_cycle()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i,dataset in enumerate(sorted(self.packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 1:
                    poly = dataset.hs_poly
                    structural_label = dataset.structural_label 
                    print poly
                    outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                    x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    #weights = 1./np.array(dataset.free_energies_err)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    weights = np.ones(len(dataset.free_energies))[np.array(outliers.non_outliers_indexes, dtype="i")]
                    print len(x), len(y)
                    x = np.log(x)
                    ax.scatter(x, y, label='{} {}'.format(structural_label, poly), color=color_cycle.next())
                    fit, cov = np.polyfit(x, y, 1, w=weights, cov=True)
                    fit_err = np.sqrt(np.diag(cov))
                    fit_fn = np.poly1d(fit)
                    #ax.plot(x, fit_fn(x), color='k')
                    dataset.add_extras((fit, fit_err))
                    print dataset.extras
            ax.legend(frameon=False, loc='center', prop={'size':18}, numpoints=1, scatterpoints=1, markerscale=1, columnspacing=0.25, labelspacing=0.25,
                      handletextpad=0, bbox_to_anchor=[0.08, 0.3])
            plt.ylabel(r"$F$")
            plt.xlabel(r"$\log \mathcal{P}$")
        if True:
            self.plot_all(plot_type="log_gr", average=True)
            self.plot_all(plot_type="log_gr_ratio", average=True)
            self.plot_all(plot_type="gr_ratio", average=True)
            self.plot_all(plot_type="dos", average=True)
        if True:
            self.plot_all(plot_type="log_gr", average=False)
            self.plot_all(plot_type="log_gr_ratio", average=False)
            self.plot_all(plot_type="gr_ratio", average=False)
            self.plot_all(plot_type="dos", average=False)
        if False:
            self.plot_correlations(plot_type="m0_q6")
        if True:
            color_cycle = get_color_cycle()
            fig = plt.figure()
            fig1 = plt.figure()
            fig2 = plt.figure()
            ax = fig.add_subplot(111)
            ax1 = fig1.add_subplot(111)
            ax2 = fig2.add_subplot(111)
            x, y, yerr = [], [], []
            y1, y1err = [], []
            for i,dataset in enumerate(sorted(self.packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 1:
                    boo = []
                    for bunch in dataset.boos:
                        boo.append([bunch.Q4, bunch.Q6, bunch.Q8, bunch.Q10, bunch.Q12])
                    boo12 = np.array(boo)[:,4]
                    y.append(np.mean(boo12))
                    yerr.append(np.std(boo12))
                    y1.append(np.mean(dataset.contacts))
                    y1err.append(np.std(dataset.contacts))
                    x.append(dataset.hs_poly)
            ax.errorbar(x, y, yerr=yerr, fmt='bo')        
            ax.legend(frameon=False, loc='center', prop={'size':18}, numpoints=1, scatterpoints=1, markerscale=1, columnspacing=0.25, labelspacing=0.25,
                      handletextpad=0, bbox_to_anchor=[0.08, 0.3])
            ax.set_xscale('log')
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$Q12$")
            
            ax1.errorbar(x, y1, yerr=y1err, fmt='bo')        
            ax1.legend(frameon=False, loc='center', prop={'size':18}, numpoints=1, scatterpoints=1, markerscale=1, columnspacing=0.25, labelspacing=0.25,
                      handletextpad=0, bbox_to_anchor=[0.08, 0.3])
            ax1.set_xscale('log')
            ax1.set_xlabel(r"$\eta$")
            ax1.set_ylabel(r"$\mathcal{P}$")
            
            ax2.errorbar(y1, y, yerr=yerr, xerr=y1err, fmt='bo')        
            ax2.legend(frameon=False, loc='center', prop={'size':18}, numpoints=1, scatterpoints=1, markerscale=1, columnspacing=0.25, labelspacing=0.25,
                      handletextpad=0, bbox_to_anchor=[0.08, 0.3])
            ax2.set_ylabel(r"$Q12$")
            ax2.set_xlabel(r"$\mathcal{P}$")
            
        
    def plot_all(self, plot_type="gr_ratio", figname=None, title=None, show=False, savefig=False, average=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        color_cycle = get_color_cycle()
        for i,dataset in enumerate(sorted(self.packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 0:
                    ax, xlabel, ylabel = self._plot_all(ax, dataset, plot_type=plot_type, 
                                                        average=average, label='{} {}'.format(dataset.structural_label, dataset.poly), 
                                                        color=color_cycle.next())
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        try:
            ax.legend(frameon=False, loc="best")
        except Exception, e:
            print e
            
        if title:
            plt.title(title)
        if figname is None:
            if average:
                figname = plot_type + "_average.eps"
            else:
                figname = plot_type + "_all.eps"
        if savefig:
            plt.savefig(figname)
        if show:
            plt.show()
        
    def _plot_all(self, ax, mbar_data, plot_type="log_gr", label=None, average=False, color='k', ndof=None):
        if plot_type == "log_gr":
            if average:
                log_gr = mbar_data.log_gr_mean
                ax = self._plot(ax, log_gr, label=label, plot_err=False, plot_fit=False, color=color)
            else:
                log_gr = mbar_data.log_gr_data
                for i,arr in enumerate(log_gr):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, color=color)
            xlabel=r'r'
            ylabel=r'$\log(g(r))$'
        if plot_type == "log_gr_ratio":
            if average:
                log_gr_ratio = mbar_data.log_gr_ratio_mean
                ax = self._plot(ax, log_gr_ratio, label=label, plot_err=True, plot_fit=False, color=color)
            else:
                log_gr_ratio = mbar_data.log_gr_ratio_data
                for i,arr in enumerate(log_gr_ratio):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, color=color)
            xlabel=r'r'
            ylabel=r'$\log(g(r)/r^{N-1})$'
        if plot_type == "gr_ratio":
            if average:
                gr_ratio = mbar_data.gr_ratio_mean
                ax = self._plot(ax, gr_ratio, label=label, plot_err=False, plot_fit=False, color=color)
            else:
                gr_ratio = mbar_data.gr_ratio_data
                for i,arr in enumerate(gr_ratio):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, color=color)
            xlabel=r'r'
            ylabel=r'$g(r)/r^{N-1}$'
            ax.set_xlim((0,1))
        if plot_type == "dos":
            if average:
                dos = mbar_data.dos_mean
                ax = self._plot(ax, dos, label=label, plot_err=False, plot_fit=False, normalize=True, color=color)
            else:
                dos = mbar_data.dos_data
                for i,arr in enumerate(dos):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, normalize=True, color=color)
            xlabel=r'r'
            ylabel=r'$g(r)$'
            ax.set_xlim((0.5,4))
            
        return ax, xlabel, ylabel
    
    def _plot(self, ax, arr, label=None, plot_err=False, plot_fit=False, normalize=False, color='k', marker='o'):
        (x, xerr, y, yerr, fit) = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        if normalize:
            area = simps(y, x)
            y = np.array(y) / area
        if plot_err:
            ax.errorbar(x, y, yerr=yerr, ms=6, label=label, marker=marker, color=color)
        else:
            ax.plot(x, y, label=label, linewidth=2.5, color=color)
        if plot_fit:
            ax.plot(x,fit, linestyle='--', linewidth=2, color=color)
        return ax
    
    def plot_correlations(self, plot_type="f_m1", figname=None, title=None, show=False, savefig=False, logx=False, logy=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        color_cycle = get_color_cycle()
        for i,dataset in enumerate(sorted(self.packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 0:
                    ax, xlabel, ylabel = self._plot_correlations(ax, dataset, plot_type=plot_type, 
                                                                  label=dataset.hs_poly, color=color_cycle.next())
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        try:
            ax.legend(frameon=False, loc="best", numpoints=1)
        except Exception, e:
            print e
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        if title:
            plt.title(title)
        if figname is None:
            figname = plot_type + ".eps"
        if savefig:
            plt.savefig(figname)
        if show:
            plt.show()
    
    def _plot_correlations(self, ax, mbar_data, plot_type="f_m1", label=None, color='b'):
        moments = np.array(mbar_data.dos_moments)
        free_energies = np.array(mbar_data.free_energies)
        #this gives the -log probability of being in a basin
        #for i,packing in enumerate(mbar_data.packing_data):
        #    free_energies[i] += packing.Facc 
        
        boo = []
        for bunch in mbar_data.boos:
            boo.append([bunch.Q4, bunch.Q6, bunch.Q8, bunch.Q10, bunch.Q12])
        boo = np.array(boo)
        z_numbers = mbar_data.contacts
        xlabel=r'$-\log(p)=f_i+f_{acc}$'
        
        #volume-moments correlations
        if plot_type == "f_m0":
            ax.scatter(free_energies, np.log(moments[:,0]), label=label, c=color, alpha=0.5)
            ylabel=r'$\max[g(r)]$'
        elif plot_type == "f_m1":
            ax.scatter(free_energies, np.log(moments[:,1]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(\langle r \rangle)$'
        elif plot_type == "f_m2":
            ax.scatter(free_energies, np.log(moments[:,2]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(\langle (r - \langle r \rangle)^2 \rangle)$'
        elif plot_type == "f_m3":
            ax.scatter(free_energies, np.log(moments[:,3]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(\frac{\langle (r - \langle r \rangle)^3 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{3/2}})$'
        elif plot_type == "f_m4":
            ax.scatter(free_energies, np.log(moments[:,4]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(\frac{\langle (r - \langle r \rangle)^4 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{2}})$'
        
        #moment-moment correlations
        #a = np.exp(-free_energies[:,0] + np.amax(free_energies[:,0]))
        #area = np.pi * (10 *  a / np.amax(a))**2 # 0 to 10 point radiuses
        area = np.ones(free_energies.shape[0])*2*np.pi*4
        if plot_type == "m1_m2":
            ax.scatter(moments[:,1], moments[:,2], s=area, label=label, c=color, alpha=0.5)
            xlabel=r'$\langle r \rangle$'
            ylabel=r'$\langle (r - \langle r \rangle)^2 \rangle$'
        elif plot_type == "m3_m4":
            ax.scatter(moments[:,3], moments[:,4], s=area, label=label, c=color, alpha=0.5)
            xlabel=r'$\frac{\langle (r - \langle r \rangle)^3 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{3/2}}$'
            ylabel=r'$\frac{\langle (r - \langle r \rangle)^4 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{2}}$'
        
        #boo-volume correlations
        if plot_type == "f_z":
            ax.scatter(free_energies, np.log(z_numbers), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Z)$'
        elif plot_type == "f_q4":
            ax.scatter(free_energies, np.log(boo[:,0]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q4)$'
        elif plot_type == "f_q6":
            ax.scatter(free_energies, np.log(boo[:,1]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q6)$'
        elif plot_type == "f_q8":
            ax.scatter(free_energies, np.log(boo[:,2]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q8)$'
        elif plot_type == "f_q10":
            ax.scatter(free_energies, np.log(boo[:,3]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q10)$'
        elif plot_type == "f_q12":
            ax.scatter(free_energies, np.log(boo[:,4]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q12)$'
        
        #boo-moments correlations
        if plot_type == "m0_q6":
            ax.scatter(moments[:,0], boo[:,1], s=area, label=label, c=color, alpha=0.5)
            ylabel=r'$Q6$'
            xlabel=r'$\max[g(r)]$'
        elif plot_type == "m1_q6":
            ax.scatter(moments[:,1], boo[:,1], s=area, label=label, c=color, alpha=0.5)
            ylabel=r'$Q6$'
            xlabel=r'$\langle r \rangle$'
        elif plot_type == "m2_q6":
            ax.scatter(moments[:,2], boo[:,1], s=area, label=label, c=color, alpha=0.5)
            ylabel=r'$Q6$'
            xlabel=r'$\langle (r - \langle r \rangle)^2 \rangle$'
        elif plot_type == "m3_q6":
            ax.scatter(moments[:,3], boo[:,1], s=area, label=label, c=color, alpha=0.5)
            ylabel=r'$Q6$'
            xlabel=r'$\frac{\langle (r - \langle r \rangle)^3 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{3/2}}$'
        elif plot_type == "m4_q6":
            ax.scatter(moments[:,4], boo[:,1], s=area, label=label, c=color, alpha=0.5)
            ylabel=r'$Q6$'
            xlabel=r'$\frac{\langle (r - \langle r \rangle)^4 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{2}}$'
        
        #NOTE: here add boo-pressure correlations
        #NOTE: here add pressure-volume correlations
        
        return ax, xlabel, ylabel
    
if __name__ == "__main__":
    show = True
    
    pts = MBARBasinAnalysis()
    pts.collect_data_every_set_all(dir_signature='n*phi*phi*3D*')
    pmd = plot_mbar_data(pts.packing_datasets)
    pmd()
    if show:
        plt.show()
    plt.close()