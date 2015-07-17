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
    from scipy.interpolate import spline
    from scipy.integrate import simps
    import glob
    from basinvolume.post_processing import PackingData, PackingDataSet, BasinAnalysis
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
                self.pressures.append(data.P)
                self.contacts.append(data.Z)
                self.boos.append(data.boo)
                self.log_gr_data.append(data.log_gr)
                self.log_gr_ratio_data.append(data.log_gr_ratio) 
                self.gr_ratio_data.append(data.gr_ratio)
                self.dos_data.append(data.dos)

    def compute_mean(self):
        def get_mean(data):
            xref = data[0][:,0]
            all = []
            for arr in data:
                x, y = arr[:,0], arr[:,2]
                assert x.size == xref.size, 'x array size mismatches'
                assert np.allclose(x, xref), 'mismatching x arrays'
                all.append(y)
            mu = np.mean(all, axis=0)
            std = np.std(all, axis=0)
            return np.transpose(np.array([x, data[0][:,1], mu, std, data[0][:,3]]))
        
        self.log_gr_mean = get_mean(self.log_gr_data)
        self.log_gr_ratio_mean = get_mean(self.log_gr_ratio_data)
        self.gr_ratio_mean = get_mean(self.gr_ratio_data)  
        self.dos_mean = get_mean(self.dos_data)
    
    def compute_dos_moments(self):
        def normalize_dist(self, y, x):
            area = simps(y, x)
            y = np.array(y) / area
            area = simps(y, x)
            y = np.array(y) / area
            return y
        for arr in self.dos:
            x, y = arr[:,0], arr[:,2]
            y = normalize_dist(y, x)
            mode =  x[np.argmax(y)]
            mean = np.average(x, weights=y)
            var = np.average((x-mean)**2, weights=y)
            std = np.sqrt(var)
            skewness = np.average((x-mean)**3, weights=y) / std**3
            kurtosis = np.average((x-mean)**4, weights=y) / var**2
            self.dos_moments.append((mode, mean, var, skewness, kurtosis))

class MBARBasinAnalysis(BasinAnalysis):
    def __init__(self, workspace=None, packings_dir='packings', jammed_packings_dir='jammed_packings', 
                 analysis_dir='analysis', volume_file="mbar_volume", pressure_file="pressure_data", 
                 zboo_file="glob_boo", volume_title = "MBAR_VOLUME"):
        super(MBARBasinAnalysis, self).__init__(workspace=workspace, packings_dir=packings_dir, 
                                                jammed_packings_dir=jammed_packings_dir, analysis_dir=analysis_dir, 
                                                volume_file=volume_file, pressure_file=pressure_file, 
                                                zboo_file=zboo_file, volume_title=volume_title)
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
                    pd.import_volume_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.pressure_file)
                    pd.import_pressure_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.zboo_file)
                    pd.import_structural_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir)
                    pd.import_dos_data()                   
                    pd_list.append(pd)
        packing_dataset.add_data_all(pd_list)
        packing_dataset.compute_mean()
        packing_dataset.compute_moments()
        return packing_dataset

class plot_mbar_data(object):
    def __init__(self, packing_datasets, figdir="figures"):
        from scipy.optimize import curve_fit
        if not os.path.isabs(figdir):
            figdir = os.path.join(os.getcwd(), figdir)
        trymakedir(figdir)
    
    def _plot(self, ax, csv_tuple, label=None, plot_err=False, plot_fit=False, normalize=False):
        (x, xerr, y, yerr, fit) = csv_tuple
        if normalize:
            area = simps(y, x)
            y = np.array(y) / area
        if plot_err:
            ax.errorbar(x, y, yerr=yerr, fmt='bo', ms=9, label=label)
        else:
            ax.plot(x, y, label=label, linewidth=2.5)
        if plot_fit:
            ax.plot(x,fit,'b--', linewidth=2)
        return ax
    
    def _plot_all(self, ax, mbar_data, plot_type="log_gr", label=None, average=False):
        if plot_type == "log_gr":
            if average:
                log_gr = mbar_data.log_gr_mean
            else:
                log_gr = mbar_data.log_gr
            for csv_tuple in log_gr:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False)
            xlabel=r'r'
            ylabel=r'$\log(g(r))$'
        if plot_type == "log_gr_ratio":
            if average:
                log_gr_ratio = mbar_data.log_gr_ratio_mean
            else:
                log_gr_ratio = mbar_data.log_gr_ratio
            for csv_tuple in log_gr_ratio:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False)
            xlabel=r'r'
            ylabel=r'$\log(g(r)/r^{N-1})$'
        if plot_type == "gr_ratio":
            if average:
                gr_ratio = mbar_data.gr_ratio_mean
            else:
                gr_ratio = mbar_data.gr_ratio
            for csv_tuple in gr_ratio:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False)
            xlabel=r'r'
            ylabel=r'$g(r)/r^{N-1}$'
            ax.set_xlim((0,1))
        if plot_type == "dos":
            if average:
                dos = mbar_data.dos_mean
            else:
                dos = mbar_data.dos
            for csv_tuple in dos:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False, normalize=True)
            xlabel=r'r'
            ylabel=r'$g(r)$'
            ax.set_xlim((0.5,4))
        
        return ax, xlabel, ylabel
    
    def plot_all(self, plot_type="log_gr", figname=None, title=None, show=False, savefig=False, average=False):
        dlabel = None
        if average:
            dlabel = "disordered"
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax, xlabel, ylabel = self._plot_all(ax, self.fcc_data, plot_type=plot_type, average=average, label=r"fcc $s=0.05$")
        ax, xlabel, ylabel = self._plot_all(ax, self.fcc_mono_data, plot_type=plot_type, average=average, label=r"fcc mono")
        ax, xlabel, ylabel = self._plot_all(ax, self.disordered_data, plot_type=plot_type, average=average, label=dlabel)
        
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
    
    def _plot_correlations(self, ax, mbar_data, plot_type="f_m1", label=None, color='b'):
        moments = np.reshape(mbar_data.dos_moments, (-1,5))
        free_energies = np.reshape(mbar_data.free_energies, (-1,2))
        free_energies[:,0] += np.array(mbar_data.acc_free_energies) #this gives the -log probability of being in a basin
        
        #free_energies[:,0] = np.exp(-free_energies[:,0])
        boo = np.reshape(mbar_data.boo, (-1,5))
        z_numbers = mbar_data.z_numbers
        xlabel=r'$-\log(p)=f_i+f_{acc}$'
        
        #volume-moments correlations
        if plot_type == "f_m0":
            ax.scatter(free_energies[:,0], np.log(moments[:,0]), label=label, c=color, alpha=0.5)
            ylabel=r'$\max[g(r)]$'
        elif plot_type == "f_m1":
            ax.scatter(free_energies[:,0], np.log(moments[:,1]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(\langle r \rangle)$'
        elif plot_type == "f_m2":
            ax.scatter(free_energies[:,0], np.log(moments[:,2]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(\langle (r - \langle r \rangle)^2 \rangle)$'
        elif plot_type == "f_m3":
            ax.scatter(free_energies[:,0], np.log(moments[:,3]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(\frac{\langle (r - \langle r \rangle)^3 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{3/2}})$'
        elif plot_type == "f_m4":
            ax.scatter(free_energies[:,0], np.log(moments[:,4]), label=label, c=color, alpha=0.5)
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
            ax.scatter(free_energies[:,0], np.log(z_numbers), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Z)$'
        elif plot_type == "f_q4":
            ax.scatter(free_energies[:,0], np.log(boo[:,0]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q4)$'
        elif plot_type == "f_q6":
            ax.scatter(free_energies[:,0], np.log(boo[:,1]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q6)$'
        elif plot_type == "f_q8":
            ax.scatter(free_energies[:,0], np.log(boo[:,2]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q8)$'
        elif plot_type == "f_q10":
            ax.scatter(free_energies[:,0], np.log(boo[:,3]), label=label, c=color, alpha=0.5)
            ylabel=r'$log(Q10)$'
        elif plot_type == "f_q12":
            ax.scatter(free_energies[:,0], np.log(boo[:,4]), label=label, c=color, alpha=0.5)
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
    
    def plot_correlations(self, plot_type="f_m1", figname=None, title=None, show=False, savefig=False, logx=False, logy=False):
        dlabel = None
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax, xlabel, ylabel = self._plot_correlations(ax, self.fcc_data, plot_type=plot_type, label=r"fcc", color='g')
        ax, xlabel, ylabel = self._plot_correlations(ax, self.fcc_mono_data, plot_type=plot_type, label=r"fcc mono", color='r')
        ax, xlabel, ylabel = self._plot_correlations(ax, self.disordered_data, plot_type=plot_type, label=dlabel)
        
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
    
if __name__ == "__main__":
    show = True
    
    pts = BasinAnalysis()
    pts.collect_data_every_set_all(dir_signature='n*phi*phi*fcc*')
    #print pts.free_energies
    #plot_mbar_data(pts.packing_datasets)
    if show:
        plt.show()
    plt.close()