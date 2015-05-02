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
    from basinvolume.utils import log_factorial, read_csv_xy, read_txt, write_csv_xy
    import scipy
    from scipy.stats import t
    from scipy.interpolate import spline
    from scipy.integrate import simps
    import glob
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
####SET COLOUR MAP######                                                               
cm = plt.get_cmap('Dark2')
########################
#####################LINE STYLE CYCLER####################                             
lines = ["-","--","-."]
linecycler = cycle(lines)
color_cycle=cycle([cm(1. * i / 6) for i in xrange(6)])
##########################################################
"""
for plotting a linear fit with intervals of confidence see http://nbviewer.ipython.org/url/bagrow.com/dsv/LEC10_notes_2014-02-13.ipynb
"""
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

class mbar_data(object):
    def __init__(self, label, analysis_folder = "analysis", log_gr_file = "log_gr.csv", 
                 log_gr_ratio_file = "log_gr_ratio.csv", gr_ratio_file = "gr_ratio.csv",
                 dos_file = "dos.csv", volume_file="mbar_dos_volume_data"):
        self.label = label
        self.analysis_folder = analysis_folder
        self.volume_file = volume_file
        self.log_gr_file = log_gr_file
        self.log_gr_ratio_file = log_gr_ratio_file
        self.gr_ratio_file = gr_ratio_file
        self.dos_file = dos_file
        self.volumes = []
        self.log_gr = []
        self.log_gr_ratio = []
        self.gr_ratio = []
        self.dos = []
        self.dos_mean = []
        self.log_gr_mean = []
        self.log_gr_ratio_mean = []
        self.gr_ratio_mean = []
        self.dos_moments = []
        
    def compute_mean(self):
        self.log_gr_mean = np.mean(self.log_gr_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.log_gr[0]
        self.log_gr_mean = [(x, xerr, self.log_gr_mean, yerr, fit)]
        
        self.log_gr_ratio_mean = np.mean(self.log_gr_ratio_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.log_gr_ratio[0]
        self.log_gr_ratio_mean = [(x, xerr, self.log_gr_ratio_mean, yerr, fit)]
        
        self.gr_ratio_mean = np.mean(self.gr_ratio_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.gr_ratio[0]
        self.gr_ratio_mean = [(x, xerr, self.gr_ratio_mean, yerr, fit)]
        
        self.dos_mean = np.mean(self.dos_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.dos[0]
        self.dos_mean = [(x, xerr, self.dos_mean, yerr, fit)]
    
    def _normalize_dist(self, y, x):
        area = simps(y, x)
        y = np.array(y) / area
        area = simps(y, x)
        y = np.array(y) / area
        return y
    
    def compute_dos_moments(self):
        for csv_tuple in self.dos:
            (x, xerr, y, yerr, fit) = csv_tuple
            y = self._normalize_dist(y, x)
            mean = np.average(x, weights=y)
            var = np.average((x-mean)**2, weights=y)
            std = np.sqrt(var)
            skewness = np.average((x-mean)**3, weights=y) / std**3
            kurtosis = np.average((x-mean)**4, weights=y) / var**2
            self.dos_moments.append((mean, var, skewness, kurtosis))

class plot_mbar_data(object):
    def __init__(self, workdir=None, explore_dir='explore_bv_jammed_packing', Nrange=(0,1000)):
        if not workdir:
            workdir = os.getcwd() 
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir
        self.explore_dir = explore_dir
        self.Nrange = Nrange
        self.fcc_data = mbar_data("fcc")
        self.fcc_mono_data = mbar_data("fcc_mono")
        self.disordered_data = mbar_data("disordered")
        markers = ["bo", "r^", "gs", "kx", "c+"]
        self.markercycler = cycle(markers)
        self._collect_data(self.fcc_data)
        self._collect_data(self.fcc_mono_data)
        self._collect_data(self.disordered_data)
        self.fcc_mono_data.compute_dos_moments()
        self.fcc_data.compute_dos_moments()
        self.disordered_data.compute_dos_moments()
    
    def _collect_data(self, mbar_data):
        subdirs = get_immediate_subdirectories(os.path.join(self.workdir, mbar_data.label))
        for folder in subdirs:
            if self.explore_dir in folder:
                path = os.path.join(self.workdir, mbar_data.label, folder, mbar_data.analysis_folder)
                if os.path.isdir(path):
                    npack = int(re.findall('\d+', folder)[0])
                    if self.Nrange[0] <= npack <= self.Nrange[1]:
                        #collect volume
                        configf = ConfigParser.ConfigParser()
                        fpath = os.path.join(path, mbar_data.volume_file)
                        if os.path.isfile(fpath):
                            configf.read(fpath)
                            F, Ferr = configf.getfloat('VOLUME_DOS','F0'), configf.getfloat('VOLUME_DOS','sigF0')
                            mbar_data.volumes.append((F, Ferr))
                        #log_gr
                        fpath = os.path.join(path, mbar_data.log_gr_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            mbar_data.log_gr.append((x,xerr,y,yerr,fit))
                            mbar_data.log_gr_mean.append(y)
                        #log_gr_ratio
                        fpath = os.path.join(path, mbar_data.log_gr_ratio_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            y -= np.amax(y)
                            mbar_data.log_gr_ratio.append((x,xerr,y,yerr,fit))
                            mbar_data.log_gr_ratio_mean.append(y)
                        #gr_ratio
                        fpath = os.path.join(path, mbar_data.gr_ratio_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            mbar_data.gr_ratio.append((x,xerr,y,yerr,fit))
                            mbar_data.gr_ratio_mean.append(y)
                        #dos
                        fpath = os.path.join(path, mbar_data.dos_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            mbar_data.dos.append((x,xerr,y,yerr,fit))
                            mbar_data.dos_mean.append(y)
        mbar_data.compute_mean()
    
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
        
        ax, xlabel, ylabel = self._plot_all(ax, self.fcc_data, plot_type=plot_type, average=average, label=r"fcc")
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
    
    def _plot_correlations(self, ax, mbar_data, plot_type="v_m1", label=None, color='b'):
        moments = np.reshape(mbar_data.dos_moments, (-1,4))
        volumes = np.reshape(mbar_data.volumes, (-1,2))
        xlabel=r'F'
        if plot_type == "v_m1":
            ax.scatter(volumes[:,0], moments[:,0], label=label, c=color, alpha=0.5)
            ylabel=r'$\langle r \rangle$'
        elif plot_type == "v_m2":
            ax.scatter(volumes[:,0], moments[:,1], label=label, c=color, alpha=0.5)
            ylabel=r'$\langle (r - \langle r \rangle)^2 \rangle$'
        elif plot_type == "v_m3":
            ax.scatter(volumes[:,0], moments[:,2], label=label, c=color, alpha=0.5)
            ylabel=r'$\frac{\langle (r - \langle r \rangle)^3 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{3/2}}$'
        elif plot_type == "v_m4":
            ax.scatter(volumes[:,0], moments[:,3], label=label, c=color, alpha=0.5)
            ylabel=r'$\frac{\langle (r - \langle r \rangle)^4 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{2}}$'
        
        area = np.pi * (10 / (volumes[:,0]/100))**2 # 0 to 10 point radiuses
        if plot_type == "m1_m2":
            ax.scatter(moments[:,0], moments[:,1], s=area, label=label, c=color, alpha=0.5)
            xlabel=r'$\langle r \rangle$'
            ylabel=r'$\langle (r - \langle r \rangle)^2 \rangle$'
        elif plot_type == "m3_m4":
            ax.scatter(moments[:,2], moments[:,3], s=area, label=label, c=color, alpha=0.5)
            xlabel=r'$\frac{\langle (r - \langle r \rangle)^3 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{3/2}}$'
            ylabel=r'$\frac{\langle (r - \langle r \rangle)^4 \rangle}{\langle (r - \langle r \rangle)^2 \rangle^{2}}$'
        
        return ax, xlabel, ylabel
    
    def plot_correlations(self, plot_type="v_m1", figname=None, title=None, show=False, savefig=False):
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
            
        if title:
            plt.title(title)
        if figname is None:
            figname = plot_type + ".eps"
        if savefig:
            plt.savefig(figname)
        if show:
            plt.show()

class diffusion_data(object):
    def __init__(self, label, analysis_folder = "diffusion"):
        self.label = label
        self.analysis_folder = analysis_folder
        self.step_timeseries_mean_path = []
        self.step_timeseries_mean_path_std = [] 
        self.step_timeseries_mean_eucdist = []
        self.step_timeseries_mean_eucdist_std = []

class plot_diffusion_data(object):
    def __init__(self, workdir=None, explore_dir='explore_bv_jammed_packing', Nrange=(0,1000)):
        if not workdir:
            workdir = os.getcwd() 
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir
        self.explore_dir = explore_dir
        self.Nrange = Nrange
        self.fcc_data = diffusion_data("fcc")
        self.fcc_mono_data = diffusion_data("fcc_mono")
        self.disordered_data = diffusion_data("disordered")
        markers = ["bo", "r^", "gs", "kx", "c+"]
        self.markercycler = cycle(markers)
        self._collect_diffusion_data(self.fcc_data)
        self._collect_diffusion_data(self.fcc_mono_data)
        self._collect_diffusion_data(self.disordered_data)
    
    def _collect_diffusion_data(self, diffusion_data):
        subdirs = get_immediate_subdirectories(os.path.join(self.workdir, diffusion_data.label))
        for folder in subdirs:
            if self.explore_dir in folder:
                self._import_steps_time_series_diffusion(folder, diffusion_data)
                
    def _import_steps_time_series_diffusion(self, folder, diffusion_data, eqtime=2.5e4):
        timeseries = []
        series_order = []
        path = os.path.join(self.workdir, diffusion_data.label, folder, diffusion_data.analysis_folder)
        if os.path.isdir(path):
            npack = int(re.findall('\d+', folder)[0])
            if self.Nrange[0] <= npack <= self.Nrange[1]:
                file_list = glob.glob(os.path.join(path,'StepsTimeSeries*'))
                file_list = sorted(file_list, key = lambda x: int(x.split(".")[1]))
                for series_path in file_list:
                    fname = str(os.path.split(series_path)[-1].split())
                    digits = map(int, re.findall(r'\d+', fname))
                    series_order.append(digits[-1])
                    timeseries.append(read_txt(series_path))
                X = np.array(timeseries)
                Y = series_order
                step_timeseries = np.array([x for (y, x) in sorted(zip(Y, X))])
                step_timeseries_order =  np.sort(series_order)
                step_timeseries_mean_path = []
                step_timeseries_mean_path_std = []
                step_timeseries_mean_eucdist = []
                step_timeseries_mean_eucdist_std = []
                for i,n in enumerate(step_timeseries_order[1:]):
                    #step_timeseries[0] is the timeseries recorded at every step, we use this to compute the path length
                    #the other step_timeseries are recrded every nth entry and tells us what distance we have covered since
                    #the last nth step 
                    mean_arr = []
                    nsubs = step_timeseries[0][eqtime:].size // n
                    for j in xrange(nsubs):
                        mean_arr.append(np.sum(step_timeseries[0][eqtime+j*n:eqtime+(j+1)*n]))
                    mean, stdev = np.mean(np.array(mean_arr)), np.std(np.array(mean_arr))
                    step_timeseries_mean_path.append(mean)
                    step_timeseries_mean_path_std.append(stdev/np.sqrt(len(mean_arr)))
                    step_timeseries_mean_eucdist.append(np.mean(step_timeseries[i+1][eqtime//n:]))
                    step_timeseries_mean_eucdist_std.append(np.std(step_timeseries[i+1][eqtime//n:])/np.sqrt(len(step_timeseries[i+1])))
                diffusion_data.step_timeseries_mean_path.append(step_timeseries_mean_path)
                diffusion_data.step_timeseries_mean_path_std.append(step_timeseries_mean_path_std)
                diffusion_data.step_timeseries_mean_eucdist.append(step_timeseries_mean_eucdist)
                diffusion_data.step_timeseries_mean_eucdist_std.append(step_timeseries_mean_eucdist_std)
    
    def _plot(self, ax, csv_tuple, label=None, plot_err=False, plot_fit=False, normalize=False):
        (x, xerr, y, yerr, fit) = csv_tuple
        color = color_cycle.next()
        if normalize:
            area = simps(y, x)
            y = np.array(y) / area
        if plot_err:
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, c=color, fmt='o', ms=6, label=label)
        else:
            ax.plot(x, y, label=label, linewidth=2.5)
        if plot_fit:
            ax.plot(x,fit,'--', c=color, linewidth=2)
        return ax
    
    def _plot_all(self, ax, diffusion_data, plot_type="logr_vs_logt", label=None):
        X, DX = diffusion_data.step_timeseries_mean_path, diffusion_data.step_timeseries_mean_path_std
        Y, DY = diffusion_data.step_timeseries_mean_eucdist, diffusion_data.step_timeseries_mean_eucdist_std
        for x, dx, y, dy in zip(X, DX, Y, DY):
            x, dx, y, dy = np.array(x), np.array(dx), np.array(y), np.array(dy)
            
            if plot_type == "logr_vs_logt":
                pol = np.poly1d(np.polyfit(np.log(x)[:3], np.log(y)[:3], 1,  w=(y/dy)[:3])) #[5:-1]
                w = np.polyfit(np.log(x)[:3], np.log(y)[:3], 1)
                print w
                fit = pol(np.log(x))
                csv_tuple = (np.log(x), dx/x, np.log(y), dy/y, fit)
                ax = self._plot(ax, csv_tuple, label=label, plot_err=True, plot_fit=True)
                ylabel = r'$\log(\Delta r)$'
                xlabel = r'$\log (\Delta s)$'
            if plot_type == "red_logr_vs_logt":
                csv_tuple = (np.log(x), dx/x+dy/y, np.log(y)-0.5*np.log(x), dy/y, np.zeros(len(x)))
                ax = self._plot(ax, csv_tuple, label=label, plot_err=True, plot_fit=False)
                ylabel = (r'$\log(\Delta r) - \frac{1}{2}\log(\Delta s)$')
                xlabel = (r'$\log (\Delta s)$')
        return ax, xlabel, ylabel
                
                
    
    def plot_all(self, plot_type="logr_vs_logt", figname=None, title=None, show=False, savefig=False):
        dlabel = None
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax, xlabel, ylabel = self._plot_all(ax, self.fcc_data, plot_type=plot_type, label=r"fcc")
        ax, xlabel, ylabel = self._plot_all(ax, self.fcc_mono_data, plot_type=plot_type, label=r"fcc mono")
        ax, xlabel, ylabel = self._plot_all(ax, self.disordered_data, plot_type=plot_type, label=dlabel)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        try:
            ax.legend(frameon=False, loc="best")
        except Exception, e:
            print e
            
        if title:
            plt.title(title)
        if figname is None:
            figname = plot_type + "_all.eps"
        if savefig:
            plt.savefig(figname)
        if show:
            plt.show()
     
if __name__ == "__main__":
    show = False
    pe = plot_mbar_data(Nrange=(0,1000))
#    pe.plot_all(plot_type="log_gr_ratio", show=show, savefig=True)
#    pe.plot_all(plot_type="log_gr", show=show, savefig=True)
#    pe.plot_all(plot_type="gr_ratio", show=show, savefig=True)
#    pe.plot_all(plot_type="dos", show=show, savefig=True)
#    pe.plot_all(plot_type="log_gr_ratio", show=show, savefig=True, average=True)
#    pe.plot_all(plot_type="log_gr", show=show, savefig=True, average=True)
#    pe.plot_all(plot_type="gr_ratio", show=show, savefig=True, average=True)
#    pe.plot_all(plot_type="dos", show=show, savefig=True, average=True)

#    pe.plot_correlations(plot_type="v_m1", show=False, savefig=True)
#    pe.plot_correlations(plot_type="v_m2", show=False, savefig=True)
#    pe.plot_correlations(plot_type="v_m3", show=False, savefig=True)
#    pe.plot_correlations(plot_type="v_m4", show=False, savefig=True)
#    pe.plot_correlations(plot_type="m1_m2", show=False, savefig=True)
#    pe.plot_correlations(plot_type="m3_m4", show=False, savefig=True)

    diff = plot_diffusion_data(Nrange=(0,1))
    diff.plot_all("logr_vs_logt", show=show, savefig=True)
    diff.plot_all("red_logr_vs_logt", show=show, savefig=True)
    plt.show()