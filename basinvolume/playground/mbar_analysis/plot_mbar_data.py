from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import re
    import matplotlib
    #matplotlib.use('Agg')
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
plt.rcParams.update({'font.size': 28})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['figure.figsize'] = 10, 7.7
##########################################################
####SET COLOUR MAP######                                                               
def get_color_cycle():
    cm = plt.get_cmap('Set2')
    color_cycle=cycle([cm(1. * i / 13) for i in xrange(13)])
    return color_cycle
def get_marker_cycle():
    markers = ["o","v","s","x","^","8","p","<","*","D",">",]
    markercycle = cycle(markers)
    return markercycle
def get_line_cycle():
    lines = ["-","--","-."]
    linecycle = cycle(lines)
    return linecycle
###########################################################
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
        self.step_timeseries_mean_path = None
        self.step_timeseries_mean_path_std = None 
        self.step_timeseries_mean_eucdist = None
        self.step_timeseries_mean_eucdist_std = None
        self.step_timeseries_stepsize = None
        
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
    
    def import_steps_time_series_diffusion(self, path, niter=int(1e6), eqtime=int(5e5)):
        timeseries = []
        series_order = []
        step_timeseries_mean_path, step_timeseries_mean_path_std = [], [] 
        step_timeseries_mean_eucdist, step_timeseries_mean_eucdist_std = [], []
        if os.path.isdir(path):
            #import stepsize
            configf = ConfigParser.ConfigParser()
            diffusion_configfile = os.path.join(path, "diffusion_{}.{}.config".format(self.jammed_packing_name, niter))
            assert(os.path.isfile(diffusion_configfile))
            configf.read(diffusion_configfile)
            self.step_timeseries_stepsize = configf.getfloat('KMIN_MCRUNNER_STATUS', 'stepsize')
            #import timeseries
            file_list = glob.glob(os.path.join(path,'StepsTimeSeries.{}.*'.format(niter)))
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
                #the other step_timeseries are recorded every nth entry and tells us what distance we have covered since
                #the last nth step 
                mean_arr = []
                nsubs = step_timeseries[0][eqtime:].size // n
                for j in xrange(nsubs):
                    mean_arr.append(np.sum(step_timeseries[0][eqtime+j*n:eqtime+(j+1)*n]))
                mean, stdev = np.mean(np.array(mean_arr))*self.step_timeseries_stepsize, np.std(np.array(mean_arr))
                step_timeseries_mean_path.append(mean)
                step_timeseries_mean_path_std.append(stdev/np.sqrt(len(mean_arr)))
                step_timeseries_mean_eucdist.append(np.mean(step_timeseries[i+1][eqtime//n:]))
                step_timeseries_mean_eucdist_std.append(np.std(step_timeseries[i+1][eqtime//n:])/np.sqrt(len(step_timeseries[i+1])))
            self.step_timeseries_mean_path = copy.deepcopy(step_timeseries_mean_path)
            self.step_timeseries_mean_path_std = copy.deepcopy(step_timeseries_mean_path_std)
            self.step_timeseries_mean_eucdist = copy.deepcopy(step_timeseries_mean_eucdist)
            self.step_timeseries_mean_eucdist_std = copy.deepcopy(step_timeseries_mean_eucdist_std)
#            x, dx, y, dy = step_timeseries_mean_path, step_timeseries_mean_path_std, step_timeseries_mean_eucdist, step_timeseries_mean_eucdist_std
#            x, dx, y, dy = np.array(x), np.array(dx), np.array(y), np.array(dy)
#            fig = plt.figure()
#            ax = fig.add_subplot(111)
#            ax.errorbar(np.log(x), np.log(y), fmt='o', xerr=dx/x, yerr=dy/y)
#            plt.show()
        
class PolyPackingDataSet(PackingDataSet):
    def __init__(self, set_path):
        super(PolyPackingDataSet, self).__init__(set_path)
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
        elif 'fluid' in self.set_name:
            self.structural_label = 'fluid'
        else:
            self.structural_label = None
            print "cannot recognise structural label (fcc or disordered), set to None"

class TINTBasinAnalysis(BasinAnalysis):
    def __init__(self, workspace=None, packings_dir='packings', jammed_packings_dir='jammed_packings', 
                 analysis_dir='analysis', volume_file="volume_data", pressure_file="pressure_data", 
                 zboo_file="glob_boo", volume_title = "VOLUME_FULL_PT"):
        super(TINTBasinAnalysis, self).__init__(workspace=workspace, packings_dir=packings_dir, 
                                                jammed_packings_dir=jammed_packings_dir, analysis_dir=analysis_dir, 
                                                volume_file=volume_file, pressure_file=pressure_file, 
                                                zboo_file=zboo_file, volume_title=volume_title)
        print self.volume_title
    def _collect_data_single_all(self, set_path):
        pd_list = []
        packing_dataset = PolyPackingDataSet(set_path)
        for fname in os.listdir(os.path.join(set_path, self.jammed_packings_dir)):
            if 'xyzd' in fname or 'xyd' in fname:
                dname = self._get_dname(fname)
                dname_packing = self._get_dname_packing(fname)
                base_directory_path = os.path.join(set_path, 'explore_bv_' + str(dname))
                if os.path.isdir(base_directory_path):
                    configpath = os.path.join(set_path, self.jammed_packings_dir, dname + '.config')
                    configpath_packing = os.path.join(set_path, self.packings_dir, dname_packing + ".config")
                    pd = PackingData(str(dname), configpath, configpath_packing)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.volume_file)
                    pd.import_volume_data(path, title=self.volume_title)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.pressure_file)
                    pd.import_pressure_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.zboo_file)
                    pd.import_structural_data(path)                   
                    pd_list.append(pd)
        packing_dataset.add_data_all(pd_list)
        return packing_dataset


class MBARPackingDataSet(PolyPackingDataSet):
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
        self.step_timeseries_mean_path_data = []
        self.step_timeseries_mean_path_std_data = [] 
        self.step_timeseries_mean_eucdist_data = []
        self.step_timeseries_mean_eucdist_std_data = []
        self.step_timeseries_stepsize_data = []
        
    def add_data_all(self, packing_data):
        """
        packing data is a list of PackingData objects
        """
        self.packing_data.extend(packing_data)
        for data in packing_data:
            #the reason why they must all be true is because we are interested in the relation among these variables
            if (data.F is not None and data.P is not None and data.Z is not None and data.boo is not None and
                data.log_gr is not None and data.log_gr_ratio is not None and data.gr_ratio is not None and
                data.dos is not None and len(data.step_timeseries_mean_path) > 0 
                and len(data.step_timeseries_mean_eucdist) > 0):
                self.free_energies.append(data.F)
                self.free_energies_err.append(data.Ferr)
                self.pressures.append(data.P)
                self.contacts.append(data.Z)
                self.boos.append(data.boo)
                self.log_gr_data.append(data.log_gr)
                self.log_gr_ratio_data.append(data.log_gr_ratio) 
                self.gr_ratio_data.append(data.gr_ratio)
                self.dos_data.append(data.dos)
                self.step_timeseries_mean_path_data.append(data.step_timeseries_mean_path)
                self.step_timeseries_mean_path_std_data.append(data.step_timeseries_mean_path_std)
                self.step_timeseries_mean_eucdist_data.append(data.step_timeseries_mean_eucdist)
                self.step_timeseries_mean_eucdist_std_data.append(data.step_timeseries_mean_eucdist_std)
                self.step_timeseries_stepsize_data.append(data.step_timeseries_stepsize)
                
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
            xref = np.linspace(xmin, xmax, 100)
            all = []
            for arr in data:
                x, y = arr[:,0], arr[:,2]
                #assert x.size == xref.size, 'x array size mismatches'
                #tcky = splev(x, splrep(x, y, s=0), der=0)
                f = interp1d(x, y, bounds_error=True)
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
                 zboo_file="glob_boo", volume_title = "VOLUME_MBAR", diffusion_dir="diffusion"):
        super(MBARBasinAnalysis, self).__init__(workspace=workspace, packings_dir=packings_dir, 
                                                jammed_packings_dir=jammed_packings_dir, analysis_dir=analysis_dir, 
                                                volume_file=volume_file, pressure_file=pressure_file, 
                                                zboo_file=zboo_file, volume_title=volume_title)
        self.diffusion_dir = diffusion_dir
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
                    path = os.path.join(base_directory_path, self.diffusion_dir)
                    pd.import_steps_time_series_diffusion(path)                   
                    pd_list.append(pd)
        packing_dataset.add_data_all(pd_list)
        packing_dataset.compute_mean_and_moments()    
        return packing_dataset

class plot_mbar_data(object):
    def __init__(self, mbar_packing_datasets, tint_packing_datasets, figdir="figures", show=False):
        if not os.path.isabs(figdir):
            figdir = os.path.join(os.getcwd(), figdir)
        trymakedir(figdir)
        self.figdir = figdir
        self.mbar_packing_datasets = mbar_packing_datasets
        self.tint_packing_datasets = tint_packing_datasets
        self.show = show
    
    def __call__(self):
        if True:
            color_cycle = get_color_cycle()
            marker_cycle = get_marker_cycle()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i,dataset in enumerate(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 1:
                    poly = dataset.hs_poly
                    structural_label = dataset.structural_label 
                    print structural_label, poly
                    outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                    x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    print len(x), len(y)
                    x = np.log(x)
                    ax.scatter(x, y, label='{} {}'.format(structural_label, poly), color=color_cycle.next(), 
                               marker=marker_cycle.next(), s=70)
                    #fit, cov = np.polyfit(x, y, 1, w=weights, cov=True)
                    #fit_err = np.sqrt(np.diag(cov))
                    #fit_fn = np.poly1d(fit)
                    #ax.plot(x, fit_fn(x), color='k')
                    #dataset.add_extras((fit, fit_err))
                    #print dataset.extras
            ax.legend(frameon=False, loc='upper left', prop={'size':20}, numpoints=1, scatterpoints=1, 
                      markerscale=1, columnspacing=0.5, labelspacing=0.5)
            plt.ylabel(r"$F$")
            plt.xlabel(r"$\log \mathcal{P}$")
            fig.savefig(os.path.join(self.figdir, 'f_logp.pdf'))
        if True:
            #self.plot_all(plot_type="log_gr", average=True)
            self.plot_all(plot_type="log_gr_ratio", average=True, savefig=True, show=self.show)
            #self.plot_all(plot_type="gr_ratio", average=True)
            self.plot_all(plot_type="dos", average=True, savefig=True, show=self.show)
        if True:
            self.plot_all(plot_type="log_gr", average=False)
            self.plot_all(plot_type="log_gr_ratio", average=False, savefig=True, show=self.show)
            self.plot_all(plot_type="gr_ratio", average=False)
            self.plot_all(plot_type="dos", average=False, savefig=True, show=self.show)
        if False:
            self.plot_correlations(plot_type="m0_q6")
        if True:
            #plot Q12 vs poly and pressure vs poly
            color_cycle = get_color_cycle()
            marker_cycle = get_marker_cycle()
            fig = plt.figure()
            fig1 = plt.figure()
            fig2 = plt.figure()
            ax = fig.add_subplot(111)
            ax1 = fig1.add_subplot(111)
            ax2 = fig2.add_subplot(111)
            x, y, yerr = [], [], []
            y1, y1err = [], []
            for i,dataset in enumerate(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 1:
                    boo = []
                    for bunch in dataset.boos:
                        boo.append([bunch.Q4, bunch.Q6, bunch.Q8, bunch.Q10, bunch.Q12])
                    boo12 = np.array(boo)[:,4]
                    y = [np.mean(boo12)]
                    yerr = [np.std(boo12)]
                    y1 = [np.mean(dataset.pressures)]
                    y1err = [np.std(dataset.pressures)]
                    x = [dataset.hs_poly]
                    
                    color=color_cycle.next()
                    marker = marker_cycle.next()
                    ax.errorbar(x, y, yerr=yerr, marker=marker, ms=9, 
                                label='{} {}'.format(dataset.structural_label, x[0]), color=color)        
                    ax1.errorbar(x, y1, yerr=y1err, marker=marker, ms=9, 
                                 label='{} {}'.format(dataset.structural_label, x[0]), color=color)
                    ax2.errorbar(y1, y, yerr=yerr, xerr=y1err, marker=marker, ms=9, 
                                 label='{} {}'.format(dataset.structural_label, x[0]), color=color)
            
            ax.legend(frameon=False, loc='best', prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                      columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            ax.set_xscale('log')
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$Q12$")
            fig.savefig(os.path.join(self.figdir, 'poly_q12.pdf'))

            ax1.legend(frameon=False, loc='best', prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                      columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            ax1.set_xscale('log')
            ax1.set_xlabel(r"$\eta$")
            ax1.set_ylabel(r"$\mathcal{P}$")
            fig1.savefig(os.path.join(self.figdir, 'poly_p.pdf'))        
            
            ax2.legend(frameon=False, loc='best', prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                      columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            ax2.set_ylabel(r"$Q12$")
            ax2.set_xlabel(r"$\mathcal{P}$")
            fig2.savefig(os.path.join(self.figdir, 'q12_p.pdf'))
        
        if True:
            #plot cdf of maximum r value visited by pt simulation
            color_cycle = get_color_cycle()
            line_cycle = get_line_cycle()
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            for i,dataset in enumerate(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 1:
                    x = []
                    for arr in dataset.log_gr_ratio_data: 
                        x.append(arr[-1,0])
                    cdf = CDFAccumulator()
                    cdf.add_array(x)
                    x, cdf_x = cdf.get_vecdata()
                    f = interp1d(x, cdf_x, bounds_error=True)
                    xref = np.linspace(x[0],x[-1],1000)
                    ax3.plot(xref, f(xref), label='{} {:.3E}'.format(dataset.structural_label, dataset.hs_poly),
                            linewidth=3, color=color_cycle.next(), linestyle=line_cycle.next())
            ax3.set_xlabel(r'$r_{max}$')
            ax3.set_ylabel(r'$\mathrm{cdf}[\mathrm{max}_r(h(r)/r^{N-1})]$')
            ax3.legend(frameon=False, loc='best', prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                       columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            fig3.savefig(os.path.join(self.figdir, 'maxr_cdf.pdf'))
            
        if True:
            #plot correlation between volume and volume of core region
            color_cycle = get_color_cycle()
            marker_cycle = get_marker_cycle()
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)
            fig42 = plt.figure()
            ax42 = fig42.add_subplot(111)
            for i,dataset in enumerate(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 0:
                    log_core_vol = []
                    log_tot_vol = []
                    for packing in dataset.packing_data: 
                        arr = packing.log_gr_ratio
                        if arr is not None:
                            (x, xerr, y, yerr, fit) = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
                            j = next(idx for idx, value in enumerate(y) if value < -0.5) #-0.5 was chosen arbitrarily
                            log_core_vol.append(log_volume_nball(x[j], (packing.nparticles-1)*packing.bdim))
                            log_tot_vol.append(packing.F)
                        else:
                            print packing.configpath_packing
                    ax4.scatter(-np.array(log_core_vol), np.array(log_tot_vol), color=color_cycle.next(), 
                                marker=marker_cycle.next(), s=70,
                                label='{} {:.3E}'.format(dataset.structural_label, dataset.hs_poly))
                    ax42.scatter(np.log(dataset.pressures), -np.array(log_core_vol), color=color_cycle.next(), 
                                marker=marker_cycle.next(), s=70,
                                label='{} {:.3E}'.format(dataset.structural_label, dataset.hs_poly))
            ax4.set_xlabel(r'$-\log(V_{c})$')
            ax4.set_ylabel(r'$-\log(V_{t})$')
            ax4.legend(frameon=False, loc=2, prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                       columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            fig4.savefig(os.path.join(self.figdir, 'core_tot_vol_correlations.pdf'))
            ax42.set_ylabel(r'$-\log(V_{c})$')
            ax42.set_xlabel(r'$\log\mathcal{P}$')
            ax42.legend(frameon=False, loc=2, prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                       columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            fig42.savefig(os.path.join(self.figdir, 'core_vol_p_correlations.pdf'))
        if False:
            #plot comparison between tint and mbar
            color_cycle = get_color_cycle()
            marker_cycle = get_marker_cycle()
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(111)
            for mbar_dataset, tint_dataset in zip(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly), 
                                                  sorted(self.tint_packing_datasets, key=lambda data: data.hs_poly)):
                if len(mbar_dataset.free_energies) > 0 and len(tint_dataset.free_energies) > 0:
                    assert mbar_dataset.structural_label == tint_dataset.structural_label
                    assert mbar_dataset.hs_poly == tint_dataset.hs_poly 
                    assert len(mbar_dataset.free_energies) == len(tint_dataset.free_energies)
                    ax5.errorbar(mbar_dataset.free_energies, tint_dataset.free_energies, 
                                 xerr=mbar_dataset.free_energies_err, yerr=tint_dataset.free_energies_err, 
                                 color=color_cycle.next(), linestyle="None", marker=marker_cycle.next(), markersize=15,
                                 label='{} {:.3E}'.format(mbar_dataset.structural_label, mbar_dataset.hs_poly))
            ax5.plot(np.linspace(75,110,10),np.linspace(75,110,10),color='k',linestyle='-')
            ax5.set_xlabel(r'$-\log(V_{mbar})$')
            ax5.set_ylabel(r'$-\log(V_{tint})$')
            ax5.legend(frameon=False, loc=2, prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                       columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            fig5.savefig(os.path.join(self.figdir, 'mbar_tint_comparison.pdf'))
        
        if True:
            #plot diffusion curves 
            color_cycle = get_color_cycle()
            marker_cycle = get_marker_cycle()
            line_cycle = get_line_cycle()
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(111)
            fig7 = plt.figure()
            ax7 = fig7.add_subplot(111)
            for i,dataset in enumerate(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 0:
                    X, DX = dataset.step_timeseries_mean_path_data, dataset.step_timeseries_mean_path_std_data
                    Y, DY = dataset.step_timeseries_mean_eucdist_data, dataset.step_timeseries_mean_eucdist_std_data
                    label = '{} {:.3E}'.format(dataset.structural_label, dataset.hs_poly)
                    marker = marker_cycle.next()
                    color = color_cycle.next()
                    ls = line_cycle.next()
                    for i, (x, dx, y, dy) in enumerate(zip(X, DX, Y, DY)):
                        if i > 0:
                            label = None
                        x, dx, y, dy = np.array(x), np.array(dx), np.array(y), np.array(dy)
                        #print "logx: ", np.log(x), "logy: ", np.log(y)
                        pol = np.poly1d(np.polyfit(np.log(x)[:3], np.log(y)[:3], 1, w=(y/dy)[:3])) #[5:-1]
                        w = np.polyfit(np.log(x)[:3], np.log(y)[:3], 1)
                        #print w
                        #if w[1] > -5: #this removes a couple of outliers DEBUG, check this
                        fit = pol(np.log(x))
                        csv_tuple = np.array([np.log(x), dx/x, np.log(y), dy/y, fit]).transpose() 
                        ax6 = self._plot(ax6, csv_tuple, label=label, 
                                        plot_err=True, plot_fit=True, color=color, 
                                        marker=marker, ls=ls)
                        redy = np.log(y)-0.5*np.log(x)
                        csv_tuple = np.array([np.log(x), dx/x+dy/y, redy, dy/y, np.zeros(len(x))]).transpose() #redy - np.amax(redy)
                        ax7 = self._plot(ax7, csv_tuple, label=label, plot_err=True, plot_fit=False,
                                         color=color, marker=marker, ls=ls)
            ax6.set_ylabel(r'$\log(\Delta r)$')
            ax6.set_xlabel('$\log (\Delta s)$')
            ax6.legend(frameon=False, loc=2, prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                       columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            fig6.savefig(os.path.join(self.figdir, 'diffusion_logs_logr.pdf'))
            ax7.set_ylabel(r'$\log(\Delta r) - \frac{1}{2}\log(\Delta s)$')
            ax7.set_xlabel(r'$\log (\Delta s)$')
            ax7.legend(frameon=False, loc=2, prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                       columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
            fig7.savefig(os.path.join(self.figdir, 'diffusion_logs_logr_red.pdf'))
            
            
    def plot_all(self, plot_type="gr_ratio", figname=None, title=None, show=False, savefig=False, average=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        color_cycle = get_color_cycle()
        line_cycle = get_line_cycle()
        for i,dataset in enumerate(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly)):
                print '{} {:.3E}'.format(dataset.structural_label, dataset.hs_poly)
                print len(dataset.free_energies)
                if len(dataset.free_energies) > 1:
                    ax, xlabel, ylabel = self._plot_all(ax, dataset, plot_type=plot_type, 
                                                        average=average, 
                                                        label='{} {:.3E}'.format(dataset.structural_label, dataset.hs_poly), 
                                                        color=color_cycle.next(), ls=line_cycle.next())
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        try:
            ax.legend(frameon=False, loc='best', prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                      columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
        except Exception, e:
            print e    
        if title:
            ax.set_title(title)
        if figname is None:
            if average:
                figname = plot_type + "_average.pdf"
            else:
                figname = plot_type + "_all.pdf"
        if savefig:
            plt.savefig(os.path.join(self.figdir, figname))
        if show:
            plt.show()
        
    def _plot_all(self, ax, mbar_data, plot_type="log_gr", label=None, average=False, color='k', ls='-', ndof=None):
        if plot_type == "log_gr":
            if average:
                log_gr = mbar_data.log_gr_mean
                ax = self._plot(ax, log_gr, label=label, plot_err=False, plot_fit=False, color=color, ls=ls)
            else:
                log_gr = mbar_data.log_gr_data
                for i,arr in enumerate(log_gr):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, color=color, ls=ls)
            xlabel=r'$r$'
            ylabel=r'$\log(h(r))$'
        if plot_type == "log_gr_ratio":
            if average:
                log_gr_ratio = mbar_data.log_gr_ratio_mean
                ax = self._plot(ax, log_gr_ratio, label=label, plot_err=True, plot_fit=False, color=color, marker='', ls=ls)
            else:
                log_gr_ratio = mbar_data.log_gr_ratio_data
                for i,arr in enumerate(log_gr_ratio):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, color=color, ls=ls)
            xlabel=r'$r$'
            ylabel=r'$\log(h(r)/r^{N-1})$'
            if average:
                ax.set_ylim((-60,2))
                ax.set_xlim((0,2))
        if plot_type == "gr_ratio":
            if average:
                gr_ratio = mbar_data.gr_ratio_mean
                ax = self._plot(ax, gr_ratio, label=label, plot_err=False, plot_fit=False, color=color, ls=ls)
            else:
                gr_ratio = mbar_data.gr_ratio_data
                for i,arr in enumerate(gr_ratio):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, color=color, ls=ls)
            xlabel=r'$r$'
            ylabel=r'$h(r)/r^{N-1}$'
            ax.set_xlim((0,1))
        if plot_type == "dos":
            if average:
                dos = mbar_data.dos_mean
                ax = self._plot(ax, dos, label=label, plot_err=False, plot_fit=False, normalize=True, color=color, ls=ls)
            else:
                dos = mbar_data.dos_data
                for i,arr in enumerate(dos):
                    if i > 0:
                        label = None
                    ax = self._plot(ax, arr, label=label, plot_err=False, plot_fit=False, normalize=True, color=color, ls=ls)
            xlabel=r'$r$'
            ylabel=r'$h(r)$'
            ax.set_xlim((0.5,4))
            #ax.set_ylim((1e-27,3))
            #ax.set_yscale('log')
        return ax, xlabel, ylabel
    
    def _plot(self, ax, arr, label=None, plot_err=False, plot_fit=False, normalize=False, color='k', marker='o', ls='-'):
        (x, xerr, y, yerr, fit) = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        if normalize:
            area = simps(y, x)
            y = np.array(y) / area
        if plot_err:
            ax.errorbar(x, y, yerr=yerr, markersize=9, label=label, marker=marker, color=color, linestyle=ls, linewidth=3)
        else:
            ax.plot(x, y, label=label, linewidth=3, color=color, linestyle=ls)
        if plot_fit:
            ax.plot(x, fit, linestyle='-', linewidth=1, color='k', marker='')
        return ax
    
    def plot_correlations(self, plot_type="f_m1", figname=None, title=None, show=False, savefig=False, logx=False, logy=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        color_cycle = get_color_cycle()
        for i,dataset in enumerate(sorted(self.mbar_packing_datasets, key=lambda data: data.hs_poly)):
                if len(dataset.free_energies) > 0:
                    ax, xlabel, ylabel = self._plot_correlations(ax, dataset, plot_type=plot_type, 
                                                                  label=dataset.hs_poly, color=color_cycle.next())
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        try:
            ax.legend(frameon=False, loc='best', prop={'size':20}, numpoints=1, scatterpoints=1, markerscale=1, 
                       columnspacing=0.5, labelspacing=0.5, handletextpad=0.25)
        except Exception, e:
            print e
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        if title:
            plt.title(title)
        if figname is None:
            figname = plot_type + ".pdf"
        if savefig:
            plt.savefig(os.path.join(self.figdir, figname))
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
            ylabel=r'$\max[h(r)]$'
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
            xlabel=r'$\max[h(r)]$'
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
    
    pts_mbar = MBARBasinAnalysis()
    pts_tint = TINTBasinAnalysis()
    pts_mbar.collect_data_every_set_all(data_name="mbar_basin_analysis.pickle", dir_signature='n*phi*phi*3D*')
    pts_tint.collect_data_every_set_all(data_name="tint_basin_analysis.pickle", dir_signature='n*phi*phi*3D*')
    pmd = plot_mbar_data(pts_mbar.packing_datasets, pts_tint.packing_datasets)
    pmd()
    if show:
        plt.show()
    plt.close()
