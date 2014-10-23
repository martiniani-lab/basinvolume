from __future__ import division
try:
    import numpy as np
    from basinvolume.utils import ResultsFile, log_factorial, save_pdf, CrossValidationCost
    from scipy import integrate
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity
    from pele.optimize import LBFGS
except ImportError as err:
    print err
    
class KernelDensityLogOmega(object):
    """
    Use kernel density estimate to get callable description of PDF.
    
    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_density_estimation
    http://scikit-learn.org/stable/_downloads/plot_kde_1d.py
    http://scikit-learn.org/stable/modules/density.html
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
    """
    def __init__(self, F0_full, volume_sanity_check, kernel="gaussian", bandwidth=None):
        self.F0 = np.asarray(F0_full)
        self.volume_sanity_check = volume_sanity_check
        self.out_file_heading = "LOG_OMEGA_KERNEL_DENSITY"
        self.possible_kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        if kernel not in self.possible_kernels:
            raise Exception("KernelDensityLogOmega: illegal kernel choice")
        self.kernel = kernel
        if bandwidth == None:
            self.bandwidth = self.get_bandwidth_estimate(method="cross_validation")
            #self.bandwidth = self.get_bandwidth_estimate(method="Silverman")
        else:
            self.bandwidth = bandwidth
        if self.bandwidth <= 0:
            raise Exception("KernelDensityLogOmega: illegal bandwidth choice")
    def compute_and_write_entropy(self, file_name):
        self.compute_log_omega()
        self.write_to_file(file_name)
    def compute_log_omega(self):
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(self.F0[:, np.newaxis])
        n_integrate = 2**18 + 1
        x_integrate = np.linspace(self.volume_sanity_check.F0_acc, np.amax(self.F0) * 100, n_integrate)
        log_pdf = self.kde.score_samples(x_integrate[:, np.newaxis])
        integrand_control = np.exp(log_pdf)
        integral_control = integrate.romb(integrand_control, dx=x_integrate[1]-x_integrate[0])
        if(np.abs(integral_control - 1) > 1e-10):
            raise Exception("KernelDensityLogOmega: compute_log_omega: possible integration failure")
        integrand = np.exp(np.add(log_pdf, x_integrate))
        integral = integrate.romb(integrand, dx=x_integrate[1]-x_integrate[0])
        self.S_star = - self.volume_sanity_check.F0_acc + np.log(integral)
        self.S = self.S_star - log_factorial(self.volume_sanity_check.nr_particles)
    def write_to_file(self, file_name):
        f = ResultsFile(file_name)
        f.set_heading(self.out_file_heading)
        f.to_file("S_star", self.S_star)
        f.to_file("S", self.S)
        f.close()
        print self.out_file_heading
        print "S_star", self.S_star
        print "S", self.S
        plot_name = file_name + "_plot.pdf"
        plt.hist(self.F0, bins=42, normed=True, label="Data")
        self.x_plot_1d = np.linspace(np.amin(self.F0), np.amax(self.F0), 500)
        self.pdf_x_1d = np.exp(self.kde.score_samples(self.x_plot_1d[:, np.newaxis]))
        plt.plot(self.x_plot_1d, self.pdf_x_1d, label="PDF estimate")
        plt.legend()
        save_pdf(plt, plot_name)
    def get_bandwidth_estimate(self, method="cross_validation"):
        """
        Use some rule to get bandwidth estimate from data.
        
        References
        ----------
        http://en.wikipedia.org/wiki/Kernel_density_estimation
        http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode15.html
        http://www.control.aau.dk/~tk/undervisning/PhDAdvSI/Litterature/MadsenAndHolst2006.pdf
        """
        opt_bandwidth = None
        if method == "Silverman":
            nr_samples = len(self.F0)
            std_samples = np.std(self.F0)
            opt_bandwidth = ((4 * std_samples ** 5) / (3 * nr_samples)) ** (1/5)
        elif method == "cross_validation":
            silv_initial = ((4 * np.std(self.F0) ** 5) / (3 * len(self.F0))) ** (1/5)
            print "silv_initial", silv_initial
            loocv = CrossValidationBandwidthSelection(self.F0, kernel=self.kernel, h_initial=silv_initial)
            opt_bandwidth = loocv.opt_bandwidth 
        else:
            raise Exception("KernelDensityLogOmega: get_bandwidth_estimate: illegal method input")
        assert(opt_bandwidth != None)
        print method, "method used to estimate bandwidth"
        print "estimated optimal bandwidth", opt_bandwidth
        return opt_bandwidth
    
class CrossValidationBandwidthSelection(object):
    """
    Use leave-one-out cross validation to estimate bandwidth.
    
    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_density_estimation
    http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode15.html
    http://www.control.aau.dk/~tk/undervisning/PhDAdvSI/Litterature/MadsenAndHolst2006.pdf
    """
    def __init__(self, data, kernel="gaussian", h_initial=2):
        pot = CrossValidationCost(data, kernel=kernel)
        optimizer = LBFGS(np.asarray([h_initial]), pot, maxstep=1)
        print "run bandwidth optimization"
        result = optimizer.run()
        print "done"
        self.opt_bandwidth = result.coords

class KernelDensityLogOmegaJackKnife(object):
    """
    Use kernel density estimate to get callable description of PDF, error bar from jack knife.
    """
    def __init__(self, F0_full, volume_sanity_check, kernel="gaussian", bandwidth=None):
        self.F0 = np.asarray(F0_full)
        self.volume_sanity_check = volume_sanity_check
        self.out_file_heading = "LOG_OMEGA_KERNEL_DENSITY"
        self.possible_kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        if kernel not in self.possible_kernels:
            raise Exception("KernelDensityLogOmega: illegal kernel choice")
        self.kernel = kernel
        if bandwidth == None:
            bandwidth_estimator = KernelDensityLogOmega(F0_full, volume_sanity_check, kernel, bandwidth=None)
            self.bandwidth = bandwidth_estimator.bandwidth
        else:
            self.bandwidth = bandwidth
        if self.bandwidth <= 0:
            raise Exception("KernelDensityLogOmega: illegal bandwidth choice")