from __future__ import division
try:
    import numpy as np
    from scipy import integrate
    from basinvolume.utils import log_factorial, ResultsFile, save_pdf, gen_gauss, log_gen_gauss, get_gauss_times_expx, MLCost
    from pele.optimize import LBFGS_CPP
    from scipy.special import gamma
    import matplotlib.pyplot as plt
except ImportError as err:
    print err

class MLMethodGenGauss(object):
    """
    Maximum likelihood estimate of parmeters in generalised gauss
    """
    def __init__(self, F0):
        self.F0 = F0
    def find_get_opt_pars(self):
        initial_mu = np.mean(self.F0)
        initial_alpha = np.sqrt(2) * np.std(self.F0)
        initial_zeta = 2
        self.x = np.array([initial_mu, initial_alpha, initial_zeta])
        print "xinitial", self.x
        self.pot = MLCost(self.F0, log_probf=log_gen_gauss)
        optimizer = LBFGS_CPP(self.x, self.pot, nsteps=1e5, tol=1e-4)
        result = optimizer.run()
        if result.success:
            self.opt_x = result.coords
            print "xfinal", self.opt_x
            self.error_opt_x = self.pot.get_error_estimate(self.opt_x)
            print "xerror", self.error_opt_x
            return self.opt_x
        else:
            raise Exception("MLMethodGenGauss: optimization did not converge")

class LogOmegaBase(object):
    """
    Common functionality to all (non-Jackknife) methods estimating generalised
    gaussian parameters and un-biasing by numerical integration.
    Basically, everything after self.mu, self.alpha, self.zeta and erorrs on
    that are known. 
    """
    def __init__(self, F0, volume_sanity_check):
        self.F0 = F0
        self.volume_sanity_check = volume_sanity_check
        self.mu = None
        self.alpha = None
        self.zeta = None
    def compute_log_omega(self):
        if self.mu == None or self.alpha == None or self.zeta == None:
            raise Exception("LogOmegaBase: generalised gaussian parameters are not determined")
        integral, error_integral = integrate.quad(get_gauss_times_expx, self.volume_sanity_check.F0_acc, np.amax(self.F0) * 100, args = ([self.mu, self.alpha, self.zeta], ), points = [np.amin(self.F0), self.mu, np.amax(self.F0)])
        self.S_star = - self.volume_sanity_check.F0_acc + np.log(integral)
        self.S = self.S_star - log_factorial(self.volume_sanity_check.nr_particles)
    def write_to_file(self, file_name, title):
        f = ResultsFile(file_name)
        f.set_heading(title)
        f.to_file("S_star", self.S_star)
        f.to_file("S", self.S)
        f.to_file("mu", self.mu)
        f.to_file("alpha", self.alpha)
        f.to_file("zeta", self.zeta)
        f.close()
        print title
        print "S_star", self.S_star
        print "S", self.S
        print "Generalised Gaussian parameters:"
        print "mu", self.mu
        print "alpha", self.alpha
        print "zeta", self.zeta
        plot_name = file_name + "_plot.pdf"
        plt.hist(self.F0, bins=42, normed=True, label="Data")
        xr = np.linspace(np.amin(self.F0), np.amax(self.F0), 500)
        pars = [self.mu, self.alpha, self.zeta]
        plt.plot(xr, gen_gauss(xr, pars))
        save_pdf(plt, plot_name)

class MLLogOmega(LogOmegaBase):
    """
    Compute LogOmega by ML fit of generalised gaussian to volumes and
    un-biasing.
    
    General ML references:
    http://sites.stat.psu.edu/~sesa/stat504/Lecture/lec3_4up.pdf
    http://www.maths.manchester.ac.uk/~peterf/CSI_ch4_part1.pdf
    """
    def __init__(self, F0, volume_sanity_check):
        super(MLLogOmega, self).__init__(F0, volume_sanity_check)
    def compute_and_write_entropy(self, file_name):
        self.get_generalised_gaussian_parameters()
        self.compute_log_omega()
        self.write_to_file(file_name, "LOG_OMEGA_ML")
    def get_generalised_gaussian_parameters(self):
        ml_method_gen_gauss = MLMethodGenGauss(self.F0)
        self.mu, self.alpha, self.zeta = ml_method_gen_gauss.find_get_opt_pars()
        