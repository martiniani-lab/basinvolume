from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import gamma, gammaln
    from basinvolume.utils import to_string, save_pdf, log_factorial
    from basinvolume.utils import ResultsFile, MomentsAcc, CDFAccumulator
    from scipy import integrate
except ImportError as err:
    print err

from scipy.special import gammainc, gammaincc
    
class GeneralisedLogNormal(object):
    """
    Implements the generalised log normal distribution, see e.g.: http://link.springer.com/article/10.1007%2Fs00180-011-0233-9#page-1
    Parameters are as follows:
    PDF(mu, alpha, zeta; x) = zeta / (2**((zeta+1)/zeta) * alpha * gamma(1 / zeta) * x) * exp[-0.5*|log(x) - mu / alpha|**zeta]
    """
    def __init__(self, mu_initial = 1, alpha_initial = 1, zeta_initial = 1, alpha_min = 1e-10, zeta_min = 1e-10, verbose = False):
        if alpha_min < 0:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: alpha_min")
        if zeta_min < 0:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: zeta_min")
        self.alpha_min = alpha_min
        self.zeta_min = zeta_min
        self.mu = mu_initial
        self.alpha_offset = alpha_initial - self.alpha_min
        self.zeta_offset = zeta_initial - self.zeta_min
        self.verbose = verbose
    
    def set_mu(self, mu):
        self.mu = mu
    
    def set_alpha(self, alpha):
        if alpha < self.alpha_min:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: alpha")
        self.alpha_offset = alpha - self.alpha_min
    
    def set_zeta(self, zeta):
        if zeta < self.zeta_min:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: zeta")
        self.zeta_offset = zeta - self.zeta_min
    
    def get_alpha(self, alpha_offset):
        return np.abs(alpha_offset) + self.alpha_min
    
    def get_zeta(self, zeta_offset):
        return np.abs(zeta_offset) + self.zeta_min
    
    def get(self, x, mu, alpha_offset, zeta_offset):
        """
        Return PDF
        """
        z = self.get_zeta(zeta_offset)
        a = self.get_alpha(alpha_offset)
        logx = np.log(x)
        return z / (2**((z+1.0)/z) * a * gamma(1.0 / z)) * np.exp(-0.5*np.power(np.abs((logx - mu) / a),z) - logx)
    
    def get_cdf(self, x, mu, alpha_offset, zeta_offset):
        """
        Return CDF with the convention that for x = 0 CDF = 1
        """
        def _get_cdf(logx, mu, a, z):
            if logx <= mu:
                cdf = 0.5*gammaincc(1.0/z, 0.5*np.power((mu - logx)/a,z))    
            else:
                cdf = 1.0 - 0.5*gammaincc(1.0/z, 0.5*np.power((logx - mu)/a,z))        
            return 1.0 - cdf
        v_get_cdf = np.vectorize(_get_cdf)
        a = self.get_alpha(alpha_offset)
        z = self.get_zeta(zeta_offset)
        logx = np.log(x)
        return v_get_cdf(logx, mu, a, z)
    
    def get_times_xpow(self, x, mu, alpha_offset, zeta_offset, kappa, n):
        z = self.get_zeta(zeta_offset)
        a = self.get_alpha(alpha_offset)
        logx = np.log(x)
        return z / (2**((z+1.0)/z) * a * gamma(1.0 / z)) * np.exp(-0.5*np.power(np.abs((logx - mu) / a),z) + logx * (n/kappa - 1))
    
    def get_log_times_xpow(self, x, mu, alpha_offset, zeta_offset, kappa, n):
        """
        returns the log of the probability
        """
        z = self.get_zeta(zeta_offset)
        a = self.get_alpha(alpha_offset)
        logx = np.log(x)
        return np.log(z) - ((z+1.0)/z)*np.log(2) - np.log(a) - gammaln(1.0 / z) - 0.5*np.power(np.abs((logx - mu) / a),z) + logx * (n/kappa - 1)
    
    def get_times_xpow_with_pars(self, x, mu, alpha, zeta, kappa, n):
        return self.get_times_xpow(x, mu, alpha - self.alpha_min, zeta - self.zeta_min, kappa, n)
    
    def get_fitted(self, x):
        return self.get(x, self.mu, self.alpha_offset, self.zeta_offset)
    
    def get_fitted_times_xpow(self, x, kappa, n):
        return self.get_times_xpow(x, self.mu, self.alpha_offset, self.zeta_offset, kappa, n)
    
    def get_log_fitted_times_xpow(self, x, kappa, n):
        """
        returns the log of the probability with the fit parameters
        """
        return self.get_log_times_xpow(x, self.mu, self.alpha_offset, self.zeta_offset, kappa, n)
    
    def fit(self, data_x, data_y):
        mean = np.mean(np.log(x))
        var = np.var(np.log(x))
        opt_gen, error_gen = curve_fit(self.get, data_x, data_y, [mean, var, 2])
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print "self.mu", self.mu
            print "self.alpha_offset", self.alpha_offset
            print "self.zeta_offset", self.zeta_offset
    
    def fit_cdf(self, x, cdf_x, initial_zeta=1.5):
        mean = np.mean(np.log(x))
        var = np.var(np.log(x))
        initial_mu = mean
        initial_zeta = initial_zeta
        initial_alpha = var
        print "initial guess [mu, alpha, zeta]:", [initial_mu, initial_alpha, initial_zeta]
        opt_gen, error_gen = curve_fit(self.get_cdf, x, cdf_x, [initial_mu, initial_alpha, initial_zeta])
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print "self.mu", self.mu
            print "self.alpha_offset", self.alpha_offset
            print "self.zeta_offset", self.zeta_offset
            
class GeneralisedGauss(object):
    """
    Implements the generalised gaussian distribution, see e.g.: http://en.wikipedia.org/wiki/Generalized_normal_distribution
    Parameters are as follows:
    PDF(mu, alpha, zeta; x) = zeta / (2 * alpha * gamma(1 / zeta)) * exp[-|x - mu|**zeta / alpha**zeta]
    """
    def __init__(self, mu_initial = 1, alpha_initial = 1, zeta_initial = 1, alpha_min = 1e-10, zeta_min = 1e-10, verbose = False):
        if alpha_min < 0:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: alpha_min")
        if zeta_min < 0:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: zeta_min")
        self.alpha_min = alpha_min
        self.zeta_min = zeta_min
        self.mu = mu_initial
        self.alpha_offset = alpha_initial - self.alpha_min
        self.zeta_offset = zeta_initial - self.zeta_min
        self.verbose = verbose
    def set_mu(self, mu):
        self.mu = mu
    def set_alpha(self, alpha):
        if alpha < self.alpha_min:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: alpha")
        self.alpha_offset = alpha - self.alpha_min
    def set_zeta(self, zeta):
        if zeta < self.zeta_min:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: zeta")
        self.zeta_offset = zeta - self.zeta_min
    def get_alpha(self, alpha_offset):
        return np.abs(alpha_offset) + self.alpha_min
    def get_zeta(self, zeta_offset):
        return np.abs(zeta_offset) + self.zeta_min
    def get(self, x, mu, alpha_offset, zeta_offset):
        """
        Return PDF
        """
        return self.get_zeta(zeta_offset) / (2 * self.get_alpha(alpha_offset) * gamma(1 / self.get_zeta(zeta_offset))) * np.exp(- np.power((np.abs(x - mu) / self.get_alpha(alpha_offset)), self.get_zeta(zeta_offset)))
    def get_cdf(self, x, mu, alpha_offset, zeta_offset):
        """
        Return CDF with the convention that for x = - \infty, CDF = 1
        """
        alpha = self.get_alpha(alpha_offset)
        zeta = self.get_zeta(zeta_offset)
        return 0.5 - np.sign(x - mu) * 0.5 * gammainc(1 / zeta, (np.abs(x - mu) / alpha) ** zeta) 
    def get_times_expx(self, x, mu, alpha_offset, zeta_offset):
        return self.get_zeta(zeta_offset) / (2 * self.get_alpha(alpha_offset) * gamma(1 / self.get_zeta(zeta_offset))) * np.exp(- np.power((np.abs(x - mu) / self.get_alpha(alpha_offset)), self.get_zeta(zeta_offset)) + x)
    def get_times_expx_with_pars(self, x, mu, alpha, zeta):
        return self.get_times_expx(x, mu, alpha - self.alpha_min, zeta - self.zeta_min)
    def get_fitted(self, x):
        return self.get(x, self.mu, self.alpha_offset, self.zeta_offset)
    def get_fitted_times_expx(self, x):
        return self.get_times_expx(x, self.mu, self.alpha_offset, self.zeta_offset)
    def fit(self, data_x, data_y):
        opt_gen, error_gen = curve_fit(self.get, data_x, data_y, [np.mean(data_x), 2 * np.var(data_x), 2])
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print "self.mu", self.mu
            print "self.alpha_offset", self.alpha_offset
            print "self.zeta_offset", self.zeta_offset
    def fit_cdf(self, x, cdf_x):
        initial_mu = np.mean(x)
        initial_zeta = 1.5
        initial_alpha = np.sqrt(gamma(1 / initial_zeta) / gamma(3 / initial_zeta) * np.var(x))
        opt_gen, error_gen = curve_fit(self.get_cdf, x, cdf_x, [initial_mu, initial_alpha, initial_zeta])
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print "self.mu", self.mu
            print "self.alpha_offset", self.alpha_offset
            print "self.zeta_offset", self.zeta_offset

class JackLogOmega(object):
    """
    Compute Jackknife mean and error for the LogOmega entropy.
    Used to get an error bar on LogOmega.
    """
    def __init__(self, F0, alpha_min, zeta_min, bins, volume_sanity_check):
        self.F0 = F0
        self.alpha_min = alpha_min
        self.zeta_min = zeta_min
        self.bins = bins
        self.volume_sanity_check = volume_sanity_check
        self.compute_jack_estimates()
    def compute_jack_estimates(self):
        self.jack_acc = MomentsAcc()
        self.jack_acc_mu = MomentsAcc()
        self.jack_acc_alpha = MomentsAcc()
        self.jack_acc_zeta = MomentsAcc()
        for idx in xrange(len(self.F0)):
            S_star_red, mu_red, alpha_red, zeta_red = self.get_S_star_excluding_index(idx)
            self.jack_acc.update(S_star_red)
            self.jack_acc_mu.update(mu_red)
            self.jack_acc_alpha.update(alpha_red)
            self.jack_acc_zeta.update(zeta_red)
        self.S_star = self.jack_acc.mean
        self.error_S_star = np.sqrt(len(self.F0) - 1) * np.sqrt(self.jack_acc.get_variance())
        self.mu = self.jack_acc_mu.mean
        self.mu_error = np.sqrt(len(self.F0) - 1) * np.sqrt(self.jack_acc_mu.get_variance())
        self.alpha = self.jack_acc_alpha.mean
        self.alpha_error = np.sqrt(len(self.F0) - 1) * np.sqrt(self.jack_acc_alpha.get_variance())
        self.zeta = self.jack_acc_zeta.mean
        self.zeta_error = np.sqrt(len(self.F0) - 1) * np.sqrt(self.jack_acc_zeta.get_variance())
    def get_S_star_excluding_index(self, excluded_index):
        reduced_F0 = np.delete(self.F0, excluded_index)
        assert(len(reduced_F0) + 1 == len(self.F0))
        generalised_gauss = GeneralisedGauss(alpha_min = self.alpha_min, zeta_min = self.zeta_min)
        cdf = CDFAccumulator()
        cdf.add_array(reduced_F0)
        x, cdf_x = cdf.get_vecdata()
        generalised_gauss.fit_cdf(x, cdf_x)
        integral, integral_error = integrate.quad(generalised_gauss.get_fitted_times_expx, self.volume_sanity_check.F0_acc, np.amax(reduced_F0) * 100, points = [np.amin(reduced_F0), np.amax(reduced_F0), np.mean(reduced_F0)])
        assert(integral > 0)
        S_star_red = - self.volume_sanity_check.F0_acc + np.log(integral)
        return S_star_red, generalised_gauss.mu_fit, generalised_gauss.alpha_fit, generalised_gauss.zeta_fit
    def compute_jack_estimates_from_pdf(self):
        """
        Old version of fits, uses pdf (histogram), has binning dependence
        Does not include jack estimates for generalised gaussian parameters
        """
        self.jack_acc = MomentsAcc()
        for idx in xrange(len(self.F0)):
            self.jack_acc.update(self.get_S_star_excluding_index_from_pdf(idx))
        self.S_star = self.jack_acc.mean
        self.error_S_star = np.sqrt(len(self.F0) - 1) * np.sqrt(self.jack_acc.get_variance())
    def get_S_star_excluding_index_from_pdf(self, excluded_index):
        reduced_F0 = np.delete(self.F0, excluded_index)
        assert(len(reduced_F0) + 1 == len(self.F0))
        generalised_gauss = GeneralisedGauss(alpha_min = self.alpha_min, zeta_min = self.zeta_min)
        bins = self.bins
        hist, bin_edges = np.histogram(reduced_F0, density = True, bins = bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        generalised_gauss.fit(bin_centres, hist)
        integral, integral_error = integrate.quad(generalised_gauss.get_fitted_times_expx, self.volume_sanity_check.F0_acc, np.amax(reduced_F0) * 100, points = [np.amin(reduced_F0), np.amax(reduced_F0), np.mean(reduced_F0)])
        assert(integral > 0)
        S_star_red = - self.volume_sanity_check.F0_acc + np.log(integral)
        return S_star_red

class OutlierRemovalUnbiasingEntropyLogOmega(object):
    """
    Fits generalised gaussian to histogram of F0 data.
    Integrates that to get the unbiased mean basin volume.
    Uses that to compute
    S^\star = \log \Omega = \log(V_\text{acc}) - \log(unbiased_mean_volume)
    and
    S = S^\star - \log(N!)
    """
    def __init__(self, F0, output_path, write=True, run_jackknife=True):
        self.F0 = F0
        self.output_path = output_path
        self.entropy_file_path = self.output_path + "/entropy_LogOmega"
        self.write = write
        self.run_jackknife = run_jackknife
    def compute_log_omega_entropy(self, volume_sanity_check):
        self.alpha_min = 0.01
        self.zeta_min = 0.01
        self.maximum_av_number_per_bin = 10
        self.generalised_gauss = GeneralisedGauss(alpha_min = self.alpha_min, zeta_min = self.zeta_min)
        bins = self.compute_desired_nr_bins(self.maximum_av_number_per_bin) 
        hist, bin_edges = np.histogram(self.F0, density = True, bins = bins)
        if self.write:
            plt.hist(self.F0, bins = bins, normed = True)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        cdf = CDFAccumulator()
        cdf.add_array(self.F0)
        x, cdf_x = cdf.get_vecdata()
        self.generalised_gauss.fit_cdf(x, cdf_x)
        if self.write:
            xp = np.linspace(bin_centres[0], bin_centres[-1], num = 500)
            plt.plot(xp, [self.generalised_gauss.get_fitted(xpi) for xpi in xp], "r", label = "Generalised Gaussian")
            plt.legend(loc='best', fancybox=True, framealpha=0.5)
            plt.xlabel(r"Free energy $F$")
            plt.ylabel(r"Probability density")
            save_pdf(plt, self.output_path + "/unbiasing_fit.pdf")
            plt.close()
        self.compute_integral(volume_sanity_check)
        self.S_star_no_jack = - volume_sanity_check.F0_acc + np.log(self.integral_no_jack)
        self.S_no_jack = self.S_star_no_jack - log_factorial(volume_sanity_check.nr_particles)
        self.jack_log_omega = JackLogOmega(self.F0, self.alpha_min, self.zeta_min, bins, volume_sanity_check)
        self.S_star = self.jack_log_omega.S_star
        self.error_S_star = self.jack_log_omega.error_S_star
        self.S = self.S_star - log_factorial(volume_sanity_check.nr_particles)
        self.error_S = self.error_S_star
        self.mu = self.jack_log_omega.mu
        self.mu_error = self.jack_log_omega.mu_error
        self.alpha = self.jack_log_omega.alpha
        self.alpha_error = self.jack_log_omega.alpha_error
        self.zeta = self.jack_log_omega.zeta
        self.zeta_error = self.jack_log_omega.zeta_error
        assert(self.S > 0)
        assert(self.S_star > 0)
        if self.write:
            self.write_to_file()
            self.plot_unbiased_pdf_vs_data(volume_sanity_check)
        
    def plot_unbiased_pdf_vs_data(self, volume_sanity_check):
        plt.xlabel(r"Free energy $F$")
        plt.ylabel(r"Un-biased PDF $\propto{P_\mathcal{B}(F)}\exp(F)$")
        bins = 32
        hist, bin_edges = np.histogram(self.F0, density = True, bins = bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        normalisation, error_norm = integrate.quad(self.generalised_gauss.get_times_expx_with_pars, volume_sanity_check.F0_acc, 
                                                   np.amax(self.F0) * 100, args = (self.mu, self.alpha, self.zeta, ), 
                                                   points = [np.amin(self.F0), np.amax(self.F0), np.mean(self.F0)])
        plt.yscale('log')
        plt.plot(bin_centres, [hist[i] * np.exp(bin_centres[i]) / normalisation for i in xrange(len(hist))], "o", label = "Data")
        xp = np.linspace(bin_centres[0], bin_centres[-1], num = 1000)
        plt.plot(xp, [self.generalised_gauss.get_times_expx_with_pars(xi, self.mu, self.alpha, self.zeta) / normalisation for xi in xp], 
                 label = r"$P_\mathcal{U}(F)$")
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        save_pdf(plt, self.output_path + "/unbiased_pdf_vs_data.pdf")
        plt.close()
    def compute_desired_nr_bins(self, maximum_av_number_per_bin):
        bins = 1
        while True:
            hist, bin_edges = np.histogram(self.F0, bins = bins)
            if (np.mean(hist) > maximum_av_number_per_bin):
                bins += 1
            else:
                return bins
    def compute_integral(self, volume_sanity_check):
        self.integral_no_jack, self.integral_error = integrate.quad(self.generalised_gauss.get_fitted_times_expx, 
                                                                    volume_sanity_check.F0_acc, np.amax(self.F0) * 100, 
                                                                    points = [np.amin(self.F0), np.amax(self.F0), 
                                                                              self.generalised_gauss.mu])
        assert(self.integral_no_jack > 0)
    def write_to_file(self):
        def prnt(name, value, error):
            print name + ":", value, "+/-", error
        print "Log of Omega entropy:"
        prnt("S_star", self.S_star, self.error_S_star)
        prnt("S", self.S, self.error_S)
        print "Generalised Gaussian parameters:"
        prnt("mu", self.mu, self.mu_error)
        prnt("alpha", self.alpha, self.alpha_error)
        prnt("zeta", self.zeta, self.zeta_error)
        f = ResultsFile(self.entropy_file_path)
        f.set_heading("ENTROPY_LOG_OMEGA")
        f.to_file("S_star", self.S_star)
        f.to_file("error_S_star", self.error_S_star)
        f.to_file("S", self.S)
        f.to_file("error_S", self.error_S)
        f.to_file("mu", self.mu)
        f.to_file("mu_error", self.mu_error)
        f.to_file("alpha", self.alpha)
        f.to_file("alpha_error", self.alpha_error)
        f.to_file("zeta", self.zeta)
        f.to_file("zeta_error", self.zeta_error)
        f.close()

if __name__ == "__main__":
    mu = 1
    alpha = 0.001
    zeta = 2
    pvs = np.random.lognormal(mean=mu, sigma=alpha,size=1000)
    bins = 100 
    hist, bin_edges = np.histogram(pvs, density = True, bins = bins)
    plt.hist(pvs, bins = bins, normed = True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    generalised_lognormal = GeneralisedLogNormal(alpha_min=0.0001, zeta_min=0.1)
    cdf = CDFAccumulator()
    cdf.add_array(pvs)
    x, cdf_x = cdf.get_vecdata()
    generalised_lognormal.fit_cdf(x, cdf_x)
    print "mu: ", generalised_lognormal.mu_fit
    print "alpha: ", generalised_lognormal.alpha_fit
    print "zeta: ", generalised_lognormal.zeta_fit 
    xp = np.linspace(bin_centres[0], bin_centres[-1], num = 2000)
    plt.plot(xp, [generalised_lognormal.get_fitted(xpi) for xpi in xp], "r", label = "Generalised LogNormal")
    plt.show()
    plt.plot(xp, generalised_lognormal.get_cdf(xp, generalised_lognormal.mu_fit, generalised_lognormal.alpha_offset, 
                                               generalised_lognormal.zeta_offset), label='generalised log')
    plt.plot(x, cdf_x, label='data')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()