from __future__ import division
from __future__ import print_function
from builtins import object

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import gamma, gammaln
    from basinvolume.utils import CDFAccumulator
except ImportError as err:
    print(err)

from scipy.special import gammainc, gammaincc


class GeneralisedLogNormal(object):
    """
    Implements the generalised log normal distribution, see e.g.: http://link.springer.com/article/10.1007%2Fs00180-011-0233-9#page-1
    Parameters are as follows:
    PDF(mu, alpha, zeta; x) = zeta / (2**((zeta+1)/zeta) * alpha * gamma(1 / zeta) * x) * exp[-0.5*|log(x) - mu / alpha|**zeta]
    """

    def __init__(
        self,
        mu_initial=1,
        alpha_initial=1,
        zeta_initial=1,
        alpha_min=1e-10,
        zeta_min=1e-10,
        verbose=False,
    ):
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
        return (
            z
            / (2 ** ((z + 1.0) / z) * a * gamma(1.0 / z))
            * np.exp(-0.5 * np.power(np.abs((logx - mu) / a), z) - logx)
        )

    def get_cdf(self, x, mu, alpha_offset, zeta_offset):
        """
        Return CDF with the convention that for x = 0 CDF = 1
        """

        def _get_cdf(logx, mu, a, z):
            if logx <= mu:
                cdf = 0.5 * gammaincc(1.0 / z, 0.5 * np.power((mu - logx) / a, z))
            else:
                cdf = 1.0 - 0.5 * gammaincc(1.0 / z, 0.5 * np.power((logx - mu) / a, z))
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
        return (
            z
            / (2 ** ((z + 1.0) / z) * a * gamma(1.0 / z))
            * np.exp(-0.5 * np.power(np.abs((logx - mu) / a), z) + logx * (n / kappa - 1))
        )

    def get_log_times_xpow(self, x, mu, alpha_offset, zeta_offset, kappa, n):
        """
        returns the log of the probability
        """
        z = self.get_zeta(zeta_offset)
        a = self.get_alpha(alpha_offset)
        logx = np.log(x)
        return (
            np.log(z)
            - ((z + 1.0) / z) * np.log(2)
            - np.log(a)
            - gammaln(1.0 / z)
            - 0.5 * np.power(np.abs((logx - mu) / a), z)
            + logx * (n / kappa - 1)
        )

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
            print("self.mu", self.mu)
            print("self.alpha_offset", self.alpha_offset)
            print("self.zeta_offset", self.zeta_offset)

    def fit_cdf(self, x, cdf_x, initial_zeta=1.5):
        mean = np.mean(np.log(x))
        var = np.var(np.log(x))
        initial_mu = mean
        initial_zeta = initial_zeta
        initial_alpha = var
        print(
            "initial guess [mu, alpha, zeta]:",
            [initial_mu, initial_alpha, initial_zeta],
        )
        opt_gen, error_gen = curve_fit(
            self.get_cdf, x, cdf_x, [initial_mu, initial_alpha, initial_zeta]
        )
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print("self.mu", self.mu)
            print("self.alpha_offset", self.alpha_offset)
            print("self.zeta_offset", self.zeta_offset)


class LogNormal(GeneralisedLogNormal):
    def __init__(self, mu_initial=1, alpha_initial=1, alpha_min=1e-10, verbose=False):
        super(LogNormal, self).__init__(
            mu_initial=mu_initial,
            alpha_initial=alpha_initial,
            zeta_initial=2,
            zeta_min=2,
            alpha_min=alpha_min,
        )

    def get_zeta(self, zeta_offset):
        return 2

    def get(self, x, mu, alpha_offset):
        """
        Return PDF
        """
        z = 2
        a = self.get_alpha(alpha_offset)
        logx = np.log(x)
        return (
            z
            / (2 ** ((z + 1.0) / z) * a * gamma(1.0 / z))
            * np.exp(-0.5 * np.power(np.abs((logx - mu) / a), z) - logx)
        )

    def get_cdf(self, x, mu, alpha_offset):
        """
        Return CDF with the convention that for x = 0 CDF = 1
        """

        def _get_cdf(logx, mu, a, z):
            if logx <= mu:
                cdf = 0.5 * gammaincc(1.0 / z, 0.5 * np.power((mu - logx) / a, z))
            else:
                cdf = 1.0 - 0.5 * gammaincc(1.0 / z, 0.5 * np.power((logx - mu) / a, z))
            return 1.0 - cdf

        v_get_cdf = np.vectorize(_get_cdf)
        a = self.get_alpha(alpha_offset)
        z = 2
        logx = np.log(x)
        return v_get_cdf(logx, mu, a, z)

    def fit(self, data_x, data_y):
        mean = np.mean(np.log(x))
        var = np.var(np.log(x))
        opt_gen, error_gen = curve_fit(self.get, data_x, data_y, [mean, var])
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = 0
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = 2
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print("self.mu", self.mu)
            print("self.alpha_offset", self.alpha_offset)

    def fit_cdf(self, x, cdf_x):
        mean = np.mean(np.log(x))
        var = np.var(np.log(x))
        initial_mu = mean
        initial_alpha = var
        print("initial guess [mu, alpha]:", [initial_mu, initial_alpha])
        opt_gen, error_gen = curve_fit(self.get_cdf, x, cdf_x, [initial_mu, initial_alpha])
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = 0
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = 2
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print("self.mu", self.mu)
            print("self.alpha_offset", self.alpha_offset)


class GeneralisedGauss(object):
    """
    Implements the generalised gaussian distribution, see e.g.: http://en.wikipedia.org/wiki/Generalized_normal_distribution
    Parameters are as follows:
    PDF(mu, alpha, zeta; x) = zeta / (2 * alpha * gamma(1 / zeta)) * exp[-|x - mu|**zeta / alpha**zeta]
    """

    def __init__(
        self,
        mu_initial=1,
        alpha_initial=1,
        zeta_initial=1,
        alpha_min=1e-10,
        zeta_min=1e-10,
        verbose=False,
    ):
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
        return (
            self.get_zeta(zeta_offset)
            / (2 * self.get_alpha(alpha_offset) * gamma(1 / self.get_zeta(zeta_offset)))
            * np.exp(
                -np.power(
                    (np.abs(x - mu) / self.get_alpha(alpha_offset)),
                    self.get_zeta(zeta_offset),
                )
            )
        )

    def get_cdf(self, x, mu, alpha_offset, zeta_offset):
        """
        Return CDF with the convention that for x = - \infty, CDF = 1
        """
        alpha = self.get_alpha(alpha_offset)
        zeta = self.get_zeta(zeta_offset)
        return 0.5 - np.sign(x - mu) * 0.5 * gammainc(1 / zeta, (np.abs(x - mu) / alpha) ** zeta)

    def get_times_expx(self, x, mu, alpha_offset, zeta_offset):
        return (
            self.get_zeta(zeta_offset)
            / (2 * self.get_alpha(alpha_offset) * gamma(1 / self.get_zeta(zeta_offset)))
            * np.exp(
                -np.power(
                    (np.abs(x - mu) / self.get_alpha(alpha_offset)),
                    self.get_zeta(zeta_offset),
                )
                + x
            )
        )

    def get_times_expx_with_pars(self, x, mu, alpha, zeta):
        return self.get_times_expx(x, mu, alpha - self.alpha_min, zeta - self.zeta_min)

    def get_fitted(self, x):
        return self.get(x, self.mu, self.alpha_offset, self.zeta_offset)

    def get_fitted_times_expx(self, x):
        return self.get_times_expx(x, self.mu, self.alpha_offset, self.zeta_offset)

    def fit(self, data_x, data_y):
        opt_gen, error_gen = curve_fit(
            self.get, data_x, data_y, [np.mean(data_x), 2 * np.var(data_x), 2]
        )
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print("self.mu", self.mu)
            print("self.alpha_offset", self.alpha_offset)
            print("self.zeta_offset", self.zeta_offset)

    def fit_cdf(self, x, cdf_x):
        initial_mu = np.mean(x)
        initial_zeta = 1.5
        initial_alpha = np.sqrt(gamma(1 / initial_zeta) / gamma(3 / initial_zeta) * np.var(x))
        opt_gen, error_gen = curve_fit(
            self.get_cdf, x, cdf_x, [initial_mu, initial_alpha, initial_zeta]
        )
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = np.sqrt(np.diag(error_gen))
        if self.verbose:
            print("self.mu", self.mu)
            print("self.alpha_offset", self.alpha_offset)
            print("self.zeta_offset", self.zeta_offset)


if __name__ == "__main__":
    mu = 1
    alpha = 0.001
    zeta = 2
    pvs = np.random.lognormal(mean=mu, sigma=alpha, size=1000)
    bins = 100
    hist, bin_edges = np.histogram(pvs, density=True, bins=bins)
    plt.hist(pvs, bins=bins, normed=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    generalised_lognormal = GeneralisedLogNormal(alpha_min=0.0001, zeta_min=0.1)
    cdf = CDFAccumulator()
    cdf.add_array(pvs)
    x, cdf_x = cdf.get_vecdata()
    generalised_lognormal.fit_cdf(x, cdf_x)
    print("mu: ", generalised_lognormal.mu_fit)
    print("alpha: ", generalised_lognormal.alpha_fit)
    print("zeta: ", generalised_lognormal.zeta_fit)
    xp = np.linspace(bin_centres[0], bin_centres[-1], num=2000)
    plt.plot(
        xp,
        [generalised_lognormal.get_fitted(xpi) for xpi in xp],
        "r",
        label="Generalised LogNormal",
    )
    plt.show()
    plt.plot(
        xp,
        generalised_lognormal.get_cdf(
            xp,
            generalised_lognormal.mu_fit,
            generalised_lognormal.alpha_offset,
            generalised_lognormal.zeta_offset,
        ),
        label="generalised log",
    )
    plt.plot(x, cdf_x, label="data")
    plt.legend(loc="best", fancybox=True, framealpha=0.5)
    plt.show()
