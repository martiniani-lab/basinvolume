from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import copy
import logging
from basinvolume.utils import CrossValidationCost
from pele.potentials import BasePotential
from pele.optimize import LBFGS
from sklearn.neighbors import KernelDensity


class CrossValidationBandwidthSelection(object):
    """
    Use leave-one-out cross validation to estimate bandwidth.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_density_estimation
    http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode15.html
    http://www.control.aau.dk/~tk/undervisning/PhDAdvSI/Litterature/MadsenAndHolst2006.pdf
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
    """

    def __init__(self, data, kernel="gaussian", h_initial=2):
        pot = CrossValidationCost(data, kernel=kernel)
        optimizer = LBFGS(np.asarray([h_initial]), pot)
        logging.info("run bandwidth optimization")
        result = optimizer.run()
        # logging.info("done")
        self.opt_bandwidth = result.coords


def get_bandwidth_estimate(data, kernel="gaussian", method="cross_validation"):
    nr_samples = len(data)
    std_samples = np.std(data)
    silverman_bandwidth = ((4 * std_samples**5) / (3 * nr_samples)) ** (
        1 / 5
    )
    if method == "Silverman":
        return np.asarray([silverman_bandwidth])
    loocv = CrossValidationBandwidthSelection(
        data, kernel=kernel, h_initial=silverman_bandwidth
    )
    return loocv.opt_bandwidth


def get_pdf(data, x_sample_positions, bandwidth=2, kernel="gaussian"):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
        data[:, np.newaxis]
    )
    log_pdf = kde.score_samples(x_sample_positions[:, np.newaxis])
    return np.exp(log_pdf)


def sample_from_pdf(
    data, nr_samples, bandwidth=2, kernel="gaussian", random_state=None
):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
        data[:, np.newaxis]
    )
    return kde.sample(nr_samples, random_state=random_state)[:, 0]


def compute_raw_moment(pdf_x, pdf_pdf, exponent=0):
    if len(pdf_x) != len(pdf_pdf):
        raise Exception("illegal input")
    return integrate.romb(
        [pdf_pdfi * xi**exponent for (pdf_pdfi, xi) in zip(pdf_pdf, pdf_x)],
        dx=pdf_x[1] - pdf_x[0],
    )


def compute_central_moment(pdf_x, pdf_pdf, exponent=0):
    if len(pdf_x) != len(pdf_pdf):
        raise Exception("illegal input")
    mu = compute_raw_moment(pdf_x, pdf_pdf, exponent=1)
    return integrate.romb(
        [
            pdf_pdfi * (xi - mu) ** exponent
            for (pdf_pdfi, xi) in zip(pdf_pdf, pdf_x)
        ],
        dx=pdf_x[1] - pdf_x[0],
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    np.random.seed(42)
    data = np.random.randn(100)
    logging.info("mean data: {}".format(np.mean(data)))
    logging.info("var data: {}".format(np.var(data)))
    silverman_bw = get_bandwidth_estimate(data, method="Silverman")
    bandwidth = get_bandwidth_estimate(data)
    logging.info("Silverman bw: {}".format(silverman_bw))
    logging.info("cv bw: {}".format(bandwidth))
    n_integrate = 2**18 + 1
    pdf_x = np.linspace(-13, 13, n_integrate)
    pdf_pdf = get_pdf(data, pdf_x, bandwidth=bandwidth)
    plt.xlabel(r"Value $x$")
    plt.ylabel(r"PDF($x$)")
    plt.plot(pdf_x, pdf_pdf)

    def nd(x, h2):
        return np.exp(-0.5 * x**2 / h2) / np.sqrt(2 * np.pi * h2)

    plt.plot(pdf_x, nd(pdf_x, 1))
    plt.show()
    max_order = 4
    numerical_raw_moments = [
        compute_raw_moment(pdf_x, pdf_pdf, exponent=exponent)
        for exponent in range(max_order + 1)
    ]
    numerical_central_moments = [
        compute_central_moment(pdf_x, pdf_pdf, exponent=exponent)
        for exponent in range(max_order + 1)
    ]
    for n in range(max_order + 1):
        logging.info("order: {}".format(n))
        logging.info(numerical_raw_moments[n])
        logging.info(numerical_central_moments[n])
