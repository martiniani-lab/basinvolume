from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import numpy.polynomial.legendre as leg


class Gauss_Lobatto_abscissas(object):
    """
    Calculates the abscissas for Gauss-Lobatto integration of order n.
    Reference: http://mathworld.wolfram.com/LobattoQuadrature.html
    """

    def __init__(self, n):
        if n < 2:
            raise Exception("Gauss_Lobatto_abscissas: order n should be at least 2")
        self.n = n
        self.x = leg.legroots(leg.legder([int(i == n) for i in range(1, n + 1)]))
        self.x = np.insert(self.x, 0, -1.0)
        self.x = np.append(self.x, 1.0)

    def __call__(self):
        return self.x


class Gauss_Lobatto_weights(object):
    """
    Calculates the Gauss-Lobatto weights corresponding to the given abscissas.
    Reference: http://mathworld.wolfram.com/LobattoQuadrature.html
    """

    def __init__(self, x):
        self.n = len(x)
        if self.n < 2:
            raise Exception("Gauss_Lobatto_weights: order n should be at least 2")
        temp = 2.0 / self.n / (self.n - 1)
        self.w = np.zeros(self.n)
        for i in range(1, self.n - 1):
            poly_eval = leg.legval(x[i], [int(j == self.n - 1) for j in range(self.n)])
            self.w[i] = temp / (poly_eval * poly_eval)
        self.w[0] = temp
        self.w[-1] = temp

    def __call__(self):
        return self.w


def calculate_GL_integral(f):
    """
    Calculates the Gauss-Lobatto integral of the "function f", assumed to be evaluated
    at the correct abscissas.
    """
    weight = Gauss_Lobatto_weights(Gauss_Lobatto_abscissas(len(f))())()
    return sum(wi * fi for (wi, fi) in zip(weight, f))


def calculate_GL_integral_range(f, a, b, n=6):
    """
    Calculates the Gauss-Lobatto integral of the callable f, in the interval from a,b.
    """
    t = Gauss_Lobatto_abscissas(n)()
    x = [(ti + 1) * (b - a) / 2 + a for ti in t]
    integrand = [f(xi) * (b - a) / 2 for xi in x]
    return calculate_GL_integral(integrand)


def calculate_GL_integral_trafo(f, a, b, phi, phi_d, n=6):
    """
    Calculates the Gauss-Lobatto integral of the callable f, in the interval from a,b,
    with the coordiante transform given by phi, and its derivative phi_d.
    It is assumed that phi(1)==b and phi(-1)==a.
    """
    t = Gauss_Lobatto_abscissas(n)()
    integrand = [f(phi(ti)) * phi_d(ti) for ti in t]
    return calculate_GL_integral(integrand)


def print_Gauss_Lobatto_xw(n):
    """
    For given order n, prints the Gauss-Lobatto abscissae x and weights w.
    Sum of the weights should be 2.
    """
    x = Gauss_Lobatto_abscissas(n)()
    w = Gauss_Lobatto_weights(x)()
    print("abscissas:")
    print(x)
    print("weights:")
    print(w)
    print("integral of id:")
    print(np.sum(w))


if __name__ == "__main__":
    for i in range(2, 21):
        print("order: %d" % i)
        print_Gauss_Lobatto_xw(i)
