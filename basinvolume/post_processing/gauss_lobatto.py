from __future__ import division
import numpy as np
import numpy.polynomial.legendre as leg
   
class Gauss_Lobatto_abscissas(object):
    """
    Calculates the abscissas for Gauss-Lobatto integration of order n.
    Reference: http://mathworld.wolfram.com/LobattoQuadrature.html
    """
    def __init__(self, n):
        if (n<2):
            raise Exception("Gauss_Lobatto_abscissas: order n should be at least 2")
        self.n = n
        self.x = leg.legroots(leg.legder([int(i==n) for i in xrange(1,n+1)]))
        self.x = np.insert(self.x,0,-1.0)
        self.x = np.append(self.x,1.0)
    def __call__(self):
        return self.x

class Gauss_Lobatto_weights(object):
    """
    Calculates the Gauss-Lobatto weights corresponding to the given abscissas.
    Reference: http://mathworld.wolfram.com/LobattoQuadrature.html
    """
    def __init__(self, x):
        self.n = len(x)
        if (self.n<2):
            raise Exception("Gauss_Lobatto_weights: order n should be at least 2")
        temp = 2.0/self.n/(self.n-1)
        self.w = np.zeros(self.n)
        for i in xrange(1,self.n-1):
            poly_eval = leg.legval(x[i],[int(j==self.n-1) for j in xrange(self.n)])
            self.w[i] = temp/(poly_eval*poly_eval)
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
    return sum(wi*fi for (wi,fi) in zip(weight,f))
    
def print_Gauss_Lobatto_xw(n):
    x = Gauss_Lobatto_abscissas(n)()
    w = Gauss_Lobatto_weights(x)()
    print "abscissas:"
    print x
    print "weights:"
    print w
    print "integral of id:"
    print np.sum(w)

if __name__ == "__main__":
    for i in xrange(2,21):
        print "order: %d"%i
        print_Gauss_Lobatto_xw(i)
    
    