import numpy as np

from pele.mc import MonteCarlo
from pele.potentials import BasePotential
from pele.takestep import RandomDisplacement
from pele.utils import rotations

from pele.mindist import MeasurePeriodic

class FlatPot(BasePotential):
    def __init__(self, x0, boxvec, k=50):
        self.k = float(k)
        self.x0 = np.array(x0)
        self.measure = MeasurePeriodic(boxvec)
    
    def get_dist(self, x):
        return self.measure.get_dist(x, self.x0)
    
    def getEnergy(self, x):
        r = self.get_dist(x)
        r = max(r, 1e-6)
        
        E = +0.5 * self.k * r**2 + float(self.x0.size - 1) * np.log(r)
        return E
    

def main():
    natoms = 24
    L = 4.
    boxvec = np.ones(3) * L
    x0 = np.random.uniform(0, L, 3*natoms)
    p = FlatPot(x0, boxvec, k=50)
    stepsize = .01
    
    x = x0 + np.random.uniform(-stepsize, stepsize, x0.size) * 4
    mc = MonteCarlo(x, p, RandomDisplacement(stepsize))
    mc.setPrinting(frq=1000)
    naccepted = 0
    iprint = 500
    with open("timeseries", "w") as fout:
        for i in xrange(1000000):
            mc.run(1)
            fout.write("{}\n".format(p.get_dist(mc.coords)))
            if i % iprint == 1:
                print float(mc.naccepted - naccepted) / iprint, mc.naccepted
                naccepted = mc.naccepted

def _subtract_com(x):
    x = x.reshape(-1,3)
    com = x.mean(0)
    return (x - com[np.newaxis, :]).ravel()    

def generate_sample(x0, k=50, subtract_com=False):
    v = rotations.vec_random_ndim(x0.size)
    
    if subtract_com:
        N = x0.size / 3
#        k = k * N / (N-1)
        k = k * (N-1) / N
    
    x = v * np.abs(np.random.normal()) / np.sqrt(k)
    
    if subtract_com:
        if subtract_com:
            x = _subtract_com(x)
    
    return x0 + x 


def generate_samples(k=50, subtract_com=True):
    natoms = 4
    L = 4.
    boxvec = np.ones(3) * L
    x0 = np.random.uniform(0, L, 3*natoms)
    if subtract_com:
        x0 = _subtract_com(x0)
    p = FlatPot(x0, boxvec, k=k)
    stepsize = .01
    
    with open("timeseries", "w") as fout:
        for i in xrange(100000):
            x = generate_sample(x0, k=k, subtract_com=subtract_com)
            fout.write("{}\n".format(p.get_dist(x)))


def makehist(k=50):
    data = np.genfromtxt("timeseries")
    from matplotlib import pyplot as plt
    c, x = np.histogram(data, bins=50)
    print x.shape
    print c.shape
    cmax = np.max(c)
    print cmax
    dx = x[1]-x[0]
    plt.bar(x[:-1] + dx/2, c, dx)
    
    x0 = np.linspace(0, x.max(), 1000)
    print x0
    plt.plot(x0, cmax* np.exp(-0.5 *k * x0**2), 'k', lw=3)
    plt.show()

if __name__ == "__main__":
#    main()
    generate_samples()
    makehist()