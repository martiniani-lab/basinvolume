from __future__ import division
from basinvolume.utils import cround
from pele.distance import get_distance, Distance
import numpy as np
from numpy import linalg as la

class SimpleSolidAngleNeighbors(object):
    def __init__(self, center, coords, nparticles, boxv):
        self.center = center
        self.coords = coords
        self.nparticles = nparticles
        self.boxv = boxv
        self.boxdim = int(self.coords.size / self.nparticles)
        if self.boxdim != 3 or self.boxdim * self.nparticles != self.coords.size:
            raise Exception("SimpleSolidAngleNeighbors: only 3D")
        self.neighbor_labels = []
        self.nn_vector = []
        self.weight = []
        self.compute_neighbors_weights()

    def compute_neighbors_weights(self):
        # This follows exactly: ftp://ftp.aip.org/epaps/journ_chem_phys/E-JCPSA6-136-022224/sann.c
        # this does not exclude rattlers from the shell
        count = self.nparticles - 1
        if count < 3:
            raise Exception("SimpleSolidAngleNeighbors: too few particles")
        d = dict([(self.get_distance(k), k) for k in xrange(self.nparticles) if k != self.center])
        distance_sum = 0
        sk = sorted(d.keys())
        for s in sk[0:3]:
            distance_sum += s
            self.neighbor_labels.append(d[s])
            self.nn_vector.append(self.get_delta_vector(d[s]))
        radius = distance_sum
        i = 3
        while (i < count) and (radius > sk[i]):
            distance_sum += sk[i]
            radius = distance_sum / (i - 2)
            self.neighbor_labels.append(d[sk[i]])
            self.nn_vector.append(self.get_delta_vector(d[sk[i]]))
            i += 1
        if i == count:
            raise Exception("SimpleSolidAngleNeighbors: too few particles")
        self.weight = np.asarray([1 - s / radius for s in sk])
        self.nr_neighbors = i
        assert(self.center not in self.neighbor_labels)

    def get_delta_vector(self, j):
        return np.linalg.norm(get_distance(
            self.coords[j * self.boxdim : (j + 1) * self.boxdim],
            self.coords[self.center * self.boxdim : (self.center + 1) * self.boxdim],
            self.boxdim, Distance.PERIODIC, box=self.boxv))

    def get_distance(self, k):
        assert(k != self.center)
        return la.norm(self.get_delta_vector(k))
