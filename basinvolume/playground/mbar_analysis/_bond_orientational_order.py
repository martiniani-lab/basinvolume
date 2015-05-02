from __future__ import division
import numpy as np
from scipy.special import sph_harm

def qsum(coords, atom_index, nnatoms, order, ndim=3, deg=6):
    """
    this method compute the qsum, necessary for computing 
    coords: array
        coordinates of the whole system
    atom_index: int
        index of the particle for which to compute the qsum
    nn_atoms: array
        array of indeces of the nearest neighbours
    ndim: int
        dimensionality of space (box)
    order: int
        order of the spherical harmonic (m)
    deg: int
        degree of the spherical harmonic (l)
    """
    assert ndim == 3, "2 dimensional case not implemented yet"
    n = nnatoms.size
    i = atom_index
    qsum = np.complex(0.,0.)
    for j in nnatoms:
        r = coords[j*ndim:j*ndim+ndim] - coords[i*ndim:i*ndim+ndim]
        theta = np.arctan(r[1]/r[0])
        phi = np.arccos(r[2]/np.linalg.norm(r)) 
        Y = sph_harm(order, deg, theta, phi)
        qsum += Y
    return qsum / n

def find_nearest_neighbors(coords, hs_radii):
    ndim = coords.size // hs_radii.size
    nparticles = hs_radii.size
    nnatoms_list = [[] for _ in xrange(nparticles)]
    for i in xrange(nparticles):
        for j in xrange(i, nparticles):
            dx = coords[j*ndim:j*ndim+ndim] - coords[i*ndim:i*ndim+ndim]
            if dx <= hs_radii[i]+hs_radii[j]:
                nnatoms_list[i].append(j)
                nnatoms_list[j].append(i)
    return nnatoms_list

def _bond_orientational_order(coords, atom_index, nnatoms, ndim=3, deg=6):
    q = 0.
    for m in xrange(-deg,deg):
        c = qsum(coords, atom_index, nnatoms, m, ndim=ndim, deg=deg)
        q += np.absolute(c)**2
    return np.sqrt(q * 4 * np.pi / (2*deg+1))

def bond_orientation_order_single(coords, hs_radii, atom_index, ndim=3, deg=6):
    nnatoms_list = find_nearest_neighbors(coords, hs_radii)
    nnatoms = np.array(nnatoms_list[atom_index])
    return _bond_orientational_order(coords, atom_index, nnatoms, ndim=ndim, deg=deg)

def bond_orientation_order_all(coords, hs_radii, ndim=3, deg=6):
    nnatoms_list = find_nearest_neighbors(coords, hs_radii)
    boo_list = []
    for i in xrange(hs_radii.size):
        nnatoms = np.array(nnatoms_list[i])
        boo = _bond_orientational_order(coords, i, nnatoms, ndim=ndim, deg=deg)
        boo_list.append(boo)
    return np.array(boo_list)

if __name__ == "__main__":
    