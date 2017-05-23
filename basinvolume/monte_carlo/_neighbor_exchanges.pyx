"""
# distutils: language = C++
"""
cimport numpy as np
import numpy as np

def random_neighbor_exchanges(np.ndarray[int] exchange_pattern, np.ndarray[double] energies,
                              np.ndarray[double] betas, int nexchanges):
    cdef int nrunners = len(exchange_pattern)
    cdef int naccept = 0
    cdef int i, j, inow, jnow
    cdef double w, rand

    for i in range(nrunners):
        exchange_pattern[i] = i

    for _ in range(nexchanges):
        i = 0
        j = 0
        while i == j:
            i = np.random.randint(0, nrunners)
            j = np.random.randint(0, nrunners)

        inow = exchange_pattern[i]
        jnow = exchange_pattern[j]
        w = np.exp((energies[inow]-energies[jnow])
                   * (betas[i]-betas[j]))

        rand = np.random.rand()
        if w > rand:
            exchange_pattern[i] = jnow
            exchange_pattern[j] = inow
            naccept += 1

    return naccept
