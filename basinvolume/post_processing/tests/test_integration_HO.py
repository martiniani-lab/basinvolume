from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.integrate import quad, romb
from math import log, pi
from basinvolume.post_processing import calculate_GL_integral_range
from basinvolume.post_processing import (
    spring_constants_variable_transform,
    calculate_GL_integral_with_transform,
)
from basinvolume.post_processing import (
    F_Basin_From_MC_Data_Free_COM,
    F_Basin_From_MC_Data,
)


def F_HO(N, d, k):
    return -0.5 * N * d * log(2 * pi / k)


if __name__ == "__main__":
    """
    Test of integration.
    """
    print("===Test integration===")
    nr_particles = 10
    dimension = 3
    k1 = 22
    k2 = 44
    ref_F1 = F_HO(nr_particles, dimension, k1)
    ref_F2 = F_HO(nr_particles, dimension, k2)
    int_F1 = []
    int_F1.append(ref_F2 - 0.5 * nr_particles * dimension * log(k2 / k1))
    int_F1.append(
        ref_F2 - 0.5 * quad(lambda x: nr_particles * dimension / x, k1, k2)[0]
    )
    # computation by GL without variable transform, order 4
    int_F1.append(
        ref_F2
        - 0.5
        * calculate_GL_integral_range(lambda x: nr_particles * dimension / x, k1, k2, 4)
    )
    # computation by GL with variable transform, order 4
    kappa_const = 1
    k_order4 = spring_constants_variable_transform(
        4,
        k2,
        nr_particles * dimension / k1,
        nr_particles,
        dimension,
        k1,
        kappa_const=kappa_const,
    )
    int_F1.append(
        ref_F2
        - 0.5
        * calculate_GL_integral_with_transform(
            [nr_particles * dimension / ki for ki in k_order4],
            k2,
            nr_particles,
            dimension,
            k1,
            kappa_const=kappa_const,
        )[0]
    )
    print("reference:")
    print(ref_F1)
    print("integrated:")
    for res in int_F1:
        print(res)
    """
    Test of volume computation.
    """
    print("===Test volume computation===")
    L = 1000
    box_volume = L * L * L
    nr_points = 6
    k_max = 100
    k_min = 0
    prob = 1
    displ_k0 = (L / 2) ** 2
    nr_particles = 2
    dimension = 3
    displ_k_min_trafo = displ_k0 * 6.0

    ####spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=0.0, kappa_const=1.0):
    k = spring_constants_variable_transform(
        nr_points,
        k_max,
        displ_k0,
        nr_particles,
        dimension,
        k_min=k_min,
        kappa_const=kappa_const,
    )
    k_ = spring_constants_variable_transform(
        nr_points,
        k_max,
        displ_k_min_trafo,
        nr_particles,
        dimension,
        k_min=k_min,
        kappa_const=kappa_const,
    )
    nd = nr_particles * dimension
    usq = [nd / (ki + nd / displ_k0) for ki in k]
    usq_ = [nd / (ki + nd / displ_k0) for ki in k_]

    vol = F_Basin_From_MC_Data_Free_COM(
        dimension, nr_particles, k, usq, prob, kappa_const=kappa_const
    )
    vol_ = F_Basin_From_MC_Data_Free_COM(
        dimension,
        nr_particles,
        k_,
        usq_,
        prob,
        kappa_const=kappa_const,
        displ_k_min_trafo=displ_k_min_trafo,
    )

    F0, sigF0, far, sigfar = vol.get_free_energy_F0(np.ones(nr_points))
    F0_, sigF0_, far_, sigfar_ = vol_.get_free_energy_F0(np.ones(nr_points))
    print("F0: ", F0)
    print("F0_: ", F0_)
    print("sigF0: ", sigF0)
    print("sigF0_: ", sigF0_)
    print("far: ", far)
    print("far_: ", far_)
    print("sigfar: ", sigfar)
    print("sigfar_: ", sigfar_)

    ###__init__(self, dimension, nr_particles, k_values, displacements, box_volume, prob, kappa_const=1.0, displ_k_min_trafo=None):
    vol_fixed1 = F_Basin_From_MC_Data(
        dimension,
        nr_particles,
        k,
        usq,
        box_volume,
        prob,
        kappa_const=kappa_const,
    )
    vol_fixed1_ = F_Basin_From_MC_Data(
        dimension,
        nr_particles,
        k_,
        usq_,
        box_volume,
        prob,
        kappa_const=kappa_const,
        displ_k_min_trafo=displ_k_min_trafo,
    )
    vol_fixed1__ = F_Basin_From_MC_Data(
        dimension,
        nr_particles,
        k_,
        usq_,
        1.0,
        prob,
        kappa_const=kappa_const,
        displ_k_min_trafo=displ_k_min_trafo,
    )
    (
        F0_fixed1,
        sigF0_fixed1,
        far_fixed1,
        sigfar_fixed1,
    ) = vol_fixed1.get_free_energy_F0(np.ones(nr_points))
    (
        F0_fixed1_,
        sigF0_fixed1_,
        far_fixed1_,
        sigfar_fixed1_,
    ) = vol_fixed1_.get_free_energy_F0(np.ones(nr_points))
    (
        F0_fixed1__,
        sigF0_fixed1__,
        far_fixed1__,
        sigfar_fixed1__,
    ) = vol_fixed1__.get_free_energy_F0(np.ones(nr_points))
    print("F0_fixed1: ", F0_fixed1)
    print("F0_fixed1_: ", F0_fixed1_)
    print("F0_fixed1__: ", F0_fixed1__)
    print("sigF0_fixed1: ", sigF0_fixed1)
    print("sigF0_fixed1_: ", sigF0_fixed1_)
    print("sigF0_fixed1__: ", sigF0_fixed1__)
    print("far_fixed1: ", far_fixed1)
    print("far_fixed1_: ", far_fixed1_)
    print("far_fixed1__: ", far_fixed1__)
    print("sigfar_fixed1: ", sigfar_fixed1)
    print("sigfar_fixed1_: ", sigfar_fixed1_)
    print("sigfar_fixed1__: ", sigfar_fixed1__)
