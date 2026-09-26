from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists
from basinvolume.spheres import HS_Generate_Packing
import numpy as np

if __name__ == "__main__":

    ######################## Simplest case ### runs
    radii = np.array(
        [
            1.05147999,
            0.81830371,
            0.92429938,
            0.89301688,
            1.17161467,
            0.917398,
            1.09963772,
            1.40203985,
            1.25257231,
            0.91215703,
            0.93071242,
            1.09106393,
            0.66626746,
            0.8275829,
            1.09858217,
            0.97513732,
        ]
    )
    boxvec = np.array([5.69339383, 5.69339383, 5.69339383])
    specific = True
    ncellxscale = 1.0
    use_frozen = False
    thing = CheckOverlapPeriodicCellLists(radii, boxvec)

    # ### slightly more complex

    # nparticles = 16
    # dim = 3
    # packing_frac = 0.4
    # # radii = None
    # sig = 0.2
    # seeds = {
    #     'seed_takestep': 42,
    #     'seed_generate_packing': 43,
    #     'seed_swap': 44,
    #     'seed_probability_step_pattern': 46,
    # }
    # print("stuff")
    # thing = HS_Generate_Packing(
    #     nparticles,
    #     method="quench",
    #     bdim=dim,
    #     boxv=None,
    #     packing_frac=packing_frac,
    #     hs_radii=radii,
    #     mu=1,
    #     sig=sig,
    #     hsf_stepsize=1e-3,
    #     max_iter=1e3,
    #     use_cell_lists=True,
    #     seeds=seeds,
    #     single=True,
    # )
    # print("here")
    # thing._initialise()
    # print("not here")
    # thing.run()
    # print("d")
