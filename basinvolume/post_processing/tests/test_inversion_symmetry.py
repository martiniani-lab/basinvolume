from __future__ import division
import numpy as np
import shutil
from basinvolume.post_processing.structural_properties import InversionSymmetry
from basinvolume.spheres import HS_Generate_Packing
from basinvolume.spheres import HS_Generate_Jammed_Packing

if __name__ == "__main__":
    fcc_dens = np.pi / (np.sqrt(2) * 3) - 0.1
    gen_packing = HS_Generate_Packing(32, method='fcc', bdim=3, packing_frac=fcc_dens,
                                      mu=1., sig=0., max_iter=1, start_iteration=0,
                                      distance_method='periodic')
    gen_packing.run()

    gen_jammed_packing = HS_Generate_Jammed_Packing(target_packing_frac=fcc_dens * 1.2,
                                                    packings_dir="packings",
                                                    use_cell_lists=False,
                                                    show=False, opt_pot_str='hs_wca')
    gen_jammed_packing.run()


    invsym = InversionSymmetry(".", verbose=False, force=True, existing_only=False,
                               jammed_packings_dir='jammed_packings', prefix="explore_bv_")
    invsym.run()

    invsym_dict = InversionSymmetry.read("explore_bv_jammed_packing0/analysis/inversion_symmetry")
    invsym_fcc = 1

    print("Local inversion symmetry of an FCC crystal (shoud be {}): {}"
          .format(invsym_fcc, invsym_dict['inversion_symmetry']))

    # Clean up
    shutil.rmtree('packings')
    shutil.rmtree('jammed_packings')
    shutil.rmtree('explore_bv_jammed_packing0')
