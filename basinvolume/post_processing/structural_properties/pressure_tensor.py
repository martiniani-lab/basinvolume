from __future__ import division
import numpy as np
import os
import traceback
import ConfigParser
import logging
from basinvolume.utils import trymakedir
from pele.utils._pressure_tensor import pressure_tensor
from pele.potentials import HS_WCA, InversePowerStillingerCut
from pele.optimize._quench import modifiedfire_cpp
from _structural_analysis import StructuralAnalysis


class PressureTensor(StructuralAnalysis):

    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', force=False, existing_only=True, opt_pot_str='hs_wca',
                 prefix='explore_bv_', verbose=True, use_cell_lists=True):
        super(PressureTensor,self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                            analysis_dir=analysis_dir, force=force,
                                            existing_only=existing_only, prefix=prefix,
                                            verbose=verbose, use_cell_lists=use_cell_lists)
        self.opt_pot_str = opt_pot_str

    @staticmethod
    def read(pressure_fname):
        configf = ConfigParser.ConfigParser()
        configf.read(pressure_fname)
        pressure_dict = {}
        pressure_dict['P'] = \
            configf.getfloat('PRESSURE', 'P')
        pressure_dict['maxshear_xyplane'] = \
            configf.getfloat('PRESSURE', 'maxshear_xyplane')
        pressure_dict['Ptensor'] = \
            np.array([float(x) for x in configf.get('PRESSURE', 'Ptensor').split()])
        pressure_dict['E'] = \
            configf.getfloat('ENERGY', 'E')
        return pressure_dict

    def run(self):
        """compute the pressure tensor for packings
        exisisting_only: bool
            run on already existing packings only
        pinit : bool
            initialise printing
        """
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzd' in fname or 'xyd' in fname:
                compute = False
                packing_name = os.path.splitext(fname)[0]
                base_directory_path = os.path.join(self.workspace, self.prefix + str(packing_name))
                configpath = os.path.join(self.jammed_packings_dir, packing_name + '.config')
                self._import_packing_config_file(configpath)
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    pressure_fname = os.path.join(analysis_dir_path,'pressure_data')
                    try:
                        self.read(pressure_fname)
                    except Exception:
                        compute = True
                    if compute or self.force:
                        if self.verbose:
                            logging.info("Calculating pressure: {}"
                                         .format(self.prefix + str(packing_name)))
                        trymakedir(analysis_dir_path)
                        self.coords, self.hs_radii, self.ss_radii, _ = self._import_packing_configuration(fname)
                        potential = self.get_potential()
                        # refine structure (does not make a difference if tol was small enough to start with)
                        # if self.packing_frac < 0.835:
                        #     fire_maxstep = np.amin(self.hs_radii) * self.sca
                        #     res = modifiedfire_cpp(self.coords, potential, maxstep=fire_maxstep,
                        #                            nsteps=1e6, tol=1e-11, iprint=-1)
                        #     self.coords = res.coords
                        p, ptensor = pressure_tensor(potential, self.coords, self.vcavity, self.bdim)
                        max_shear_xyplane = np.sqrt(((ptensor[0] - ptensor[3]) / 2.) ** 2 + ptensor[1] ** 2)
                        energy = potential.getEnergy(self.coords)
                        with open(pressure_fname, 'w') as f:
                            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND \n')
                            f.write('[PRESSURE]\n')
                            f.write('P: {:.16f}\n'.format(p))
                            f.write('maxshear_xyplane: {:.16f}\n'.format(max_shear_xyplane))
                            f.write('Ptensor: ')
                            for val in ptensor:
                                f.write('{:.16f} '.format(val))
                            f.write('\n')
                            f.write('[ENERGY]\n')
                            f.write('E: {:.16f}\n'.format(energy))

    def get_potential(self):
        # here put a flag and pick potential
        if self.opt_pot_str.lower() == 'hs_wca':
            pot = HS_WCA(eps=self.eps, sca=self.sca,
                         radii=self.hs_radii, boxvec=self.boxv, ndim=self.bdim,
                         distance_method=self.distance_method, pot_kwargs=self.pot_kwargs)
        elif self.opt_pot_str.lower() == 'inverse_power_stillinger':
            pow = self.pot_kwargs['pow']
            rcut = self.pot_kwargs["rcut"]
            pot_optimizer = InversePowerStillingerCut(pow,
                self.stillinger_a_radii, ndim=self.bdim,
                boxvec=self.boxv, rcut=rcut, use_cell_lists=True)
        else:
            raise NotImplementedError
        return pot


def worker_pressure(workspace, kwargs):
    try:
        pressure = PressureTensor(workspace, **kwargs)
        pressure.run()
    except:
        logging.error('worker_pressure worker: %s' % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the pressure tensor for "
                                     "jammed packings.")
    parser.add_argument("-d", "--workspace-dir", type=str, help="Top-level dir containing "
                        "the packings, e.g. 'n32_phi88_2D'.")
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--nonex", action='store_false', help="Run also for packings "
                        "for which there are no work folders ('explore_bv_[...]', "
                        "created e.g. by parallel tempering).", default=True)
    parser.add_argument("--prefix", type=str, help="Prefix for the work directory. "
                        "Default: 'explore_bv_'", default='explore_bv_')
    parser.add_argument("--input-dir", type=str, help="Directory containing the "
                        "jammed packings. Default: 'jammed_packings'", default='jammed_packings')
    parser.add_argument("--opt-pot", type=str, help="Optimizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)

    kwargs = dict(force=args.force, existing_only=args.nonex,
                  jammed_packings_dir=args.input_dir, prefix=args.prefix,
                  opt_pot_str=args.opt_pot)

    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    pts = PressureTensor(workspace_dir, **kwargs)
    pts.run()
