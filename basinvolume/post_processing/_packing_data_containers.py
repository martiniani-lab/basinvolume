from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import object
import re
import os
import numpy as np
import configparser
from basinvolume.utils import import_packing, Bunch
from basinvolume.spheres import read_jammed_packing_config, read_packing_config

try:
    import matplotlib.pyplot as plt
except ImportError as err:
    print(err)


class PackingDataSet(object):
    """
    this assumes naming convention n24_phi50_phi70_3D

    Parameters
    ----------
    set_path : string
        path to set of explore_bv folders, for instance /path/to/n24_phi50_phi70_3D
    extras: list
        extras is any arbitrary list of data that one might want to add on to this class
        during the analysis for convenience
    """

    def __init__(self, set_path):
        self.set_path = set_path
        self.set_name = os.path.split(self.set_path)[1]
        str_values = re.findall("\d+", self.set_name)
        self.nparticles, self.hs_phi = float(str_values[0]), float("0." + str_values[1])
        self.ss_phi, self.bdim = float("0." + str_values[2]), float(str_values[3])
        self.packing_data = []
        self.free_energies = []
        self.free_energies_err = []
        self.pressures = []
        self.energies = []
        self.contacts = []
        self.boos = []
        self.invsyms = []
        self.extras = []

    def add_data_all(self, packing_data):
        """
        packing data is a list of PackingData objects
        # test that number of contacts is sufficient for bulk modulus to be positive,
        # see eq 4 in http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.095704
        # see eq 19 in arXiv:1406.1529
        """
        self.packing_data.extend(packing_data)
        for data in packing_data:
            # the reason why they must all be true is because we are interested in the realation among these variables
            if (
                data.F is not None
                and data.Ferr is not None
                and data.P is not None
                and data.energy is not None
                and data.Z is not None
                and data.boo is not None
            ):
                if int(np.sum(data.Zlist)) >= int(
                    2 * (((data.rattlers == 1).sum() // self.bdim - 1) * self.bdim + 1)
                ):
                    self.free_energies.append(data.F)
                    self.free_energies_err.append(data.Ferr)
                    self.pressures.append(data.P)
                    self.energies.append(data.energy)
                    self.contacts.append(data.Z)
                    self.boos.append(data.boo)
                    if data.invsym is not None:
                        self.invsyms.append(data.invsym)

    def add_data_structure(self, packing_data):
        """
        packing data is a list of PackingData objects
        # test that number of contacts is sufficient for bulk modulus to be positive,
        # see eq 4 in http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.095704
        # see eq 19 in arXiv:1406.1529
        """
        self.packing_data.extend(packing_data)
        for data in packing_data:
            # the reason why they must all be true is because we are interested in the realation among these variables
            if data.P is not None and data.Z is not None and data.boo is not None:
                if int(np.sum(data.Zlist)) >= int(
                    2 * (((data.rattlers == 1).sum() // self.bdim - 1) * self.bdim + 1)
                ):
                    self.pressures.append(data.P)
                    self.contacts.append(data.Z)
                    self.boos.append(data.boo)
                    if data.invsym is not None:
                        self.invsyms.append(data.invsym)

    def add_extras(self, extra):
        self.extras.extend(np.array(extra).tolist())


class PackingData(object):
    def __init__(self, name, configpath, packing_path=None):
        self.eps = 1.0
        self.frozen = False
        self.name = name
        self.configpath = configpath
        imp_jammed_packing = read_jammed_packing_config(self.configpath)
        self.nparticles = imp_jammed_packing["nparticles"]
        self.packing_frac = imp_jammed_packing["packing_frac"]
        self.bdim = imp_jammed_packing["bdim"]
        self.ndim = imp_jammed_packing["ndim"]
        self.boxv = imp_jammed_packing["boxv"].copy()
        self.vcavity = imp_jammed_packing["vcavity"]
        self.sca = imp_jammed_packing["sca"]
        if packing_path is not None:
            self._import_packing_configuration(packing_path)
        self.jammed_packing_name = os.path.split(os.path.splitext(self.configpath)[0])[1]
        self.F = None
        self.Ferr = None
        self.P = None
        self.Ptensor = None
        self.Facc = None
        self.Z = None
        self.Zlist = None
        self.boo = None
        self.boolist = None
        self.invsym = None

    def _import_packing_configuration(self, path):
        # path = os.path.join(self.packings_dir, fname)
        packing = import_packing(path, True, self.bdim, self.sca)
        self.coords = packing["coords"]
        self.hs_radii = packing["hs_radii"]
        self.ss_radii = packing["ss_radii"]
        self.rattlers = packing["stable_atoms_float_bdim"]

    def import_packing_config(self, configpath_packing):
        imp_packing = read_packing_config(configpath_packing)
        self.hs_mean = imp_packing["radii_mean"]
        self.hs_stddev = imp_packing["radii_stddev"]

    def import_volume_data(self, path, title="VOLUME_FULL_PT", vfluid_title="VOLUME_HS_FLUID"):
        if os.path.isfile(path):
            configf = configparser.ConfigParser()
            configf.read(path)
            self.F, self.Ferr = configf.getfloat(title, "F0"), configf.getfloat(title, "sigF0")
            try:
                self.Facc = configf.getfloat(vfluid_title, "F0_acc")
            except Exception as e:
                pass

    def import_pressure_data(self, path, pressure_title="PRESSURE", energy_title="ENERGY"):
        if os.path.isfile(path):
            configf = configparser.ConfigParser()
            configf.read(path)
            self.P = configf.getfloat(pressure_title, "P")
            Ptensor = configf.get(pressure_title, "Ptensor")
            self.Ptensor = np.array([float(x) for x in Ptensor.split()])
            self.energy = configf.getfloat(energy_title, "E")

    def import_structural_data(self, path, path2, title_boo="BOO", title_z="Z"):
        """
        import average contact number and bond orientational order parameters
        """
        if os.path.isfile(path):
            configf = configparser.ConfigParser()
            configf.read(path)
            if self.bdim == 3:
                Q4, Q6 = configf.getfloat(title_boo, "Q4"), configf.getfloat(title_boo, "Q6")
                Q8, Q10 = configf.getfloat(title_boo, "Q8"), configf.getfloat(title_boo, "Q10")
                Q12 = configf.getfloat(title_boo, "Q12")
                self.boo = Bunch(Q4=Q4, Q6=Q6, Q8=Q8, Q10=Q10, Q12=Q12)
            elif self.bdim == 2:
                Q6 = configf.getfloat(title_boo, "Q6")
                self.boo = Bunch(Q6=Q6)
            z = configf.getfloat(title_z, "Z")
            self.Z = z
            self.boolist, self.Zlist = np.loadtxt(path2, unpack=True)

    def import_invsym_data(self, path, title_invsym="INVERSION_SYMMETRY"):
        """
        import average contact number and bond orientational order parameters
        """
        if os.path.isfile(path):
            configf = configparser.ConfigParser()
            configf.read(path)
            self.invsym = configf.getfloat(title_invsym, "inversion_symmetry")
