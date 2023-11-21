import numpy as np
import os
from pele.utils.cell_scale import get_box_length, get_ncellsx_scale
from pele.potentials import InversePower, PyInversePower
from pele.optimize._quench import quench_cvode_opt
import multiprocessing as mp
import yaml

BINARY_SOFT_SPHERE_DEFAULTS = {
    "ndim": 2,
    "phi": 0.9,
    "seed": 0,
    "n_part": 16,
    "r1": 1.0,
    "r2": 1.4,
    "rstd1": 0.05,
    "rstd2": 0.05 * 1.4,
    "use_cell_lists": 0,
    "power": 2.5,
    "eps": 1,
}


def setup_without_radii_bl(parameters, radii, box_length):
    """Sets up the inverse power potential.
        Uses a python based implementation that accounts
        for multiple intersections with periodic boundary conditions.
        when the box is small.

    Parameters
    ----------
    SysParams : Enum
        Parameters for the system we wish to study
    radii : np.ndarray
    radii of particles in the system
    box_length : float
    length of the box
    """
    if len(radii) != parameters["n_part"]:
        raise ValueError("Number of radii must match number of particles")
    box_vec = np.array([box_length] * int(parameters["ndim"]))

    # setup defaults
    if "power" not in parameters:
        parameters["power"] = 2.5
    if "eps" not in parameters:
        parameters["eps"] = 1.0
    if "use_cell_lists" not in parameters:
        # use cell lists if the box is large enough
        if len(radii) > 100:  # set from scaling in Johannes thesis
            parameters["use_cell_lists"] = True
        else:
            parameters["use_cell_lists"] = False
    if "ndim" not in parameters:
        parameters["ndim"] = 2
    if "non_additivity" not in parameters:
        parameters["non_additivity"] = 0.0

    if np.amin(box_vec) < 4 * np.amax(radii):
        if "non_additivity" in parameters:
            raise ValueError("non_additivity not implemented for python potential")
        potential = PyInversePower(
            parameters["power"],
            parameters["eps"],
            use_cell_lists=parameters["use_cell_lists"],
            ndim=parameters["ndim"],
            radii=radii * 1.0,
            boxvec=box_vec,
        )
    else:
        potential = InversePower(
            parameters["power"],
            parameters["eps"],
            use_cell_lists=parameters["use_cell_lists"],
            ndim=parameters["ndim"],
            radii=radii * 1.0,
            boxvec=box_vec,
            non_additivity=parameters["non_additivity"],
            ncellx_scale=get_ncellsx_scale(radii, box_vec),
        )
    return potential


def setup_bidisperse(parameters, seed=0):
    """Generates a bidisperse system potential and also returns box length and radii

    Parameters
    ----------
    parameters : dict
        parameters for the system.
        should have the keys phi, r1, r2, rstd1, rstd2 in addition to
        the potential parameters to setup the radii

    seed : int
        seed for the random number generator

    Returns
    -------
    dict: With keys potential, radii, box_length
    """
    if "radii" in parameters:
        return {
            "potential": setup_without_radii_bl(
                parameters, parameters["radii"], parameters["box_length"]
            ),
            "radii": parameters["radii"],
            "box_length": parameters["box_length"],
        }

    r1 = parameters["r1"]
    r2 = parameters["r2"]
    rstd1 = parameters["rstd1"]
    rstd2 = parameters["rstd2"]
    ndim = parameters["ndim"]
    phi = parameters["phi"]
    n_part = parameters["n_part"]
    rng = np.random.default_rng(seed)
    n_part_by_2 = n_part // 2
    hs_radii = np.array(
        list(r1 + rstd1 * rng.normal(size=n_part_by_2))
        + list(r2 + rstd2 * rng.normal(size=n_part - n_part_by_2))
    )
    box_length = get_box_length(hs_radii, ndim, phi)

    potential = setup_without_radii_bl(parameters, hs_radii, box_length)

    return {
        "potential": potential,
        "radii": hs_radii,
        "box_length": box_length,
    }


def quench_minima(initial_condition, parameters, quench_params):
    potential = setup_bidisperse(parameters)["potential"]
    print("potential")
    print("initial_condition")
    print("quench_params")
    result = quench_cvode_opt(potential, initial_condition.flatten(), **quench_params)
    return result["coords"]


class SoftSphereMinimaEnsemble:
    """Create a set of minima for a given packing fraction."""

    def __init__(
        self, parameters: dict, save_path: str, save_folder_name="packings"
    ) -> None:
        """Create a set of minima for a given packing fraction."""
        if parameters["n_part"] > 128 and parameters["use_cell_lists"] == 0:
            print("WARNING: Not using cell lists for large system")
        self.save_path = os.path.join(save_path, save_folder_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.parameters = parameters
        result_dict = setup_bidisperse(parameters, parameters["seed"])
        self.potential = result_dict["potential"]
        self.parameters["radii"] = result_dict["radii"]
        self.parameters["box_length"] = result_dict["box_length"]

    def generate_initial_conditions(self, n_ensemble: int):
        """Generate initial conditions for points and save them to disk."""
        array_shape = (
            n_ensemble,
            self.parameters["n_part"],
            self.parameters["ndim"],
        )

        # conditions are uniform random in the box
        self.initial_conditions = np.random.uniform(
            low=0.0,
            high=self.parameters["box_length"],
            size=array_shape,
        )

    def generate_minima(self, processes=1, quench_params={"tol": 1e-10}):
        """Generate the minimum(attractor) for each initial condition."""

        pool = mp.Pool(processes=processes)

        results = pool.starmap(
            quench_minima,
            (
                (ic, self.parameters.copy(), quench_params.copy())
                for ic in self.initial_conditions
            ),
        )

        pool.close()
        pool.join()

        results = np.array(results)

        results = results.reshape(self.initial_conditions.shape)
        self.minima = results

    def save(self):
        """Save the results to disk."""
        with open(os.path.join(self.save_path, "parameters.yaml"), "w") as par_file:
            yaml.dump(self.parameters, par_file)
        for i, minimum in enumerate(self.minima):
            np.savetxt(
                os.path.join(self.save_path, f"minimum_{i}.csv"),
                minimum,
                delimiter=",",
            )
        for i, initial_condition in enumerate(self.initial_conditions):
            np.savetxt(
                os.path.join(
                    self.save_path,
                    f"initial_condition_{i}.csv",
                ),
                initial_condition,
                delimiter=",",
            )


def generate_soft_sphere_ensemble(
    parameters: dict = BINARY_SOFT_SPHERE_DEFAULTS,
    save_path: str = ".",
    n_ensemble: int = 10,
    save_folder_name: str = "packings",
    processes: int = 1,
    quench_params: dict = {"tol": 1e-10},
) -> None:
    """Generate minima."""
    sf = SoftSphereMinimaEnsemble(parameters, save_path, save_folder_name)
    sf.generate_initial_conditions(n_ensemble)
    sf.generate_minima(processes, quench_params)
    sf.save()
    return sf


if __name__ == "__main__":
    # end to end test
    generate_soft_sphere_ensemble()
