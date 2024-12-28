
import numpy as np
import matplotlib.pyplot as plt
from basinerror.utils.params import setup_inverse_power
import pandas as pd
import os
from basinerror.utils.params import INVERSE_POWER_BINARY
from basinerror.minima_finders.pele_quenches import quench_LBFGS

data_loc = "/home/praharsh/pack_8_checker/explore_bv_jammed_packing2/0"
data = pd.read_hdf(data_loc + "/trajectory.hd5", "data")


data



def read_all_data():
    data_folder = "/home/praharsh/pack_8_checker/explore_bv_jammed_packing2/"
    # find integer folders
    integer_folders = [f for f in os.listdir(data_folder) if f.isdigit()]
    integer_folder_paths = [os.path.join(data_folder, f) for f in integer_folders]
    # check that they are folders
    integer_folder_paths = [f for f in integer_folder_paths if os.path.isdir(f)]
    load_all_samples = []
    for folder in integer_folder_paths:
        print(folder)
        data = pd.read_hdf(folder + "/trajectory.hd5", "data")
        print(data.shape)
        load_all_samples.append(data)

    # pandas dataframe
    return pd.concat(load_all_samples, ignore_index=True)


data = read_all_data()


data




 [markdown]
#


import os

jammed_packing_dir = "/home/praharsh/pack_8_checker/jammed_packings/"
xydr_file = os.path.join(jammed_packing_dir, "jammed_packing2.xydr")
xydr = np.loadtxt(xydr_file, usecols=(0, 1, 2))


minimum_coordinates = xydr[:, :2].flatten()
radii = xydr[:, 2].flatten() / 2.0


minimum_coordinates


minimum_coordinates


np.amin(radii) * 0.25




params = INVERSE_POWER_BINARY

INVERSE_POWER_BINARY["n_part"] = 8
box_length = 6.5623088820508721
potential = setup_inverse_power(params, radii, box_length)


quench_LBFGS(potential, minimum_coordinates, tol=1e-10, maxstep=np.amin(radii) * 0.25)


# convert everything but the last column to a numpy array
samples = data.iloc[:, :-1].values



def quench(x):
    return quench_LBFGS(potential, x, tol=1e-10, maxstep=np.amin(radii) * 0.25, maxErise=0)



new_minimum = quench_LBFGS(
    potential, samples[5], tol=1e-10, maxstep=np.amin(radii) * 0.25, maxErise=0
)["coords"]


new_minimum


from basinerror.analysis.check_same_structure import CheckSameStructurePeriodic


csm = CheckSameStructurePeriodic(potential=potential, radii=radii, ndim=2, box_length=box_length)


csm.check_same_structure(new_minimum, minimum_coordinates, 1e-2, csm.find_rattlers(new_minimum)[1])


new_minimum


csm.find_rattlers(new_minimum)





from basinerror.minima_finders.pele_quenches import quench_cvode_opt, quench_modified_fire


def quench_cvode(x):
    return quench_cvode_opt(potential, x, tol=1e-10, rtol=1e-8, atol=1e-8)






true_list = []

not_same_list_coords = []

for sample in samples:
    sample = sample.flatten()
    # quenched_sample = quench_LBFGS(potential, sample, tol=1e-10, maxstep = np.amin(radii) * 0.25, maxErise=0)["coords"]

    quenched_sample = quench_cvode(sample)["coords"]
    same = csm.check_same_structure(
        quenched_sample, minimum_coordinates, 1e-2, csm.find_rattlers(quenched_sample)[1]
    )
    if not same:
        not_same_list_coords.append(quenched_sample)
    true_list.append(
        csm.check_same_structure(
            quenched_sample, minimum_coordinates, 1e-2, csm.find_rattlers(minimum_coordinates)[1]
        )
    )


not_same_list_coords


not_same_1 = not_same_list_coords[0]


aligned_structures = csm._align_structures(not_same_1, minimum_coordinates, 0)


not_same_1


not_same_1 - aligned_structures


np.set_printoptions(precision=16, suppress=False)


trial_coords = np.array(
    [
        5.803669189512445,
        -3.1609968789880059,
        5.4094118313020791,
        -11.366789832309168,
        -3.5969985782913199,
        1.973069422918885,
        2.5771582500183325,
        -3.5865023239004428,
        1.4969461545367875,
        -3.4355138079279968,
        3.1544076441152704,
        0.17522094129574381,
        0.92918028079423975,
        3.316584187737869,
        5.748157984360442,
        -2.8762394169308507,
    ]
)


minimized = quench(trial_coords)


minimized


minimized["coords"]


np.average(true_list)


coords_2 = np.array(
    [
        5.5829482911497275,
        -3.5962210445670983,
        4.8695974013063932,
        -12.189866134267243,
        -3.6374170931445895,
        2.1769736873430601,
        3.8198096215024679,
        -2.634997752439117,
        1.4579048948277062,
        -2.6266548210000038,
        3.0130310181264179,
        -0.31743641552142576,
        0.63794318519152637,
        1.3608447874735059,
        5.7781154373886192,
        -1.1338100151256656,
    ]
)





csm.find_rattlers(coords_2)


csm.find_rattlers(minimum_coordinates)


csm.check_same_structure(
    coords_2, minimum_coordinates, 1e-2, rattlers=csm.find_rattlers(minimum_coordinates)[1]
)


csm.check_same_structure(
    minimized["coords"],
    minimum_coordinates,
    1e-2,
    rattlers=csm.find_rattlers(minimum_coordinates)[1],
)


minimized["coords"] - coords_2





coords = np.array(
    [
        1.8978809272453303,
        -3.0218783184402005,
        1.1507322356634324,
        1.8354570588387784,
        -0.60406870834585602,
        2.7267467954785767,
        0.18249504275119358,
        -2.0850730739820209,
        -2.2136660331953371,
        -2.0847747054117347,
        -0.65853969823234926,
        0.22444240186255468,
        -3.0336286896041509,
        1.9027284038545309,
        2.153553881375176,
        -0.63579952033565756,
    ]
)


res_coords = quench_cvode(coords)


res_coords["coords"]


final_coords = np.array(
    [
        1.9208251943048162,
        -3.0513591704369234,
        1.1866746261504817,
        1.801897164020053,
        -0.62871624332499321,
        2.7393006282396084,
        0.15765418903310971,
        -2.0901829113787826,
        -2.2042635026202482,
        -2.0818466522414125,
        -0.64915084714387161,
        0.22734835603170025,
        -3.0242152729101415,
        1.9056644252546424,
        2.1159508141682957,
        -0.58897279762405474,
    ]
)


csm.check_same_structure(
    final_coords, res_coords["coords"], 1e-2, rattlers=csm.find_rattlers(minimum_coordinates)[1]
)


csm.check_same_structure(
    final_coords, minimum_coordinates, 1e-2, rattlers=csm.find_rattlers(minimum_coordinates)[1]
)



rms_vs_inside = []
rms_vs_inside_fire = []
for sample in samples:
    sample = sample.flatten()
    rmsd = csm.get_rmsd_wo_rattlers(
        sample, minimum_coordinates, 1e-2, csm.find_rattlers(minimum_coordinates)[1]
    )
    quenched_sample = quench_LBFGS(
        potential, sample, tol=1e-10, maxstep=np.amin(radii) * 0.25, maxErise=0
    )["coords"]
    quenched_sample_fire = quench_modified_fire(
        potential, sample, tol=1e-10, maxstep=np.amin(radii) * 0.25
    )["coords"]
    same = csm.check_same_structure(
        quenched_sample, minimum_coordinates, 1e-2, csm.find_rattlers(quenched_sample)[1]
    )
    same_fire = csm.check_same_structure(
        quenched_sample_fire, minimum_coordinates, 1e-2, csm.find_rattlers(quenched_sample_fire)[1]
    )
    rms_vs_inside_fire.append((rmsd, same_fire))
    rms_vs_inside.append((rmsd, same))


# get histogram of rmsd
rms_vs_inside = np.array(rms_vs_inside)
rmsd = rms_vs_inside[:, 0]
same = rms_vs_inside[:, 1].astype(bool)

rmsd_fire = np.array(rms_vs_inside_fire)[:, 0]
same_fire = np.array(rms_vs_inside_fire)[:, 1].astype(bool)

from statsmodels.stats.proportion import proportion_confint

splits = np.linspace(0, 8, 9)

bins = [[] for i in range(len(splits) - 1)]
bins_fire = [[] for i in range(len(splits) - 1)]


for data_point in rms_vs_inside:
    bin_index = np.digitize(data_point[0], splits) - 1
    if bin_index > len(bins) - 1:
        continue
    bins[bin_index].append(data_point[1])

for data_point in rms_vs_inside_fire:
    bin_index = np.digitize(data_point[0], splits) - 1
    if bin_index > len(bins_fire) - 1:
        continue
    bins_fire[bin_index].append(data_point[1])

# calculate the fraction of same structures in each bin
n_counts = np.array([len(b) for b in bins])
n_same = np.array([np.sum(b) for b in bins])

n_counts_fire = np.array([len(b) for b in bins_fire])
n_same_fire = np.array([np.sum(b) for b in bins_fire])


low_error, high_error = proportion_confint(n_same, n_counts, method="beta")
low_error_fire, high_error_fire = proportion_confint(n_same_fire, n_counts_fire, method="beta")
x_vals = (splits[1:] + splits[:-1]) / 2
y_vals = n_same / n_counts
y_err = np.array([y_vals - low_error, high_error - y_vals])

y_fire = n_same_fire / n_counts_fire
y_err_fire = np.array([y_fire - low_error_fire, high_error_fire - y_fire])


plt.errorbar(
    x_vals, y_vals, yerr=y_err, fmt="o", markersize=8, capsize=6, elinewidth=1.5, label="LBFGS"
)
plt.errorbar(
    x_vals, y_fire, yerr=y_err_fire, fmt="o", markersize=8, capsize=6, elinewidth=1.5, label="FIRE"
)
plt.axvline(x=3.29, color='grey', linestyle='--')
# set limits
plt.ylim(0, 1)
plt.xlim(0, 8)
plt.xticks(fontsize=14)  # Adjust font size of x-axis tick labels
plt.yticks(fontsize=14)  # Adjust font size of y-axis tick labels
# plt.legend(fontsize=14)  # Adjust font size of legend








print(splits)


splits = np.linspace(0, 7, 7)


splits


location = "/home/praharsh/pack_8_checker/explore_bv_jammed_packing2/analysis/dos.csv"


dos_data = np.loadtxt(location, delimiter=",", usecols=(0, 2))


r, dos = dos_data[:, 0], dos_data[:, 1]


# do kde for dos
from sklearn.neighbors import KernelDensity


kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
kde.fit(r[:, None], sample_weight=dos)
prediction = kde.score_samples(r[:, None])

plt.plot(r, np.exp(prediction), linewidth=2, alpha=0.7)
plt.xticks(fontsize=14)  # Adjust font size of x-axis tick labels
plt.yticks(fontsize=14)  # Adjust font size of y-axis tick labels


# find the maximum of the kde
max_index = np.argmax(np.exp(prediction))
max_r = r[max_index]


plt.plot(r, dos, label="DOS", linewidth=2)


max_r


