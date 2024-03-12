#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from basinerror.utils.params import setup_inverse_power
import pandas as pd


# In[2]:


def read_all_data():
    data_folder = "/home/praharsh/pack_8_check/explore_bv_jammed_packing0/"
    # find integer folders
    integer_folders = [f for f in os.listdir(data_folder) if f.isdigit()]
    integer_folder_paths = [os.path.join(data_folder, f) for f in integer_folders]
    # check that they are folders
    integer_folder_paths = [f for f in integer_folder_paths if os.path.isdir(f)]
    load_all_samples = []
    for folder in integer_folder_paths:
        print(folder)
        data = pd.read_hdf(folder + "/trajectory.hd5", "data")
        load_all_samples.append(data)

    # pandas dataframe
    return pd.concat(load_all_samples, ignore_index=True)


data_loc = "/home/praharsh/pack_8_check/explore_bv_jammed_packing0/0"
data = pd.read_hdf(data_loc + "/trajectory.hd5", "data")

data)read_all_data()
# In[3]:


data


#

# In[4]:


import os

jammed_packing_dir = "/home/praharsh/pack_8_check/jammed_packings/"
xydr_file = os.path.join(jammed_packing_dir, "jammed_packing0.xydr")
xydr = np.loadtxt(xydr_file, usecols=(0, 1, 2))


# In[5]:


minimum_coordinates = xydr[:, :2].flatten()
radii = xydr[:, 2].flatten() / 2.0


# In[6]:


minimum_coordinates


# In[7]:


minimum_coordinates


# In[8]:


np.amin(radii) * 0.25


# In[9]:


from basinerror.utils.params import INVERSE_POWER_BINARY
from basinerror.minima_finders.pele_quenches import quench_LBFGS

params = INVERSE_POWER_BINARY

INVERSE_POWER_BINARY["n_part"] = 8
box_length = 6.7131862671232589
potential = setup_inverse_power(params, radii, box_length)


# In[10]:


quench_LBFGS(potential, minimum_coordinates, tol=1e-10, maxstep=np.amin(radii) * 0.25)


# In[11]:


# convert everything but the last column to a numpy array
samples = data.iloc[:, :-1].values


# In[12]:


def quench(x):
    return quench_LBFGS(potential, x, tol=1e-10, maxstep=np.amin(radii) * 0.25, maxErise=0)


# In[13]:


new_minimum = quench_LBFGS(
    potential, samples[5], tol=1e-10, maxstep=np.amin(radii) * 0.25, maxErise=0
)["coords"]


# In[14]:


new_minimum


# In[15]:


from basinerror.analysis.check_same_structure import CheckSameStructurePeriodic


# In[16]:


csm = CheckSameStructurePeriodic(potential=potential, radii=radii, ndim=2, box_length=box_length)


# In[17]:


csm.check_same_structure(new_minimum, minimum_coordinates, 1e-2, csm.find_rattlers(new_minimum)[1])


# In[18]:


new_minimum


# In[19]:


csm.find_rattlers(new_minimum)


# In[ ]:


# In[22]:


from basinerror.minima_finders.pele_quenches import quench_cvode_opt


def quench_cvode(x):
    return quench_cvode_opt(potential, x, tol=1e-10, rtol=1e-8, atol=1e-8)


# In[20]:


potential.getEnergy(minimum_coordinates)


# In[23]:


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


# In[66]:


not_same_list_coords


# In[23]:


not_same_1 = not_same_list_coords[0]


# In[24]:


aligned_structures = csm._align_structures(not_same_1, minimum_coordinates, 0)


# In[59]:


not_same_1


# In[60]:


not_same_1 - aligned_structures


# In[61]:


np.set_printoptions(precision=16, suppress=False)


# In[28]:


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


# In[29]:


minimized = quench(trial_coords)


# In[30]:


minimized


# In[31]:


minimized["coords"]


# In[25]:


np.average(true_list)


# In[36]:


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


# In[35]:


csm.find_rattlers(coords_2)


# In[36]:


csm.find_rattlers(minimum_coordinates)


# In[37]:


csm.check_same_structure(
    coords_2, minimum_coordinates, 1e-2, rattlers=csm.find_rattlers(minimum_coordinates)[1]
)


# In[38]:


csm.check_same_structure(
    minimized["coords"],
    minimum_coordinates,
    1e-2,
    rattlers=csm.find_rattlers(minimum_coordinates)[1],
)


# In[40]:


minimized["coords"] - coords_2


# In[62]:


# In[17]:


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


# In[19]:


res_coords = quench_cvode(coords)


# In[21]:


res_coords["coords"]


# In[22]:


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


# In[35]:


csm.check_same_structure(
    final_coords, res_coords["coords"], 1e-2, rattlers=csm.find_rattlers(minimum_coordinates)[1]
)


# In[ ]:


csm.check_same_structure(
    final_coords, minimum_coordinates, 1e-2, rattlers=csm.find_rattlers(minimum_coordinates)[1]
)
