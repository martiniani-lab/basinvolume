import numpy as np

def get_means_cov(gauss_path):
    print("reading means, cov from the following gauss path")
    print("gauss_path", gauss_path)
    mean = []
    cov = []
    f = open(gauss_path, "r")
    for line in f.readlines():
        if line.startswith("["):
            m = map(float, (line.split("\t")[0].replace("[", "")).replace("]", "").split())
            c = map(float, (line.split("\t")[1].replace("[", "")).replace("]", "").split())
            mean.append(m)
            cov.append(c)
    f.close()
    return np.asarray(mean), np.asarray(cov)
