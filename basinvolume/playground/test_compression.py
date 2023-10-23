from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
import os
import hickle as hkl
import pandas as pd
import pickle as pkl
import time

def humansize(nbytes, suffixes=['B', 'KB', 'MB', 'GB', 'TB', 'PB']):
    if nbytes == 0: return '0 B'
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def timeit(func):
    def func_wrapper(*args, **kawrgs):
        nruns = 10
        start = time.clock()
        for _ in range(nruns):
            func(*args, **kawrgs)
        end = time.clock()
        print("time: {}s".format((end-start)/nruns))
    return func_wrapper

@timeit
def pandas_hdf5_index(array, path, key="test", **kwargs):
    nind, ncol = array.shape
    ind = [i for i in range(nind)]
    col = [i for i in range(ncol)]
    df = pd.DataFrame(array, index=ind, columns=col)
    df.to_hdf(path, key, **kwargs)

@timeit
def pandas_hdf5(array, path, key="test", **kwargs):
    df = pd.DataFrame(array)
    df.to_hdf(path, key, **kwargs)

@timeit
def hickle(array, path, **kwargs):
    hkl.dump(array, path, **kwargs)

@timeit
def pickle(array, path, **kwargs):
    pkl.dump(array, open(path, 'wb'), **kwargs)

@timeit
def numpy_tofile(array, path, **kwargs):
    array.tofile(open(path, 'wb'), **kwargs)

@timeit
def numpy_save(array, path, **kwargs):
    np.save(open(path, 'wb'), array, **kwargs)

@timeit
def numpy_savetxt(array, path, **kwargs):
    np.savetxt(open(path, 'wb'), array, **kwargs)


if __name__ == "__main__":
    array = np.random.rand(int(1e4),10)
    array = np.asarray(array, dtype='double')
    print("array shape: {} \n".format(array.shape))

    fname = "pandas_index.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5_index(array, fpath)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pandas_index_blosc.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5_index(array, fpath, complevel=9, complib='blosc')
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pandas_index_zlib.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5_index(array, fpath, complevel=9, complib='zlib')
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pandas_index_bzip2.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5_index(array, fpath, complevel=9, complib='bzip2')
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pandas_index_lzo.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5_index(array, fpath, complevel=9, complib='lzo')
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pandas.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5(array, fpath)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pandas_blosc.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5(array, fpath, complevel=9, complib='blosc')
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pandas_zlib.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5(array, fpath, complevel=9, complib='zlib')
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    # fname = "pandas_bzip2.h5"
    # fpath = os.path.abspath(fname)
    # pandas_hdf5(array, fpath, complevel=9, complib='bzip')
    # nbytes = os.path.getsize(fpath)
    # print "{} size: {}".format(fname, humansize(nbytes))
    # os.remove(fpath)

    fname = "pandas_lzo.h5"
    fpath = os.path.abspath(fname)
    pandas_hdf5(array, fpath, complevel=9, complib='lzo')
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "hickle.hl"
    fpath = os.path.abspath(fname)
    hickle(array, fpath)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "hickle_lzf.hl"
    fpath = os.path.abspath(fname)
    hickle(array, fpath, compression='lzf', shuffle=True, chunks=True)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "hickle_gzip.hl"
    fpath = os.path.abspath(fname)
    hickle(array, fpath, compression='gzip', shuffle=True, compression_opts=9, chunks=True)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pickle.pickle"
    fpath = os.path.abspath(fname)
    pickle(array, fpath)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "pickle_highest.pickle"
    fpath = os.path.abspath(fname)
    pickle(array, fpath, protocol=-1)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "numpy_tofile.npy"
    fpath = os.path.abspath(fname)
    numpy_tofile(array, fpath)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "numpy_save.npy"
    fpath = os.path.abspath(fname)
    numpy_save(array, fpath)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "numpy_save_nopickle.npy"
    fpath = os.path.abspath(fname)
    numpy_save(array, fpath, allow_pickle=False)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

    fname = "numpy_savetxt.txt"
    fpath = os.path.abspath(fname)
    numpy_savetxt(array, fpath)
    nbytes = os.path.getsize(fpath)
    print("{} size: {} \n".format(fname, humansize(nbytes)))
    os.remove(fpath)

