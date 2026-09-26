# basinvolume

[![Tests](https://github.com/martiniani-lab/basinvolume/actions/workflows/test.yml/badge.svg?branch=Lees_Edwards_BC_3)](https://github.com/martiniani-lab/basinvolume/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/martiniani-lab/basinvolume/branch/Lees_Edwards_BC_3/graph/badge.svg)](https://codecov.io/gh/martiniani-lab/basinvolume)

A library to calculate basin volumes for the basins of attraction of the energy minima of jammed packings. 

## Installation

We recommend creating a conda environment to work with the package. basinvolume builds on
[pele](https://github.com/martiniani-lab/pele), [mcpele](https://github.com/martiniani-lab/mcpele)
and [PyCG_DESCENT](https://github.com/martiniani-lab/PyCG_DESCENT), which are installed first:

```bash
conda create -n basinvolume -c conda-forge python compilers sundials eigen blas-devel mpi4py
conda activate basinvolume
pip install git+https://github.com/martiniani-lab/pele
pip install git+https://github.com/martiniani-lab/mcpele
pip install git+https://github.com/martiniani-lab/PyCG_DESCENT
pip install git+https://github.com/martiniani-lab/basinvolume
```

If the machine already has gcc, g++ and gfortran (e.g. `sudo apt install gcc g++ gfortran`),
leave out `compilers` for a much smaller environment.

### Development

From a clone, in the same environment:

```bash
pip install .                  # install, or
python setup.py build_ext -i   # build in place; then put the clone on PYTHONPATH
```

Build options are environment variables: `BV_BUILD_TYPE=Debug`, `BV_JOBS=8`, and
`BV_NATIVE=1` (add `-march=native`; off by default).

## Tests

```bash
pip install pytest
OMP_NUM_THREADS=1 pytest --pyargs basinvolume
```



## Contributors

This repository was migrated from Bitbucket, so GitHub's contributors does not reflect the actual authorship. The contributors are, 

- Julian Schrenk
- Stefano Martiniani
- Johannes Gasteiger, né Klicpera
- Praharsh Suryadevara
- Mathias Casiulis
- Philipp Hoellmer
- Jacob Stevenson