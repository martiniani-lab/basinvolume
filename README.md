# basinvolume

[![Tests](https://github.com/martiniani-lab/basinvolume/actions/workflows/test.yml/badge.svg?branch=Lees_Edwards_BC_3)](https://github.com/martiniani-lab/basinvolume/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/martiniani-lab/basinvolume/branch/Lees_Edwards_BC_3/graph/badge.svg)](https://codecov.io/gh/martiniani-lab/basinvolume)

A library to calculate basin volumes for the basins of attraction of the energy minima of jammed packings. 

## Installation

basinvolume builds against installed [pele](https://github.com/martiniani-lab/pele),
[mcpele](https://github.com/martiniani-lab/mcpele) and
[PyCG_DESCENT](https://github.com/martiniani-lab/PyCG_DESCENT) (none are on PyPI).
All build dependencies come from conda-forge:

```bash
conda create -n basinvolume -c conda-forge python=3.12 compilers cmake ninja meson \
    "sundials>=6.2" eigen blas-devel llvm-openmp numpy "cython>=3" setuptools pip \
    scipy networkx matplotlib-base "sqlalchemy>=1.4,<2" munkres pyro4 future \
    toml pymbar pandas joblib pyyaml mpi4py pytest
conda activate basinvolume
pip install --no-build-isolation git+https://github.com/martiniani-lab/pele
pip install --no-build-isolation git+https://github.com/martiniani-lab/mcpele
pip install --no-build-isolation git+https://github.com/martiniani-lab/PyCG_DESCENT
pip install --no-build-isolation .   # or: pip install --no-build-isolation git+https://github.com/martiniani-lab/basinvolume
```

Build options are environment variables, e.g. `BV_BUILD_TYPE=Debug`, `BV_JOBS=8`, and
`BV_NATIVE=1` (add `-march=native`; off by default).

On macOS, unless `CC`/`CXX` are set, the build (like pele's) uses the newest Homebrew
`gcc-N` (`brew install gcc`); Apple clang has no OpenMP support.

For development, the in-place build still works: `python setup.py build_ext -i`.

## Tests

```bash
pytest --pyargs basinvolume
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