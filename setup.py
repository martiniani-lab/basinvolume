import glob
import os
import sys
import subprocess

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy.distutils.misc_util import has_cxx_sources
import numpy as np
import pele
import mcpele
import PyCG_DESCENT

## Numpy header files
numpy_lib = os.path.split(np.__file__)[0]
numpy_include = os.path.join(numpy_lib, 'core/include')

##find mcpele path
pypath = os.environ['PYTHONPATH'].split(os.pathsep)
mcpele_found = False
pele_found = False

try:
    pelepath = os.path.dirname(pele.__file__)[:-len("/pele")]
except:
    sys.stderr.write("WARNING: could't find path to pele\n")
    sys.exit()
try:
    mcpelepath = os.path.dirname(mcpele.__file__)[:-len("/mcpele")]
    print mcpelepath
except:
    sys.stderr.write("WARNING: could't find path to mcpele\n")
    sys.exit()

try:
    py_cgdescentpath = os.path.dirname(PyCG_DESCENT.__file__)[:-len("/PyCG_DESCENT")]
    print py_cgdescentpath
except:
    sys.stderr.write("WARNING: could't find path to PyCG_DESCENT\n")
    sys.exit()

# print git version to version.py file, copied from pele
#
# Make the git revision visible.  Most of this is copied from scipy
#
# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def write_version_py(filename='basinvolume/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SCIPY SETUP.PY
git_revision = '%(git_revision)s'
"""
    GIT_REVISION = git_version()

    a = open(filename, 'w')
    try:
        a.write(cnt % dict(git_revision=GIT_REVISION))
    finally:
        a.close()
write_version_py()

# need to pass cython the include directory so it can find the .pyx files
cython_flags=["-I"] + [os.path.abspath(pelepath)+"/pele/potentials"] + ["-v"]

def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                          os.path.join(cwd, 'cythonize.py'),
                          'basinvolume'] + cython_flags,
                         cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")

generate_cython()

#
# compile fortran extension modules
#

class ModuleList:
    def __init__(self, **kwargs):
        self.module_list = []
        self.kwargs = kwargs
    def add_module(self, filename):
        modname = filename.replace("/", ".")
        modname, ext = os.path.splitext(modname)
        self.module_list.append(Extension(modname, [filename], **self.kwargs))

setup(name='basinvolume',
      version='0.1',
      description="Python implementation of the basin volume method",
      url='https://bitbucket.org/smcantab/basinvolume',
      packages=["basinvolume",
                "basinvolume.monte_carlo",
                "basinvolume.utils",
                "basinvolume.spheres",
                "basinvolume.gui",
                # add the test directories
                "basinvolume.monte_carlo.tests",
                "basinvolume.spheres.tests",
                ],
        )

#
# build the c++ files
#

include_sources_bv = ["source/basinvolume/" + f for f in os.listdir("source/basinvolume")
                   if f.endswith(".cpp")]

include_sources_mcpele = [mcpelepath+"/source/" + f for f in os.listdir(mcpelepath+"/source")
                   if f.endswith(".cpp")]

include_sources_pele = [pelepath+"/source/" + f for f in os.listdir(pelepath+"/source")
                   if f.endswith(".cpp")]

include_sources_py_cgdescent = [py_cgdescentpath+"/source/" + f for f in os.listdir(py_cgdescentpath+"/source")
                   if f.endswith(".cpp") or f.endswith(".c")]

include_sources_py_cgdescent += [py_cgdescentpath+"/source/CG_DESCENT/" + f for f in os.listdir(py_cgdescentpath+"/source/CG_DESCENT/")
                   if f.endswith(".cpp") or f.endswith(".c")]

include_dirs = [numpy_include, "source"]

depends_bv = [os.path.join("source/basinvolume", f) for f in os.listdir("source/basinvolume/")
              if f.endswith(".cpp") or f.endswith(".h") or f.endswith(".hpp")]

depends_mcpele = [os.path.join(mcpelepath+"/source/mcpele", f) for f in os.listdir(mcpelepath+"/source/mcpele")
                  if f.endswith(".cpp") or f.endswith(".h") or f.endswith(".hpp")]

depends_pele = [os.path.join(pelepath+"/source/pele", f) for f in os.listdir(pelepath+"/source/pele")
                if f.endswith(".cpp") or f.endswith(".h") or f.endswith(".hpp")]

depends_py_cgdescent = [os.path.join(py_cgdescentpath+"/source/CG_DESCENT", f) for f in os.listdir(py_cgdescentpath+"/source/CG_DESCENT/")
           if f.endswith(".cpp") or f.endswith(".c") or f.endswith(".h") or f.endswith(".hpp")]
depends_py_cgdescent += [os.path.join(py_cgdescentpath+"/source/PyCG_DESCENT", f) for f in os.listdir(py_cgdescentpath+"/source/PyCG_DESCENT/")
           if f.endswith(".cpp") or f.endswith(".c") or f.endswith(".h") or f.endswith(".hpp")]

# note: on my computer (ubuntu 12.04 gcc version 4.6.3), when compiled with the
# flag -march=native I run into problems.  Everything seems to run ok, but when
# I run it through valgrind, valgrind complains about an unrecognized
# instruction.  I don't have a clue what is causing this, but it's probably
# better to be on the safe side and not use -march=native
#extra_compile_args = ['-I/home/sm958/Work/pele/source','-std=c++0x',"-Wall", "-Wextra", "-O3", '-funroll-loops', "-fopenmp", "-mavx"]
# uncomment the next line to add extra optimization options

include_pele_source = '-I'+ pelepath + '/source'
include_mcpele_source = '-I'+ mcpelepath + '/source'
include_py_cgdescent_source = '-I'+ py_cgdescentpath + '/source'

extra_compile_args = [include_pele_source,include_mcpele_source,include_py_cgdescent_source,'-std=c++0x',"-Wall", '-Wextra','-pedantic','-O3', "-fopenmp", "-mavx"]

# note: to compile with debug on and to override extra_compile_args use, e.g.
# OPT="-g -O2 -march=native" python setup.py ...

include_sources_all = include_sources_bv + include_sources_py_cgdescent + include_sources_mcpele + include_sources_pele
#recompile if any of depends has been modified
depends_all = depends_bv + depends_py_cgdescent + depends_mcpele + depends_pele

cxx_modules = [
    Extension("basinvolume.monte_carlo._conf_test_cpp",
              ["basinvolume/monte_carlo/_conf_test_cpp.cxx"] + include_sources_all,
              include_dirs=include_dirs,
              extra_compile_args=extra_compile_args,
              language="c++", depends=depends_all,
              ),
    Extension("basinvolume.monte_carlo._action_cpp",
              ["basinvolume/monte_carlo/_action_cpp.cxx"] + include_sources_all,
              include_dirs=include_dirs,
              extra_compile_args=extra_compile_args,
              language="c++", depends=depends_all,
              ),
    Extension("basinvolume.monte_carlo._takestep_cpp",
              ["basinvolume/monte_carlo/_takestep_cpp.cxx"] + include_sources_all,
              include_dirs=include_dirs,
              extra_compile_args=extra_compile_args,
              language="c++", depends=depends_all,
              ),
    Extension("basinvolume.utils._utils_cpp",
              ["basinvolume/utils/_utils_cpp.cxx"],
              include_dirs=include_dirs,
              extra_compile_args=extra_compile_args,
              language="c++", depends=depends_bv,
              ),
    Extension("basinvolume.utils._cross_validation_cost_cpp",
              ["basinvolume/utils/_cross_validation_cost_cpp.cxx"],
              include_dirs=include_dirs,
              extra_compile_args=extra_compile_args,
              language="c++", depends=depends_bv,
              ),
               ]

setup(ext_modules=cxx_modules,
      )
