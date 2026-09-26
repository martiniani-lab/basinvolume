"""Build basinvolume. Works with `pip install .` and `python setup.py build_ext -i`.

pele, mcpele and PyCG_DESCENT must be importable (installed, or source checkouts
on PYTHONPATH): their C/C++ sources are compiled into basinvolume. The Cython
extensions are compiled by CMake (from CMakeLists.txt.in) inside build_ext;
setuptools then just copies them.

Options (command-line flags for direct `setup.py` use, env vars for pip):
  -j N / BV_JOBS               parallel build jobs (default: all cores)
  -c COMPILER                  unix (default) or intel
  --build-type / BV_BUILD_TYPE Release (default), Debug, RelWithDebInfo, MemCheck, Greene
  --native / BV_NATIVE         1 adds -march=native (default 0, as before)
"""
import argparse
import importlib.machinery
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext as old_build_ext

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-j", type=int, default=int(os.environ.get("BV_JOBS", os.cpu_count() or 4)))
parser.add_argument("-c", "--compiler", type=str, default=None)
parser.add_argument("--opt-report", action="store_true", default=False,
                    help="Print optimization report (for Intel compiler)")
parser.add_argument("--build-type", type=str, default=os.environ.get("BV_BUILD_TYPE", "Release"),
                    help="Release, Debug, RelWithDebInfo, MemCheck, Greene")
parser.add_argument("--native", type=int, default=int(os.environ.get("BV_NATIVE", 0)),
                    help="Compile with -march=native (non-portable binaries)")
jargs, remaining_args = parser.parse_known_args(sys.argv)

if not jargs.compiler or jargs.compiler in ("unix", "gnu", "gcc"):
    idcompiler = "unix"
elif jargs.compiler in ("intelem", "intel", "icc", "icpc"):
    idcompiler = "intel"
else:
    raise ValueError("unknown compiler " + jargs.compiler)
# Only add the option back if it was really set (setup.py install does not allow -c)
if jargs.compiler:
    remaining_args += ["-c", idcompiler]
sys.argv = remaining_args

build_type = jargs.build_type
build_type_args = {
    "Release": ["-O3", "-DNDEBUG"],
    "Greene": ["-O3", "-DNDEBUG", "-unroll", "-ip", "-axCORE-AVX512", "-qopenmp",
               "-qopt-report-stdout", "-qopt-report-phase=openmp"],
    "Debug": ["-ggdb3", "-O0"],
    "RelWithDebInfo": ["-g", "-O3"],
    "MemCheck": ["-g", "-O0", "-fsanitize=address", "-fsanitize=leak"],
}
if build_type not in build_type_args:
    raise ValueError("Unknown build type: " + build_type)
# env CXXFLAGS first (conda-forge hardening/arch flags), ours after so they win
cmake_compiler_extra_args = (
    shlex.split(os.environ.get("CXXFLAGS", ""))
    + ["-std=c++2a", "-Wall", "-Wextra", "-pedantic", "-fPIC"]
    + build_type_args[build_type]
)
if jargs.native and build_type in ("Release", "RelWithDebInfo"):
    cmake_compiler_extra_args += ["-march=native"]
if idcompiler == "intel" and jargs.opt_report:
    cmake_compiler_extra_args += ["-qopt-report=5"]

cmake_build_dir = os.path.join("build", "cmake")

cxx_files = [
    "basinvolume/monte_carlo/_conf_test_cpp.cxx",
    "basinvolume/monte_carlo/_action_cpp.cxx",
    "basinvolume/monte_carlo/_takestep_cpp.cxx",
    "basinvolume/monte_carlo/_independence_sampling.cxx",
    "basinvolume/utils/_utils_cpp.cxx",
    "basinvolume/utils/_cross_validation_cost_cpp.cxx",
]


def git_version():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C"})
        return out.stdout.strip().decode("ascii") or "Unknown"
    except OSError:
        return "Unknown"


def package_dir(name):
    """Directory of an installed dependency, found without importing it. pele, mcpele and
    PyCG_DESCENT are not on PyPI so they can't be build requirements; under pip's build
    isolation the environment's site-packages is off sys.path, so it is searched too."""
    site = list({sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]})
    spec = importlib.util.find_spec(name) or importlib.machinery.PathFinder.find_spec(name, site)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"{name} must be installed first: "
                           f"pip install git+https://github.com/martiniani-lab/{name}")
    return os.path.abspath(spec.submodule_search_locations[0])


def source_dir(pkg):
    """C/C++ sources of a dependency, same rule as its get_include(): installed next to
    the package, or `source/` in a source checkout"""
    installed = os.path.join(pkg, "source")
    return installed if os.path.isdir(installed) else os.path.join(os.path.dirname(pkg), "source")


def dependency_paths():
    pkgs = {name: package_dir(name) for name in ("pele", "mcpele", "PyCG_DESCENT")}
    pele_include = source_dir(pkgs["pele"])
    # a pele source checkout may carry its own sundials/eigen in extern/install
    extern = os.path.join(os.path.dirname(pele_include), "extern", "install")
    return dict(
        pele=pele_include,
        pele_pkg=pkgs["pele"],
        mcpele=source_dir(pkgs["mcpele"]),
        cgd=source_dir(pkgs["PyCG_DESCENT"]),
        # dirs holding the packages, so their .pxd cimports resolve under build isolation
        pxd_dirs=sorted({os.path.dirname(d) for d in pkgs.values()}),
        prefix=[extern] if os.path.isdir(extern) else [],
    )


def generate_cython(paths):
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    cmd = [sys.executable, os.path.join(cwd, "cythonize.py"), "basinvolume",
           "-I", os.path.join(paths["pele_pkg"], "potentials"),
           *[a for d in paths["pxd_dirs"] for a in ("-I", d)],
           "-X", "language_level=3", "-X", "c_string_type=unicode", "-X", "c_string_encoding=utf-8"]
    if subprocess.call(cmd, cwd=cwd) != 0:
        raise RuntimeError("Running cythonize failed!")


def get_ldflags():
    """linker flags for libpython (only used on macOS, see CMakeLists.txt.in)"""
    getvar = sysconfig.get_config_var
    libs = (getvar("LIBS") or "").split() + (getvar("SYSLIBS") or "").split()
    if not getvar("Py_ENABLE_SHARED"):
        libs.insert(0, "-L" + getvar("LIBDIR"))
    if not getvar("PYTHONFRAMEWORK"):
        # See https://github.com/kovidgoyal/kitty/issues/289#issuecomment-416040645
        libs.extend((getvar("LINKFORSHARED") or "").replace("-Wl,-stack_size,1000000", "").split())
    return " ".join(libs)


def write_cmakelists(paths):
    """create CMakeLists.txt from CMakeLists.txt.in"""
    import numpy as np

    with open("CMakeLists.txt.in", "r") as fin:
        cmake_txt = fin.read()
    python_includes = {sysconfig.get_path("include"), sysconfig.get_path("platinclude")}
    for key, value in [
        ("__PELE_INCLUDE__", paths["pele"]),
        ("__MCPELE_INCLUDE__", paths["mcpele"]),
        ("__PY_CGDESCENT_INCLUDE__", paths["cgd"]),
        ("__EXTRA_PREFIX_PATH__", " ".join(paths["prefix"])),
        ("__PYTHON_INCLUDE__", " ".join(sorted(python_includes))),
        ("__NUMPY_INCLUDE__", np.get_include()),
        ("__PYTHON_LDFLAGS__", get_ldflags()),
        ("__COMPILER_EXTRA_ARGS__", '"{}"'.format(" ".join(cmake_compiler_extra_args))),
    ]:
        cmake_txt = cmake_txt.replace(key, value)
    with open("CMakeLists.txt", "w") as fout:
        fout.write(cmake_txt)
        fout.write("\n")
        for fname in cxx_files:
            fout.write("make_cython_lib(${CMAKE_SOURCE_DIR}/%s)\n" % fname)


def which(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError("could not find " + name + " on PATH")
    return path


def get_compiler_env(compiler_id):
    """Environment and cmake args for the compilers.

    CC/CXX from the environment (e.g. conda compilers) are respected. Otherwise
    gcc is used, on macOS the newest homebrew gcc-N.
    """
    env = os.environ.copy()
    cmake_args = ["-DCMAKE_EXPORT_COMPILE_COMMANDS=1"]
    if compiler_id == "unix":
        if sys.platform.startswith("darwin") and "CC" not in env:
            version = next((v for v in range(20, 9, -1) if shutil.which(f"gcc-{v}")), None)
            if version is None:
                raise RuntimeError(
                    "Could not find a homebrew GNU compiler gcc-N (N=10..20) on PATH. "
                    "Install one or set CC and CXX."
                )
            env["CC"], env["CXX"] = which(f"gcc-{version}"), which(f"g++-{version}")
            prefixes = [subprocess.check_output(["brew", "--prefix", p]).decode().strip()
                        for p in ("openblas", "gettext")]
            cmake_args.append("-DCMAKE_PREFIX_PATH=" + ";".join(prefixes))
        env.setdefault("CC", "gcc")
        env.setdefault("CXX", "g++")
    elif compiler_id == "intel":
        env["CC"], env["CXX"], env["AR"] = which("icc"), which("icpc"), which("xiar")
        cmake_args.append("-DCMAKE_AR=" + env["AR"])
    else:
        raise Exception("compiler id not known")
    cmake_args += ["-DCMAKE_C_COMPILER=" + env["CC"], "-DCMAKE_CXX_COMPILER=" + env["CXX"]]
    return env, cmake_args


def run_cmake():
    os.makedirs(cmake_build_dir, exist_ok=True)
    print("\nrunning cmake in directory", cmake_build_dir)
    cwd = os.path.abspath(os.path.dirname(__file__))
    env, cmake_args = get_compiler_env(idcompiler)
    if shutil.which("ninja"):
        cache = os.path.join(cmake_build_dir, "CMakeCache.txt")
        # cmake refuses to switch generators in an existing build dir
        if os.path.isfile(cache) and "CMAKE_GENERATOR:INTERNAL=Ninja\n" not in open(cache).read():
            os.remove(cache)
            shutil.rmtree(os.path.join(cmake_build_dir, "CMakeFiles"), ignore_errors=True)
        cmake_args += ["-G", "Ninja"]
    if subprocess.call(["cmake"] + cmake_args + [cwd], cwd=cmake_build_dir, env=env) != 0:
        raise Exception("running cmake failed")
    print("\nbuilding files in cmake directory, jobs:", jargs.j)
    if subprocess.call(["cmake", "--build", ".", "-j", str(jargs.j)], cwd=cmake_build_dir, env=env) != 0:
        raise Exception("building libraries with CMake failed")
    print("finished building the extension modules with cmake\n")


class build_ext_precompiled(old_build_ext):
    """Build everything with CMake, then copy each library (stored in
    extension.sources[0]) to where setuptools expects the extension."""

    def run(self):
        paths = dependency_paths()
        generate_cython(paths)
        write_cmakelists(paths)
        run_cmake()
        super().run()

    def build_extension(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        pre_compiled_library = ext.sources[0]
        if not os.path.isfile(pre_compiled_library):
            raise RuntimeError(
                "file does not exist: " + pre_compiled_library + " Did CMake not run correctly"
            )
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        print("copying", pre_compiled_library, "to", ext_path)
        shutil.copy2(pre_compiled_library, ext_path)


ext_modules = [
    Extension(f.replace("/", ".")[: -len(".cxx")],
              [os.path.join(cmake_build_dir, os.path.basename(f).replace(".cxx", ".so"))])
    for f in cxx_files
]

# written before setup() so build_py installs the current one
# (layout, git_revision on the 3rd line, is parsed by basinvolume.utils._utils)
with open("basinvolume/version.py", "w") as f:
    f.write("\n# THIS FILE IS GENERATED FROM SCIPY SETUP.PY\ngit_revision = '%s'\n" % git_version())

# metadata lives in pyproject.toml
setup(
    # namespace: several subpackages (e.g. inverse_power_soft) have no __init__.py
    packages=find_namespace_packages(include=["basinvolume", "basinvolume.*"]),
    package_data={"": ["*.pxd"]},
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=build_ext_precompiled),
)
