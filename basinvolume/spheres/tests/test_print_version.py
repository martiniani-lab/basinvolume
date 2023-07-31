from __future__ import print_function
from basinvolume.utils import (
    get_git_version,
    get_cython_version,
    get_python_version,
    get_git_version_from_build,
)

if __name__ == "__main__":
    git_stamp_basinvolume = get_git_version("basinvolume", False)
    bv_build = get_git_version_from_build("basinvolume")
    git_stamp_mcpele = get_git_version("mcpele", False)
    mcpele_build = get_git_version_from_build("mcpele")
    git_stamp_pele = get_git_version("pele", False)
    pele_build = get_git_version_from_build("pele")
    python_version = get_python_version()
    cython_version = get_cython_version()
    print("git_stamp_basinvolume:\n", git_stamp_basinvolume)
    print("from build:\n", bv_build)
    print("git_stamp_mcpele:\n", git_stamp_mcpele)
    print("from build:\n", mcpele_build)
    print("git_stamp_pele:\n", git_stamp_pele)
    print("from build:\n", pele_build)
    print("python_version:\n", python_version)
    print("cython_version:\n", cython_version)
