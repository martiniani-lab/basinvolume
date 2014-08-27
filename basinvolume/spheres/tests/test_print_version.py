from basinvolume.utils import *

if __name__ == "__main__":
    git_stamp_basinvolume = get_git_version('basinvolume')
    git_stamp_mcpele = get_git_version('mcpele')
    git_stamp_pele = get_git_version('pele')
    python_version = get_python_version()
    cython_version = get_cython_version()
    print "git_stamp_basinvolume:\n", git_stamp_basinvolume
    print "git_stamp_mcpele:\n", git_stamp_mcpele
    print "git_stamp_pele:\n", git_stamp_pele
    print "python_version:\n", python_version
    print "cython_version:\n", cython_version