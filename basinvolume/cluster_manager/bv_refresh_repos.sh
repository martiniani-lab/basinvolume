rm -rf pymbar pele mcpele basinvolume PyCG_DESCENT trajectories
git clone git@github.com:choderalab/pymbar.git
git clone git@github.com:pele-python/pele.git
git clone git@github.com:pele-python/mcpele.git
git clone git@github.com:smcantab/PyCG_DESCENT.git
git clone git@bitbucket.org:smcantab/basinvolume.git
git clone git@bitbucket.org:trajectories_team/trajectories.git
cd pymbar
python setup.py build_ext -i
cd ..
for r in pele mcpele PyCG_DESCENT basinvolume
do
	cd $r
	python setup_with_cmake.py build_ext -i
	cd ..
done
cd trajectories
python setup.py build_ext -i
cd ..
