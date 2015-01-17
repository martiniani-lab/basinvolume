rm -rf pymbar pele mcpele basinvolume
git clone git@github.com:choderalab/pymbar.git
git clone git@github.com:pele-python/pele.git
git clone git@github.com:pele-python/mcpele.git
git clone git@bitbucket.org:smcantab/basinvolume.git
cd pymbar
python setup.py build_ext -i
cd ..
for r in pele mcpele basinvolume 
do
	cd $r
	python setup_with_cmake.py build_ext -i
	cd ..
done
