rm -rf build
rm -rf maskrcnn_benchmark.egg-info
python setup.py build_ext install_lib
python setup.py build develop
