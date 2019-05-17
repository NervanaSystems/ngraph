# Building manylinux2010 version of Python API for nGraph (nGraph-core)

## Requirements

Linux system with Git and Docker installed.

## Supported Python versions

Python 2.7, 3.4, 3.5, 3.6, 3.7

## Build instructions

Assume you want to place your build directory `build` parallel to ngraph source directory.
Then follow the steps below.

    $ git clone https://github.com/NervanaSystems/ngraph.git
    $ mkdir build
    $ cd build/
    $ ../ngraph/contrib/manylinux2010/make-manylinux2010.sh

After this procedure completes, the `python/dist` directory should contain Python packages.

    $ ls python/dist
    ngraph_core-0.19.0rc0-cp27-cp27m-manylinux2010.whl
    ngraph_core-0.19.0rc0-cp34-cp34m-manylinux2010.whl
    ngraph_core-0.19.0rc0-cp36-cp36m-manylinux2010.whl
    ngraph_core-0.19.0rc0-cp27-cp27mu-manylinux2010.whl
    ngraph_core-0.19.0rc0-cp35-cp35m-manylinux2010.whl
    ngraph_core-0.19.0rc0-cp37-cp37m-manylinux2010.whl
