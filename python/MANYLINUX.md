# Building manylinux1 version of Python API for nGraph (nGraph-core)

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
    $ ../ngraph/contrib/docker/make-manylinux1.sh

After this procedure completes, the `dist` directory should contain Python packages.

    $ ls dist
    ngraph_core-0.10.0-cp27-cp27m-manylinux1.whl
    ngraph_core-0.10.0-cp34-cp34m-manylinux1.whl
    ngraph_core-0.10.0-cp36-cp36m-manylinux1.whl
    ngraph_core-0.10.0-cp27-cp27mu-manylinux1.whl
    ngraph_core-0.10.0-cp35-cp35m-manylinux1.whl
    ngraph_core-0.10.0-cp37-cp37m-manylinux1.whl
