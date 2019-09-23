# Building the Python API for nGraph

## Building nGraph Python Wheels

If you want to try a newer version of nGraph's Python API than is available 
from PyPI, you can build the latest version from source code. This process is 
very similar to what is outlined in our [ngraph_build] instructions with two 
important differences:

1. You must specify: `-DNGRAPH_PYTHON_BUILD_ENABLE=ON` and `-DNGRAPH_ONNX_IMPORT_ENABLE=ON` 
   when running `cmake`.

2. Instead of running `make`, use the command `make python_wheel`.

    `$ cmake ../ -DNGRAPH_PYTHON_BUILD_ENABLE=ON -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DNGRAPH_USE_PREBUILT_LLVM=ON`

    `$ make python_wheel`

After this procedure completes, the `ngraph/build/python/dist` directory should 
contain the Python packages of the version you cloned. For example, if you 
checked out and built `0.21` for Python 3.7, you might see something like: 

    $ ls python/dist/
    ngraph-core-0.21.0rc0.tar.gz  
    ngraph_core-0.21.0rc0-cp37-cp37m-linux_x86_64.whl  

### Using a virtualenv (optional)

You may wish to use a virutualenv for your installation.

    $ virtualenv -p $(which python3) venv
    $ source venv/bin/activate
    (venv) $

### Installing the wheel

You may wish to use a virutualenv for your installation.

    (venv) $ pip install ngraph/build/python/dist/ngraph_core-0.21.0rc0-cp37-cp37m-linux_x86_64.whl

## Running tests

Unit tests require additional packages be installed:

    (venv) $ cd ngraph/python
    (venv) $ pip install -r test_requirements.txt

Then run tests:

    (venv) $ pytest test/ngraph/

[ngraph_build]: http://ngraph.nervanasys.com/docs/latest/buildlb.html
