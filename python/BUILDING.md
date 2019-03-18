# Building the Python API for nGraph

## Building nGraph Python Wheels

[nGraph's build instructions][ngraph_build] give detailed instructions on building nGraph on different operating systems. Please make sure you specify the options `-DNGRAPH_PYTHON_BUILD_ENABLE=ON` and `-DNGRAPH_ONNX_IMPORT_ENABLE=ON` when building nGraph. Use the `make python_wheel` command to build nGraph and create Python packages. 

Basic build procedure on an Ubuntu system: 
    
    # apt-get install build-essential cmake clang-3.9 clang-format-3.9 git curl zlib1g zlib1g-dev libtinfo-dev unzip autoconf automake libtool
    # apt-get install python3 python3-dev python python-dev python-virtualenv
    
    $ git clone https://github.com/NervanaSystems/ngraph.git
    $ cd ngraph/
    $ mkdir build
    $ cd build/
    $ cmake ../ -DNGRAPH_PYTHON_BUILD_ENABLE=ON -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DNGRAPH_USE_PREBUILT_LLVM=ON
    $ make python_wheel

After this procedure completes, the `ngraph/build/python/dist` directory should contain Python packages. 

    $ ls python/dist/
    ngraph-core-0.10.0.tar.gz  
    ngraph_core-0.10.0-cp27-cp27mu-linux_x86_64.whl  
    ngraph_core-0.10.0-cp35-cp35m-linux_x86_64.whl

### Using a virtualenv (optional)

You may wish to use a virutualenv for your installation.

    $ virtualenv -p $(which python3) venv
    $ source venv/bin/activate
    (venv) $

### Installing the wheel

You may wish to use a virutualenv for your installation.

    (venv) $ pip install ngraph/build/python/dist/ngraph_core-0.10.0-cp35-cp35m-linux_x86_64.whl

## Running tests

Unit tests require additional packages be installed:

    (venv) $ cd ngraph/python
    (venv) $ pip install -r test_requirements.txt

Then run tests:

    (venv) $ pytest test/ngraph/

[ngraph_build]: http://ngraph.nervanasys.com/docs/latest/buildlb.html
