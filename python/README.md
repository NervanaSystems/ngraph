# Python binding for the nGraph Library

## Build the nGraph Library (required)

Follow the [build instructions] to build nGraph. When you get to the `cmake` 
command, be sure to specify the option that enables ONNX support in the Library: 

    $ cmake ../ -DNGRAPH_ONNX_IMPORT_ENABLE=ON 


Next, clone the `pybind11` repository:

    $ cd ngraph/python
    $ git clone --recursive https://github.com/pybind/pybind11.git


Set the environment variables:

    export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
    export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib
    export DYLD_LIBRARY_PATH=$HOME/ngraph_dist/lib # (Only needed on MacOS)
    export PYBIND_HEADERS_PATH=pybind11


Install the wrapper (Python binding):

    $ python setup.py install


Unit tests require additional packages be installed:

    $ pip install -r test_requirements.txt


Then run a test:

    $ pytest test/test_ops.py
    $ pytest test/ngraph/


## Running tests with tox

[Tox] is a Python [virtualenv] management and test command line tool. In our 
project it automates:

* running of unit tests with [pytest]
* checking that code style is compliant with [PEP8] using [Flake8]
* static type checking using [MyPy]
* testing across Python 2 and 3


Installing and running test with Tox:

    $ pip install tox
    $ tox


You can run tests using only Python 3 or 2 using the `-e` (environment) switch:

    $ tox -e py36
    $ tox -e py27


You can check styles in a particular code directory by specifying the path:

    $ tox ngraph/


If you run into any problems, try recreating the virtual environments by 
deleting the `.tox` directory:

    $ rm -rf .tox
    $ tox

[build instructions]:http://ngraph.nervanasys.com/docs/latest/buildlb.html
[Tox]:https://tox.readthedocs.io/
[virtualenv]:https://virtualenv.pypa.io/
[pytest]:https://docs.pytest.org/
[PEP8]:https://www.python.org/dev/peps/pep-0008
[Flake8]:http://flake8.pycqa.org
[MyPy]:http://mypy.readthedocs.io