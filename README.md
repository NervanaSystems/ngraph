# ngraph-neon

## Installation - New Way

Checkout ngraph++ and python wrapper code and build bdist wheel.

```
git clone --branch python_binding --recursive https://github.com/NervanaSystems/ngraph-cpp.git
cd ngraph-cpp/python
```
To build python2 bdist wheel type
```
./build2.sh
```
To build python3 bdist wheel type
```
./build3.sh
```

The bdist wheel will be placed in ngraph-cpp/python/build/dist
Activate your virtual environment and install the bdist wheel

```
pip install -U <full path to the bdist wheel>
```

For example, On MacOS you would run a command like,

```
pip install -U build/dist/pyngraph-0.0.1-cp27-cp27m-macosx_10_13_intel.whl
```

To run unit tests, first install additional required packages.

```
pip install -r test_requirements.txt
```

Then run a test.
```
pytest test/test_ops.py
```

## Running tests with tox

[Tox](https://tox.readthedocs.io/) is a Python [virtualenv](https://virtualenv.pypa.io/) management and test command line tool. In our project it automates:

* running unit tests using [pytest](https://docs.pytest.org/)
* checking that code style is compliant with [PEP8](https://www.python.org/dev/peps/pep-0008/) using [Flake8](http://flake8.pycqa.org/)
* static type checking using [MyPy](http://mypy.readthedocs.io)
* testing across Python 2 and 3

Installing and running test with Tox:

    pip install tox
    export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
    tox

You can run tests using only Python 3 or 2 using the `-e` (environment) switch:

    tox -e py36
    tox -e py27

You can check styles in a particular code directory by specifying the path:

    tox ngraph_api/

In case of problems, try to recreate the virtual environments by deleting the `.tox` directory:

```
rm -rf .tox
tox
```

## Installation - Old Way (Still works)

Follow these steps to install the ngraph's python wrapper and its prerequisites.


### ngraph-cpp

Download the required version of ngraph-cpp and install it.
```
git clone https://github.com/NervanaSystems/ngraph-cpp.git
cd ngraph-cpp
git checkout 89da71d33656b972e85ce4107e82643bfa195b5b -b "local branch name"
Build and Install it : https://github.com/NervanaSystems/ngraph-cpp#steps
```

### ngraph-neon

After installing ngraph-cpp, follow the steps below to install ngraph-neon.
The NGRAPH_CPP_BUILD_PATH is set to default installation location of ngraph-cpp.
```
git clone --recursive https://github.com/NervanaSystems/ngraph-neon.git
cd ngraph-neon
export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist/
pip install -U .
```
