# nGraph python binding

## Installation

### Install nGraph (Required)

```
Follow the [steps](http://ngraph.nervanasys.com/docs/latest/install.html) to build and install ngraph.
```

Clone the pybind repository

```
cd ngraph/python
git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
```

Install Wrapper (python binding)

```
python ../build/setup.py install
```

To run unit tests, first install additional required packages.

```
pip install -r test_requirements.txt
```

Then run a test.

```
pytest test/test_ops.py
pytest test/ngraph/
```

## Running tests with tox

[Tox](https://tox.readthedocs.io/) is a Python [virtualenv](https://virtualenv.pypa.io/) management and test command line tool. In our project it automates:

* running unit tests using [pytest](https://docs.pytest.org/)
* checking that code style is compliant with [PEP8](https://www.python.org/dev/peps/pep-0008/) using [Flake8](http://flake8.pycqa.org/)
* static type checking using [MyPy](http://mypy.readthedocs.io)
* testing across Python 2 and 3

Installing and running test with Tox:

    pip install tox
    tox

You can run tests using only Python 3 or 2 using the `-e` (environment) switch:

    tox -e py36
    tox -e py27

You can check styles in a particular code directory by specifying the path:

    tox ngraph/

In case of problems, try to recreate the virtual environments by deleting the `.tox` directory:

```
rm -rf .tox
tox
```

