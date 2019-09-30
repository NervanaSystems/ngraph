.. tutorials/onnx-tutorial.rst:

.. _onnx_tutorial:

Get Started with nGraph for ONNX
################################

Learn how to use nGraph to accelerate inference on ONNX workloads.

.. contents::

Overview
========

This tutorial is divided into two parts: a) building and installing 
nGraph for ONNX, and b) an example of how to use nGraph to accelerate 
inference on an ONNX model.


Build and install nGraph
========================

Software requirements
---------------------

* Python 3.4 or higher
* Protocol Buffers (``protobuf``) ``v2.6.1`` or higher
* `OpenCL runtime <opencl_drivers_>`_, required if you plan to use nGraph 
  with an Intel GPU backend. See also: :ref:`opencl`.
* `PlaidML <plaidml_pypi_>`_  ``v0.6.3`` or higher, required if you plan 
   to use nGraph’s :ref:`ngraph_plaidml_backend`.

Install protobuf for Ubuntu:

::

    # apt update
    # apt install -y protobuf-compiler libprotobuf-dev

Use pre-built packages
----------------------

The easiest way to install ``ngraph`` and ``ngraph-onnx`` is to use pre-built
packages from PyPI. Pre-built packages include the CPU backend and Intel GPU
backend.

.. note:: Pre-built packages (binaries) are currently not available for macOS.

Install ``ngraph-core``:

::

    pip install ngraph-core

Install ``ngraph-onnx``:

::

    pip install ngraph-onnx


Install ``plaidml`` (optional):

::

    pip install plaidml

.. note:: Installing the ``plaidml`` package is only required for users who plan to use nGraph with the PlaidML backend

Build from source
-----------------

Complete the following steps to build nGraph with Python bindings from source.
These steps have been tested on Ubuntu 18.04.

Before you build
~~~~~~~~~~~~~~~~

Prepare your system:

::

    apt update
    apt install -y python3 python3-pip python3-dev python-virtualenv
    apt install -y build-essential cmake curl clang-3.9 git zlib1g zlib1g-dev libtinfo-dev unzip autoconf automake libtool


Choose which backends to enable: 

**Intel GPU backend**

For ``INTELGPU`` support, use nGraph version ``0.24``.

To build nGraph with an Intel GPU backend, add ``-DNGRAPH_INTELGPU_ENABLE=TRUE``
to the cmake command. For example:

::

    cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_INTELGPU_ENABLE=TRUE

**PlaidML backend** 

To build nGraph with a PlaidML backend, add ``-DNGRAPH_PLAIDML_ENABLE=TRUE`` to 
the cmake command. For example:

::

    cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_PLAIDML_ENABLE=TRUE

To build nGraph with more than one backend, pass multiple flags to ``cmake``. 
For example:

:: 

    cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_PLAIDML_ENABLE=TRUE DNGRAPH_INTELGPU_ENABLE=TRUE

Build the nGraph wheel
~~~~~~~~~~~~~~~~~~~~~~


Clone nGraph's ``master`` branch and then build nGraph:

::

    cd # Change directory to where you would like to clone nGraph sources
    git clone -b master --single-branch --depth 1 https://github.com/NervanaSystems/ngraph.git
    mkdir ngraph/build && cd ngraph/build
    cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DNGRAPH_USE_PREBUILT_LLVM=TRUE 
    make install

Prepare a Python virtual environment for nGraph (recommended):
 
::

    mkdir -p ~/.virtualenvs && cd ~/.virtualenvs
    virtualenv -p $(which python3) nGraph
    source nGraph/bin/activate
    (nGraph) $ 

``(nGraph)`` indicates that you have created and activated a Python virtual 
environment called ``nGraph``.

Build a Python wheel for nGraph:

::

    (nGraph) $ cd # Change directory to where you have cloned nGraph sources
    (nGraph) $ cd ngraph/python
    (nGraph) $ git clone --recursive https://github.com/jagerman/pybind11.git
    (nGraph) $ export PYBIND_HEADERS_PATH=$PWD/pybind11
    (nGraph) $ export NGRAPH_CPP_BUILD_PATH=../build/ngraph_dist
    (nGraph) $ export NGRAPH_ONNX_IMPORT_ENABLE=TRUE
    (nGraph) $ pip install numpy
    (nGraph) $ python setup.py bdist_wheel

Navigate to the ``dist`` subdirectory to locate the Python wheel: ``ngraph-*.whl``

For additional information on how to build nGraph Python bindings see the
`Python API documentation <python_api_>`_.

Install the nGraph wheel
~~~~~~~~~~~~~~~~~~~~~~~~

Once the Python wheel ``ngraph-*.whl`` is built, install it
using ``pip``. For example:

::

    (nGraph) $ pip install -U dist/ngraph_core-0.0.0.dev0-cp36-cp36m-linux_x86_64.whl

Verify installation of nGraph (optional):

To verify that nGraph is properly installed in your Python shell:

.. code-block:: python

    >>> import ngraph as ng
    >>> ng.abs([[1, 2, 3], [4, 5, 6]])
    <Abs: 'Abs_1' ([2, 3])>

Additionally, check that nGraph and nGraph's Python wheel were
both built with the ``NGRAPH_ONNX_IMPORT_ENABLE`` option:

.. code-block:: python

    from ngraph.impl import onnx_import

If you don't see any errors, nGraph should be installed correctly.

Install ngraph-onnx
~~~~~~~~~~~~~~~~~~~

``ngraph-onnx`` is an additional Python library that provides a Python API to run
ONNX models using nGraph. 

To install ``ngraph-onnx``:

Clone ``ngraph-onnx`` sources to the same directory where you cloned ``ngraph`` 
sources.

::

    (nGraph) $ cd # Change directory to where you have cloned nGraph sources
    (nGraph) $ git clone -b master --single-branch --depth 1 https://github.com/NervanaSystems/ngraph-onnx.git
    (nGraph) $ cd ngraph-onnx

In your Python virtual environment, install the required packages and 
``ngraph-onnx``:

::

    (nGraph) $ pip install -r requirements.txt
    (nGraph) $ pip install -r requirements_test.txt
    (nGraph) $ pip install -e .
 
Verify installation of ``ngraph-onnx`` (optional):

To verify that ``ngraph-onnx`` installed correctly, you can run our test suite
using:

::

    (nGraph) $ pytest tests/ --backend=CPU -v
    (nGraph) $ NGRAPH_BACKEND=CPU TOX_INSTALL_NGRAPH_FROM=../ngraph/python tox

Run inference on an ONNX model
==============================

After installing ``ngraph-onnx`` from source, you can run inference on an
ONNX model. The model is a file which contains a graph representing a
mathematical formula (for example, a function such as y = f(x)). 

**Import a model**

See also: :ref:`import_serialized_onnx`

Download a model from the `ONNX model zoo <onnx_model_zoo_>`_. For example,
ResNet-50:

::

    wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
    tar -xzvf resnet50.tar.gz

Use the following Python commands to convert the downloaded model to an
nGraph model:

.. code-block:: python

    # Import ONNX and load an ONNX file from disk
    >>> import onnx
    >>> onnx_protobuf = onnx.load('resnet50/model.onnx')

    # Convert ONNX model to an ngraph model
    >>> from ngraph_onnx.onnx_importer.importer import import_onnx_model
    >>> ng_function = import_onnx_model(onnx_protobuf)

    # The importer returns a list of ngraph models for every ONNX graph output:
    >>> print(ng_function)
    <Function: 'resnet50' ([1, 1000])>

This creates an nGraph ``Function`` object, which can be used to execute a
computation on a chosen backend.

**Run the computation**

An ONNX model usually contains a trained neural network. To run inference on
this model, you execute the computation contained within the model.

After importing an ONNX model, you will have an nGraph ``Function`` object.
Now you can create an nGraph ``Runtime`` backend and use it to compile your
``Function`` to a backend-specific ``Computation`` object.

Execute your model by calling the created ``Computation`` object with input data:

.. code-block:: python

    # Using an ngraph runtime (CPU backend) create a callable computation object
    >>> import ngraph as ng
    >>> runtime = ng.runtime(backend_name='CPU')
    >>> resnet_on_cpu = runtime.computation(ng_function)

    # Load an image (or create a mock as in this example)
    >>> import numpy as np
    >>> picture = np.ones([1, 3, 224, 224], dtype=np.float32)

    # Run computation on the picture:
    >>> resnet_on_cpu(picture)
    [array([[2.16105007e-04, 5.58412226e-04, 9.70510227e-05, 5.76671446e-05,
             7.45318757e-05, 4.80892748e-04, 5.67404088e-04, 9.48728994e-05,
             ...

Use a different backend
-----------------------

A backend is a layer between nGraph and the device on your machine that executes the model.

You can substitute the default CPU backend with a different backend such as 
``INTELGPU`` or ``PlaidML``.

For running the computation on an Intel GPU, use the following line to create
the runtime:

.. code-block:: python

    runtime = ng.runtime(backend_name='INTELGPU')

Feedback
========

If you encounter any problems with this tutorial, please submit a ticket to our
`issues <issues_>`_ page on GitHub.

.. _onnx_model_zoo: https://github.com/onnx/models
.. _python_api: https://github.com/NervanaSystems/ngraph/blob/master/python/README.md
.. _opencl_drivers: https://software.intel.com/en-us/articles/opencl-drivers
.. _plaidml_pypi: https://pypi.org/project/plaidml/
.. _issues: https://github.com/NervanaSystems/ngraph/issues
