.. frameworks/onnx_integ.rst:


ONNX Support
============


nGraph is able to import and execute ONNX models. Models are converted to 
nGraph's internal representation and converted to ``Function`` objects, which 
can be compiled and executed on one of nGraph's backends.

You can use nGraph's Python API to run an ONNX model and nGraph can be used 
as an ONNX backend using the add-on package `nGraph ONNX`_.


.. note:: In order to support ONNX, nGraph must be built with the 
   ``NGRAPH_ONNX_IMPORT_ENABLE`` flag. See `Building nGraph-ONNX 
   <ngraph_onnx_building>`_ for more information. All nGraph packages 
   published on PyPI are built with ONNX support.


Installation
------------

To prepare your environment to use nGraph and ONNX, install the Python packages
for nGraph, ONNX and NumPy:

::

    $ pip install ngraph-core onnx numpy


Importing an ONNX model
-----------------------

You can download models from the `ONNX Model Zoo`_. For example, ResNet-50:

::

    $ wget https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz
    $ tar -xzvf resnet50.tar.gz


Use the following Python commands to convert the downloaded model to an nGraph 
``Function``:

.. code-block:: python

    # Import ONNX and load an ONNX file from disk
    >>> import onnx
    >>> onnx_protobuf = onnx.load('resnet50/model.onnx')

    # Convert ONNX model to an ngraph model
    >>> from ngraph.impl.onnx_import import import_onnx_model
    >>> ng_function = import_onnx_model(onnx_protobuf.SerializeToString())

    # The importer returns a list of ngraph models for every ONNX graph output:
    >>> print(ng_function)
    <Function: 'resnet50' ([1, 1000])>


This creates an nGraph ``Function`` object, which can be used to execute a 
computation on a chosen backend.

Running a computation
---------------------

You can now create an nGraph ``Runtime`` backend and use it to compile your 
``Function`` to a backend-specific ``Computation`` object. Finally, you can 
execute your model by calling the created ``Computation`` object with input 
data:

.. code-block:: python

    # Using an nGraph runtime (CPU backend) create a callable computation object
    >>> import ngraph as ng
    >>> runtime = ng.runtime(backend_name='CPU')
    >>> resnet_on_cpu = runtime.computation(ng_function)
    >>> print(resnet_on_cpu)
    <Computation: resnet50(Parameter_269)>

    # Load an image (or create a mock as in this example)
    >>> import numpy as np
    >>> picture = np.ones([1, 3, 224, 224], dtype=np.float32)

    # Run computation on the picture:
    >>> resnet_on_cpu(picture)
    [array([[2.16105007e-04, 5.58412226e-04, 9.70510227e-05, 5.76671446e-05,
             7.45318757e-05, 4.80892748e-04, 5.67404088e-04, 9.48728994e-05,
             ...


Find more information about nGraph and ONNX in the 
`nGraph ONNX`_ GitHub repository.


.. _ngraph ONNX: https://github.com/NervanaSystems/ngraph-onnx
.. _ngraph ONNX building: https://github.com/NervanaSystems/ngraph-onnx/blob/master/BUILDING.md
.. _ONNX model zoo: https://github.com/onnx/models
