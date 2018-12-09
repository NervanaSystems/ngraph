.. onnx_ssg_tutorial:

Shared subgraphs with ONNX 
==========================


ShuffleNet Example
------------------

.. note:: The following example describes how additional ops documented 
   with :doc:`../python_api/index` can be applied; it is not necessarily 
   the only or best bridge (or bridge-enabling) mechanism.


`ShuffleNet`_ describes itself as "deep convolutional networks for classification",
and has a ``model.onnx`` format available for testing.  This example assumes that 
you have installed the ``ngraph-onnx`` package and the accompanying ``ngraph-core`` 
from pip:

.. code-block:: console

   (venv) $ pip install ngraph-core
   (venv) $ pip install ngraph-onnx


.. code-block:: python

   import onnx
   onnx_protobuf = onnx.load('/path/to/downloaded/onnx_models/shufflenet/model.onnx')
   from ngraph_onnx.onnx_importer.importer import import_onnx_model
   ng_model = import_onnx_model(onnx_protobuf)[0]
	

The output should look something like this: 

:: 

    ONNX `ai.onnx` opset version 9 is not supported. Falling back to latest supported version: 7
	More than one different shape in input nodes [<Constant: 'Constant_1111' ([])>, <BatchNormInference: 'gpu_0/gconv1_3_bn_1' ([1, 136, 28, 28])>].
	More than one different shape in input nodes [<Constant: 'Constant_1564' ([])>, <BatchNormInference: 'gpu_0/gconv1_5_bn_1' ([1, 136, 28, 28])>].
	More than one different shape in input nodes [<Constant: 'Constant_2017' ([])>, <BatchNormInference: 'gpu_0/gconv1_7_bn_1' ([1, 136, 28, 28])>].
	More than one different shape in input nodes [<Constant: 'Constant_3326' ([])>, <BatchNormInference: 'gpu_0/gconv1_11_bn_1' ([1, 272, 14, 14])>].
	More than one different shape in input nodes [<Constant: 'Constant_4187' ([])>, <BatchNormInference: 'gpu_0/gconv1_13_bn_1' ([1, 272, 14, 14])>].
	More than one different shape in input nodes [<Constant: 'Constant_5048' ([])>, <BatchNormInference: 'gpu_0/gconv1_15_bn_1' ([1, 272, 14, 14])>].
	More than one different shape in input nodes [<Constant: 'Constant_5909' ([])>, <BatchNormInference: 'gpu_0/gconv1_17_bn_1' ([1, 272, 14, 14])>].
	More than one different shape in input nodes [<Constant: 'Constant_6770' ([])>, <BatchNormInference: 'gpu_0/gconv1_19_bn_1' ([1, 272, 14, 14])>].
	More than one different shape in input nodes [<Constant: 'Constant_7631' ([])>, <BatchNormInference: 'gpu_0/gconv1_21_bn_1' ([1, 272, 14, 14])>].
	More than one different shape in input nodes [<Constant: 'Constant_8492' ([])>, <BatchNormInference: 'gpu_0/gconv1_23_bn_1' ([1, 272, 14, 14])>].
	More than one different shape in input nodes [<Constant: 'Constant_11025' ([])>, <BatchNormInference: 'gpu_0/gconv1_27_bn_1' ([1, 544, 7, 7])>].
	More than one different shape in input nodes [<Constant: 'Constant_12702' ([])>, <BatchNormInference: 'gpu_0/gconv1_29_bn_1' ([1, 544, 7, 7])>].
	More than one different shape in input nodes [<Constant: 'Constant_14379' ([])>, <BatchNormInference: 'gpu_0/gconv1_31_bn_1' ([1, 544, 7, 7])>].


These outputs can now be used to start working with, for example, the  `broadcast shapes`_ op: 

.. code-block:: python

   import ngraph as ng
   runtime = ng.runtime(backend_name='CPU')
   print(runtime)
   <Runtime: Backend='CPU'>
   import numpy as np
   input_node = ng.constant([1, 136, 28, 28])
   current_shape = [28]
   new_shape = [14, 14]
















.. _nGraph-ONNX pyapi: https://ngraph.nervanasys.com/docs/latest/python_api/_autosummary/ngraph.html

.. _ShuffleNet: https://github.com/onnx/models/blob/master/shufflenet/README.md
.. _broadcast shapes: https://ngraph.nervanasys.com/docs/latest/python_api/_autosummary/ngraph.html#ngraph.ops.broadcast_to
