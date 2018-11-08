onnx_ssg.rst


Shared subgraphs with ONNX 
==========================



ShuffleNet Example
------------------


`ShuffleNet`_ describes itself as "deep convolutional networks for classification",
and has a ``model.onnx`` format available for testing.  

To use nGraph's "import a model" feature to load ShuffleNet and enable working 
with subgraphs, build the library as described in :doc:`../howto/import`. Be sure to 
append the :command:`cmake` with correct ``DNGRAPH_ONNX_IMPORT_ENABLE=TRUE`` 
option 

.. code-block:: console

    cmake ../ -DCMAKE_INSTALL_PREFIX=~/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE


And, after running ``make``, export the environment variables as follows:

.. code-block:: console

    export NGRAPH_CPP_BUILD_PATH=~/ngraph_dist/
    export LD_LIBRARY_PATH=~/ngraph_dist/lib

If you already have your ``onnx`` virtual environment set from the install, reactivate it. 


.. code-block:: console

   /opt/libraries/ngraph/onnx$ . bin/activate
   (onnx) indie@toimisaki:/opt/libraries/ngraph/onnx$ python3
   Python 3.6.6 (default, Sep 12 2018, 18:26:19) 
   [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]] on linux
   Type "help", "copyright", "credits" or "license" for more information.

.. code-block:: python

   import onnx
   onnx_protobuf = onnx.load('/path/to/downloaded/onnx_models/shufflenet/model.onnx')
   from ngraph_onnx.onnx_importer.importer import import_onnx_model
   ng_model = import_onnx_model(onnx_protobuf)[0]
	

The output looks something like this: 

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



.. code-block:: python

	import ngraph as ng
	runtime = ng.runtime(backend_name='CPU')
	shufflenet = runtime.computation(ng_model['output'], *ng_model['inputs'])

















.. _ShuffleNet: https://github.com/onnx/models/blob/master/shufflenet/README.md
