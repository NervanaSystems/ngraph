.. import.rst:

###############
Import a model
###############

:ref:`from_onnx`

nGraph APIs can be used to run inference on a model that has been *exported* 
from a Deep Learning framework. An export produces a file with a serialized 
model that can be loaded and passed to one of the nGraph backends.  


.. _from_onnx:

Importing a model from ONNX
============================

The most-widely supported :term:`export` format available today is `ONNX`_.
Models that have been serialized to ONNX are easy to identify; they are 
usually named ``<some_model>.onnx`` or ``<some_model>.onnx.pb``. These 
`tutorials from ONNX`_ describe how to turn trained models into an 
``.onnx`` export.

.. important:: If you landed on this page and you already have an ``.onnx`` or 
   an ``.onnx.pb`` formatted file, you should be able to run the inference without
   needing to dig into anything from the "Frameworks" sections. You will, however, 
   need to have completed the steps outlined in our :doc:`../../buildlb` guide.  

To demonstrate functionality, we'll use an already-serialized CIFAR10 model 
trained via ResNet20. Remember that this model has already been trained and 
exported from a framework such as Caffe2, PyTorch or CNTK; we are simply going 
to build an nGraph representation of the model, execute it, and produce some 
outputs.

Installing ``ngraph_onnx`` with nGraph from scratch
====================================================

See the documentation on: `building nGraph and nGraph-ONNX`_ for the latest 
instructions. 
 

.. _import_model:

Importing a serialized model
=============================

After building and installing ``ngraph_onnx``, we can import a model that has 
been serialized by ONNX, interact locally with the model by running 
Python code, create and load objects, and run inference. 

This section assumes that you have your own ONNX model. With this 
example model from Microsoft\*'s Deep Learning framework, `CNTK`_,
we can outline the procedure to show how to run ResNet on model 
that has been trained on the CIFAR10 data set and serialized with 
ONNX. 


(Optional) Localize your export to the virtual environment 
----------------------------------------------------------

For this example, let's say that our serialized file was output under our $HOME 
directory, say at ``~/onnx_conversions/trained_model.onnx``. To make loading this 
file easier, you can run the example below from your Venv in that directory. If 
you invoke your python interpreter in a different directory, you will need to 
specify the relative path to the location of the ``.onnx`` file.

.. important:: If you invoke your Python interpreter in directory other than 
   where you outputted your trained model, you will need to specify the 
   **relative** path to the location of the ``.onnx`` file.


.. code-block:: console 

   (onnx) $ cd ~/onnx_conversions 
   (onnx) $ python3 


Enable ONNX and load an ONNX file from disk
--------------------------------------------

.. literalinclude:: ../../../../examples/onnx/onnx_example.py
   :language: python
   :lines: 17-19

 
Convert an ONNX model to an ngraph model 
-------------------------------------------

.. literalinclude:: ../../../../examples/onnx/onnx_example.py
   :language: python
   :lines: 22-23


The importer returns a list of ngraph models for every ONNX graph 
output:


.. code-block:: python

   print(ng_models)
   [{
       'name': 'Plus5475_Output_0',
       'output': <Add: 'Add_1972' ([1, 10])>,
       'inputs': [<Parameter: 'Parameter_1104' ([1, 3, 32, 32], float)>]
    }]

The ``output`` field contains the ngraph node corrsponding to the output node 
in the imported ONNX computational graph. The ``inputs`` list contains all 
input parameters for the computation which generates the output.


 
Using ngraph_api, create a callable computation object
-------------------------------------------------------

.. literalinclude:: ../../../../examples/onnx/onnx_example.py
   :language: python
   :lines: 27-29


Load or create an image
------------------------

.. literalinclude:: ../../../../examples/onnx/onnx_example.py
   :language: python
   :lines: 32-33

Run ResNet inference on picture
---------------------------------

.. literalinclude:: ../../../../examples/onnx/onnx_example.py
   :language: python
   :lines: 36-37
 

Put it all together
===================

.. literalinclude:: ../../../../examples/onnx/onnx_example.py
   :language: python
   :lines: 17-37
   :caption: "Demo sample code to run inference with nGraph"


Outputs will vary greatly, depending on your model; for
demonstration purposes, the code will look something like: 


.. code-block:: python 

   array([[ 1.312082 , -1.6729496,  4.2079577,  1.4012241, -3.5463796,
         2.3433776,  1.7799224, -1.6155214,  0.0777044, -4.2944093]],
      dtype=float32)


.. _building nGraph and nGraph-ONNX: https://github.com/NervanaSystems/ngraph-onnx/blob/master/BUILDING.md
.. _ngraph-onnx: https://github.com/NervanaSystems/ngraph-onnx#ngraph
.. _ONNX: http://onnx.ai
.. _tutorials from ONNX: https://github.com/onnx/tutorials
.. _CNTK: https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/
