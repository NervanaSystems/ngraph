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

.. important:: If you landed on this page and you already have an ``.onnx`` 
   or ``.onnx.pb`` formatted file, you should be able to run the inference 
   without needing to dig into anything from the "Frameworks" sections. You 
   will, however, need to have completed the steps outlined in 
   our :doc:`../install` guide.  

To demonstrate functionality, we'll use an already serialized CIFAR10 model 
trained via ResNet20. Remember that this model has already been trained and 
exported from a framework such as Caffe2, PyTorch or CNTK; we are simply going 
to build an nGraph representation of the model, execute it, and produce some 
outputs.


Installing ``ngraph_onnx`` with nGraph from scratch
====================================================

To use ONNX models with nGraph, you will also need the companion tool 
``ngraph_onnx``, which requires Python 3.4 or higher. If nGraph has not 
yet been installed to your system, you can follow these steps to install 
everything at once; if an `ngraph_dist` is already installed on your system, 
skip ahead to the next section, :ref:`install_ngonnx`.
   

#. Install prerequisites for the system and install nGraph as ``ngraph_dist``:
  
   .. code-block:: console

      $ apt update
      $ apt install python3 python3-pip python3-dev
      $ apt install build-essential cmake curl clang-3.9 git zlib1g zlib1g-dev libtinfo-dev
      $ git clone https://github.com/NervanaSystems/ngraph.git
      $ cd ngraph && mkdir build
      $ cd build && cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE
      $ make install

#. Build the Python package (binary wheel) for ngraph:

   .. code-block:: console

      $ cd ngraph/python
      $ git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
      $ export PYBIND_HEADERS_PATH=$PWD/pybind11
      $ export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
      $ python3 setup.py bdist_wheel

#. Check for the binary wheel file under ``/ngraph/python/dist/`` and activate a 
   Python3 virtual environment to work with ONNX.

   .. code-block:: console

      $ cd dist/
      $ python3 -m venv onnx
      $ cd onnx/
      $ . bin/activate
      (onnx)$ pip install -U ngraph-0.2.0-cp35-cp35m-linux_x86_64.whl    


#. Confirm ngraph is properly installed through a Python interpreter:

   .. code-block:: console

      (onnx) $ python3

   .. code-block:: python
      
      import ngraph as ng
      ng.abs([[1, 2, 3], [4, 5, 6]])
      <Abs: 'Abs_1' ([2, 3])>

   If you don't see any errors, ngraph should be installed correctly.


.. _install_ngonnx:

Installing ngraph-onnx
-----------------------

Install the ``ngraph-onnx`` companion tool using pip:

.. code-block:: console

   (onnx) $ pip install git+https://github.com/NervanaSystems/ngraph-onnx/
 

Importing a serialized model
=============================

With the dependencies added, we can now import a model that has 
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

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-19

 
Convert an ONNX model to an ngraph model 
-------------------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
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

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 27-29


Load or create an image
------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 32-33

Run ResNet inference on picture
---------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 36-37
 

Put it all together
===================

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-37
   :caption: "Demo sample code to run inference with nGraph"


Outputs will vary greatly, depending on your model; for
demonstration purposes, the code will look something like: 


.. code-block:: python 

   array([[ 1.312082 , -1.6729496,  4.2079577,  1.4012241, -3.5463796,
         2.3433776,  1.7799224, -1.6155214,  0.0777044, -4.2944093]],
      dtype=float32)



.. Importing models from NNVM
   ---------------------------

.. if you work on NNVM you can add this instruction here. 



.. Importing models serialized with XLA
   -------------------------------------

.. if you work on XLA you can add this instruction here.


.. etc, eof 


.. _ngraph-onnx: https://github.com/NervanaSystems/ngraph-onnx#ngraph
.. _ONNX: http://onnx.ai
.. _tutorials from ONNX: https://github.com/onnx/tutorials
.. _CNTK: https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/
