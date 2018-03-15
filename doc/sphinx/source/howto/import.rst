.. import.rst:

################
Importing models
################

:ref:`from_onnx`


.. TODO Make sure that this is the first page data scientists find when they 
.. are simply trying to run a trained model; they DO NOT need to do a system
.. install of the Intel nGraph++ bridges; they can use our Python APIs to run 
.. a trained model. 

Intel nGraph APIs can be used to run inference on a model that has been *exported* 
from a Deep Learning framework. The entity exporting the model produces a file
with a serialized model that can be passed to one of the nGraph backends for computation.


.. _from_onnx:

Importing models from ONNX
===========================

The most-widely supported :term:`export` format available today is `ONNX`_.
Models that have been serialized to ONNX are easy to identify; they are 
usually named ``<some_model>.onnx`` or ``<some_model>.onnx.pb``. These 
`tutorials from ONNX`_ describe how to turn trained models into an 
``.onnx`` export.

If you landed on this page and you already have an ``.onnx`` formatted-file, you 
should be able to run the inference without needing to dig into anything from 
the "Frameworks" sections. You will, however, need to have completed the 
steps described in our :doc:`../install` guide.  

To demonstrate the functionality of the nGraph inference, we'll use an 
`already serialized CIFAR10`_ ResNet model for demonstration purposes. Remember 
that this model has already been trained and exported from a framework such as 
Caffe2, PyTorch or CNTK; we are simply going to build an nGraph representation 
of the model, execute it, and produce some outputs.


Installing ngraph_onnx
-----------------------

In order to use ONNX models, you will also need the companion tool ``ngraph_onnx``. 
``ngraph_onnx`` requires Python 3.5 or higher.


#. First set the environment variables to where we built the nGraph++ libraries;
   This code assumes that you followed the default instructions from the 
   :doc:`../install` guide and ``ngraph_dist`` can be found at ``$HOME/ngraph_dist``:

   .. code-block:: bash

      export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib
      export DYLD_LIBRARY_PATH=$HOME/ngraph_dist/lib  # On MacOS

#. Now add Protocol Buffers and Python PIP dependencies to your system. ONNX requires 
   Protocol Buffers version 2.6.1 or higher.  For example, on Ubuntu:

   .. code-block:: console

      $ sudo apt install protobuf-compiler libprotobuf-dev python3-pip

#. Checkout the branch named `python_binding`: 

   .. code-block:: console

      $ cd /opt/libraries/ngraph-cpp
      $ git checkout python_binding
        Switched to branch 'python_binding'
        Your branch is up-to-date with 'origin/python_binding'.       

#. Recursively update the submodule and install the Python dependencies. 

   .. code-block:: console

      $ git submodule update --init --recursive
      $ cd /path/to/ngraph/python
	   $ pip3 install -r requirements.txt
	   $ pip3 install .


#. Clone the ``ngraph-onnx`` repo and pip install the Python dependencies
   for this repo as well; if you set up your original nGraph installation 
   under a ``libraries`` directory as recommended, it's a good idea to 
   clone this repo there, too.
   
   .. code-block:: console

      $ cd /opt/libraries
      $ git clone git@github.com:NervanaSystems/ngraph-onnx
      $ cd ngnraph-onnx
      $ pip3 install -r requirements.txt
      $ pip3 install .
 
Now we can use the following Python code to run a ResNet model:
 


Import ONNX and load an ONNX file from disk
--------------------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-19

 
Prepare an ONNX model for computation
--------------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 22-23

 
Create a callable computation object
-------------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 27-29


Load or create an image
------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 32-33

Run ResNet20 inference on picture
----------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 36
 

Put it all together
===================

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-37
   :caption: "Code to run inference on a CIFAR10 trained model"



.. Importing models from NNVM
   ---------------------------

.. Importing models from XLA
   --------------------------

.. etc, eof 



.. _ONNX: http://onnx.ai
.. _tutorials from ONNX: https://github.com/onnx/tutorials
.. _already serialized CIFAR10: https://github.com/NervanaSystems/ngraph-onnx-val/tree/master/models

