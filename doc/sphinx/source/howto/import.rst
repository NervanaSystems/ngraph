.. import.rst:

###############
Import a model
###############

:ref:`from_onnx`


.. TODO Make sure that this is the first page data scientists find when they 
.. are simply trying to run a trained model; they DO NOT need to do a system
.. install of the Intel nGraph++ bridges; they can use our Python APIs to run 
.. a trained model. 

Intel nGraph APIs can be used to import a model that has been *exported* from 
a Deep Learning framework. The export producess a file with a serialized model 
that can be loaded and passed to one of the nGraph backends for execution.

.. _from_onnx:

Importing models from ONNX
===========================

The most-widely supported :term:`export` format available today is `ONNX`_.
Models that have been serialized to ONNX are easy to identify; they are 
usually named ``<some_model>.onnx`` or ``<some_model>.onnx.pb``. These 
`tutorials from ONNX`_ describe how to turn trained models into an 
``.onnx`` export.

.. important:: If you landed on this page and you already have an ``.onnx`` 
   or ``.onnx.pb`` formatted file, you should be able to run the inference 
   without needing to dig into anything from the "Frameworks" sections. You 
   will, however, need to have completed the steps described in 
   our :doc:`../install` guide.  

To demonstrate this functionality, we'll use an `already serialized CIFAR10`_ 
model trained via ResNet20. Remember that this model *has already been trained* to 
a degree deemed well enough by a developer, and then exported from a framework 
such as Caffe2, PyTorch or CNTK. We are simply going to build an nGraph 
representation of the model, execute it, and produce some outputs.


Installing ``ngraph_onnx``
--------------------------

In order to use ONNX models, you will also need the companion tool ``ngraph_onnx``. 
``ngraph_onnx`` requires Python 3.5 or higher.


#. First set the environment variables to where we built the nGraph++ libraries;
   This code assumes that you followed the default instructions from the 
   :doc:`../install` guide and that your version of ``ngraph_dist`` can be found 
   at ``$HOME/ngraph_dist``:

   .. code-block:: bash

      export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib
      export DYLD_LIBRARY_PATH=$HOME/ngraph_dist/lib  # On MacOS

#. Now add *Protocol Buffers* and Python3 PIP dependencies to your system. ONNX 
   requires Protocol Buffers version 2.6.1 or higher. For example, on Ubuntu:

   .. code-block:: console

      $ sudo apt install protobuf-compiler libprotobuf-dev python3-pip

#. Checkout the branch named `python_binding`: 

   .. code-block:: console

      $ cd /opt/libraries/ngraph
      $ git checkout python_binding
        Switched to branch 'python_binding'
        Your branch is up-to-date with 'origin/python_binding'.       

#. Recursively update the submodule and install the Python dependencies. 

   .. code-block:: console

      $ git submodule update --init --recursive
      $ cd python
	   $ pip3 install -r requirements.txt
	   $ pip3 install .

#. Finally, clone the ``ngraph-onnx`` repo and use :command:`pip` to 
   install the Python dependencies for this tool; if you set up your 
   original nGraph library installation under a ``libraries`` directory 
   as recommended, it's a good idea to clone this repo there, too.
   
   .. code-block:: console

      $ cd /opt/libraries
      $ git clone git@github.com:NervanaSystems/ngraph-onnx
      $ cd ngnraph-onnx
      $ pip3 install -r requirements.txt
      $ pip3 install .
 

Importing a serialized model
-----------------------------

.. Now we can import any model that has been serialized by ONNX, 
   run Python code locally to interact with that model, create and
   load objects, and run inference.

These instructions demonstrate how to run ResNet on an 
`already serialized CIFAR10`_ model trained via ResNet20. 


Import ONNX and load an ONNX file from disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-19

 
Convert an ONNX model to an ngraph model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 22-23

 
Create a callable computation object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 27-29


Load or create an image
~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 32-33

Run ResNet inference on picture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 36
 

Put it all together
===================

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-37
   :caption: "Code to run inference on a CIFAR10 trained model"


If you tested the ``.onnx`` file used in the example above, the outputs 
should look something like:  

.. code-block:: python 

   Attempting to write a float64 value to a <Type: 'float32'> tensor. Will attempt type conversion.
   array([[ 1.3120822 , -1.6729498 ,  4.2079573 ,  1.4012246 , -3.5463796 ,
        2.343378  ,  1.7799224 , -1.6155218 ,  0.07770489, -4.2944083 ]],
     dtype=float32)



.. Importing models from NNVM
   ---------------------------

.. if you work on NNVM you can add this instuction here. 



.. Importing models from XLA
   --------------------------

.. if you work on XLA you can add this instruction here.


.. etc, eof 



.. _ONNX: http://onnx.ai
.. _tutorials from ONNX: https://github.com/onnx/tutorials
.. _already serialized CIFAR10: https://github.com/NervanaSystems/ngraph-onnx-val/tree/master/models

