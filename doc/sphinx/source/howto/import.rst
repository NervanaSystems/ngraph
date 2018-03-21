.. import.rst:

###############
Import a model
###############

:ref:`from_onnx`

.. That can be the first page data scientists find when they are simply trying 
.. to run a trained model; they DO NOT need to do a system install of the Intel
.. nGraph++ bridges; they can use our Python APIs to run a trained model.
..  

The Intel nGraph APIs can be used to run inference on a model that has been 
*exported* from a Deep Learning framework. An export produces a file with 
a serialized model that can be loaded and passed to one of the nGraph 
backends.  

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


Installing ``ngraph_onnx``
==========================

To use ONNX models with ngraph, you will also need the companion tool 
``ngraph_onnx``. ``ngraph_onnx`` requires Python 3.4 or higher.

This code assumes that you already followed the default instructions from the 
:doc:`../install` guide; ``ngraph_dist`` was installed to ``$HOME/ngraph_dist``
and the `ngraph` repo was cloned to ``/opt/libraries/``

#. First set the environment variables to where we built the nGraph++ libraries:

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

#. Recursively update the submodule and install the Python dependencies: 

   .. code-block:: console

      $ git submodule update --init --recursive
      $ cd python
      $ pip3 install -r requirements.txt
      $ pip3 install .

#. Finally, clone the ``ngraph-onnx`` repo and use :command:`pip` to install the 
   Python dependencies for this tool; if you set up your original nGraph library 
   installation under a ``libraries`` directory    as recommended, it's a good 
   idea to clone this repo there, as well.
   
   .. code-block:: console

      $ cd /opt/libraries
      $ git clone git@github.com:NervanaSystems/ngraph-onnx
      $ cd ngnraph-onnx
      $ pip3 install -r requirements.txt
      $ pip3 install .
 

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


Enable ONNX and load an ONNX file from disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-19

 
Convert an ONNX model to an ngraph model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 22-23

 
Using ngraph_api, create a callable computation object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. if you work on NNVM you can add this instuction here. 



.. Importing models serialized with XLA
   -------------------------------------

.. if you work on XLA you can add this instruction here.


.. etc, eof 



.. _ONNX: http://onnx.ai
.. _tutorials from ONNX: https://github.com/onnx/tutorials
.. _CNTK: https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/