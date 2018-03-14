.. handle.rst:

#################
Handle an Export 
#################

:ref:`from_onnx`


.. TODO Make sure that this is the first page data scientists find when they 
.. are simply trying to run a trained model; they DO NOT need to do a system
.. install of the Intel nGraph++ bridges; they can use our Python APIs to run 
.. a trained model. 

Intel nGraph APIs can be used to run inference on a model that has been *exported* 
from a framework. A model that has been determined to be trained "well enough" 
can be extracted from a framework and serialized. The entity extracting the model 
produces an :term:`export`, which usually is simply a serialized format that can 
be passed to one of the nGraph backends for computation.  


.. _from_onnx:

from ONNX
==========

The most common kind of :term:`export` available today is an `ONNX`_ export. 
Models that have been serialized by ONNX are easy to identify; they are usually 
named ``<some_model>.onnx``. These `tutorials from ONNX`_ describe how to turn
trained models into an ``.onnx`` export.  

If you landed on this page and you already have an ``.onnx`` formatted-file, you 
should be able to run the inference without needing to dig into anything from 
the "Frameworks" sections. You will, however, need to have completed the 
steps described in our :doc:`../install` guide.  

To demonstrate the functionality of the nGraph inference, we'll use an 
`already serialized CIFAR10`_ ResNet model for demonstration purposes. Remember 
that this model has already been trained and exported from some framework; we 
are simply going to build an nGraph representation of the model and produce some 
outputs.  


#. First set the environment variables to where we built the nGraph++ libraries;
   This code assumes that you followed the default instructions from the 
   :doc:`../install` guide and ``ngraph_dist`` can be found at ``$HOME/ngraph_dist``:

   .. code-block:: bash

      export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib
      export DYLD_LIBRARY_PATH=$HOME/ngraph_dist/lib  # On MacOS

#. Now add the following dependencies to your system:

   .. code-block:: console

      $ sudo apt-get update && install -y protobuf-compiler libprotobuf-dev python3-pip 
      $ sudo apt-get clean autoclean

#. Checkout the branch named `python_binding`: 

   .. code-block:: console

      $ cd /opt/libraries/ngraph-cpp
      $ git checkout python_binding
        Switched to branch 'python_binding'
        Your branch is up-to-date with 'origin/python_binding'.       

#. Recursively update the submodule 

   .. code-block:: console

      $ git submodule update --init --recursive
      $ cd /path/to/ngraph/python
	   $ pip3 install -r requirements.txt
	   $ pip3 install .


#. Clone the ``ngraph-onnx`` repo and pip install the Python dependencies; if 
   you set up your original nGraph installation under a ``libraries`` 
   directory as recommended, it's a good idea to clone this repo there, too.
   
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

 
Convert ONNX model to an ngraph model
--------------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 22-23

 
Create a callable computation
-----------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 27-29


Load or create an image
------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 32-33

Run ResNet inference on picture
-------------------------------

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 35
 

Put it all together
===================

.. literalinclude:: ../../../examples/onnx_example.py
   :language: python
   :lines: 17-36
   :caption: "Code to run inference on a CIFAR10 trained model"



.. From NNVM
   ----------

.. From XLA
   --------

.. etc, eof 



.. _ONNX: http://onnx.ai
.. _tutorials from ONNX: https://github.com/onnx/tutorials
.. _already serialized CIFAR10: https://github.com/NervanaSystems/ngraph-onnx-val/tree/master/models

