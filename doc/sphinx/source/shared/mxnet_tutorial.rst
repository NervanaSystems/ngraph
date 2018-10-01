.. shared/mxnet_tutorial.rst

Accelerating subgraphs with MXNet and nGraph compiler
#####################################################

One framework working on advancing shared graph optimizations is Apache MXNet\* 
with its new `subgraph acceleration API`_, where Intel nGraph has been merged as 
an experimental backend.  

The Intel nGraph Compiler provides an in-memory abstraction layer for converting 
the mathematical representation of a DL model into an optimized-for-execution 
format that can be understood *by* and run *on* multiple hardware backends. A 
primary goal of this integration is to provide a seamless development and 
deployment experience to Data Scientists and Machine Learning Engineers. It 
simplifies development by providing a common API for framework and hardware 
developers to jointly optimize a Deep Learning model; deployment is simplified 
by bringing highly-optimized CPU performance to a wide range of MXNet models, 
especially in inference.

Future releases of the nGraph Library will improve training support and add 
other hardware backends currently under development, including Nvidia\* GPU, 
Intel GPU, and custom silicon like the Intel® Nervana™ NNP. Future releases are 
planned to work with pip installation, and to offer support for the MS Windows\* 
operating system.



Tutorial: Compiling MXNet with nGraph and running Resnet-18 Inference
=====================================================================

This tutorial supports compiling MXNet with nGraph's CPU backend.

To compile MXNet with nGraph, follow steps 1-3 in the ``NGRAPH_README`` file 
located in the `nGraph-MXNet`_ repository. When building MXNet, be sure to 
append the make command with ``USE_NGRAPH=1``

MXNet's build system will automatically download, configure, and build the 
nGraph library, then link it into ``libmxnet.so``. Once this is complete, we 
recommend building a python3 virtual environment for testing, and then 
install MXNet to the virtual environment:

.. code-block:: console

   python3 -m venv .venv
   . .venv/bin/activate
   cd python
   pip install -e .
   cd ../

Now we're ready to use nGraph to run any model on a CPU backend. Building MXNet 
with nGraph automatically enabled nGraph on your model scripts, and you 
shouldn't need to do anything special. If you run into trouble, you can disable 
nGraph by setting ``MXNET_SUBGRAPH_BACKEND=1``. If you do see trouble, please 
report it and we'll address it as soon as possible.

Running Resnet-18 Inference
---------------------------


.. TODO  copy the Resnet 18 model we're running to the /doc/examples directory and 
   document it as per previous examples.  




.. _subgraph acceleration API: https://cwiki.apache.org/confluence/display/MXNET/Unified+integration+with+external+backend+libraries
.. _nGraph-MXNet: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/NGRAPH_README.md