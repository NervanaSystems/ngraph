.. shared/mxnet_tutorial.rst:


Working with subgraphs on MXNet
################################

One DL framework with advancing efforts on shared graph optimizations is Apache 
MXNet\*, where Intel has contributed efforts showing how to work with our
nGraph Compiler stack as an `experimental backend`_. Our approach provides more
opportunities to start working with different kinds of graph optimizations than 
would be available to the MXNet framework alone, for reasons outlined in our 
`features`_ documentation.

.. TODO :  link to latest on mxnet when/if they do this instead of linking to PR;
   keep in mind this will still work without being merged if you already use the
   ngraph-mxnet Github repo

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
planned to further simplify installation with ``pip``, and to offer support for 
the MS Windows\* operating system.



Tutorial: Compiling MXNet with nGraph and running Resnet-50-v2 Inference
========================================================================

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

Running Resnet-50-V2 Inference
------------------------------

To show a working example, we'll demonstrate how MXNet may be used to run 
Resnet-50 Inference. For ease, we'll consider the standard MXNet Resnet-50-V2 
model from the `gluon model zoo`_, and we'll test with latency at ``batch_size=1``. 
Note that the nGraph-MXNet bridge supports static graphs only (dynamic graphs 
are in the works); so for this example, we begin by converting the gluon model 
into a static graph. Also note that any model with a saved checkpoint can be 
considered a "static graph" in nGraph. For this example, we'll presume that the 
model has been deemed "trained".   

.. literalinclude:: ../../../examples/subgraph_snippets/mxnet-gluon-example.py
   :language: python
   :lines: 17-32


To load the model into nGraph, we simply bind the symbol into a Executor. 

.. literalinclude:: ../../../examples/subgraph_snippets/mxnet-gluon-example.py
   :language: python
   :lines: 34-35

At binding, the MXNet Subgraph API finds nGraph, determines how to partition 
the graph, and in the case of Resnet, sends the entire graph to nGraph for 
compilation. This produces a single call to an NNVM ``NGraphSubgraphOp`` embedded 
with the compiled model. At this point, we can test the model's performance.

.. literalinclude:: ../../../examples/subgraph_snippets/mxnet-gluon-example.py
   :language: python
   :lines: 40-48


.. _experimental backend: https://github.com/apache/incubator-mxnet/pull/12502
.. _features: http://ngraph.nervanasys.com/docs/latest/project/about.html
.. _gluon model zoo: https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/resnet.py#L499
.. _subgraph acceleration API: https://cwiki.apache.org/confluence/display/MXNET/Unified+integration+with+external+backend+libraries
.. _nGraph-MXNet: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/NGRAPH_README.md
