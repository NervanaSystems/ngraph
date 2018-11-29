.. framework-integration-guides:

###############################
Integrate Supported Frameworks
###############################

* :ref:`mxnet_intg`
* :ref:`tensorflow_intg`

A framework is "supported" when there is a framework :term:`bridge` that can be 
cloned from one of our GitHub repos and built to connect to nGraph device backends, 
all the while maintaining the framework's programmatic or user interface. Bridges 
currently exist for the TensorFlow\* and MXNet\* frameworks. 

.. figure:: graphics/bridge-to-graph-compiler.png
    :width: 733px
    :alt: JiT compiling of a computation

    :abbr:`Just-in-Time (JiT)` Compiling for computation

Once connected via the bridge, the framework can then run and train a deep 
learning model with various workloads on various backends using nGraph Compiler 
as an optimizing compiler available through the framework.  


.. _mxnet_intg:

MXNet\* bridge
===============

* See the README on `nGraph-MXNet`_ Integration for how to enable the bridge.

* Optional: For experimental or alternative approaches to distributed training
  methodologies, including data parallel training, see the MXNet-relevant sections
  of the docs on :doc:`distr/index` and :doc:`How to <howto/index>` topics like
  :doc:`howto/distribute-train`. 


.. _tensorflow_intg:

TensorFlow\* bridge
===================

See the `ngraph tensorflow bridge README`_ for how to install the `DSO`_ for the 
nGraph-TensorFlow bridge.



.. _nGraph-MXNet: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/README.md
.. _MXNet: http://mxnet.incubator.apache.org
.. _DSO: http://csweb.cs.wfu.edu/%7Etorgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
.. _being the fastest: https://github.com/soumith/convnet-benchmarks
.. _ngraph tensorflow bridge README: https://github.com/NervanaSystems/ngraph-tf
