.. features.rst

.. _features:

Features
========

What follows here are a few notable features of the nGraph Compiler stack. 

..  as well as a brief illustration or demonstration of that feature.

* **Fusion** -- Fuse multiple ops to to decrease memory usage.
* **Data layout abstraction** -- Make abstraction easier and faster with nGraph 
  translating element order to work best for a given or available device.
* **Data reuse** -- Save results and reuse for subgraphs with the same input.
* **Graph scheduling** -- Run similar subgraphs in parallel via multi-threading.
* **Graph partitioning** -- Partition subgraphs to run on different devices to 
  speed up computation; make better use of spare CPU cycles with nGraph.
* **Memory management** -- Prevent peak memory usage by intercepting a graph 
  with or by a "saved checkpoint," and to enable data auditing.

