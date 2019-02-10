.. howto/index: 

Constructing Graphs
===================

.. toctree::
   :maxdepth: 1
   :caption: How to 

   execute.rst
   operator.rst
   update.rst
   derive-for-training.rst
   distribute-train.rst
   import.rst    

The "How to" articles in this section explain how to do specific tasks with 
nGraph components. The recipes are all framework agnostic; in other words, 
if an entity (framework or user) wishes to make use of target-based computational 
resources, it can either:

* Do the tasks programatically through a framework, or 
* Provide a serialized model that can be imported to run on one of the nGraph
  backends. 

.. note:: This section is aimed at intermediate-level developers. It assumes an
   understanding of the concepts in the previous sections. It does not assume 
   knowledge of any particular frontend framework. 
  
Since our primary audience is developers who are pushing the boundaries of deep 
learning systems, we go beyond the use of deep learning primitives, and include 
APIs and documentation for developers who want the ability to write programs 
that use custom backends. For example, we know that GPU resources can be useful 
backends for *some* kinds of algorithmic operations while they impose inherent 
limitations or slow down others. 

One of our goals with the nGraph library is to enable developers with tools to 
quickly build programs that access and process data from a breadth of edge and 
networked devices. This might mean bringing compute resources closer to edge 
devices, or it might mean programatically adjusting a model or the compute 
resources it requires, at an unknown or arbitrary time after it has been deemed 
to be trained well enough.

To get started, we've provided a basic example for how to :doc:`execute` a
computation that can run on an nGraph backend; this is analogous to a
framework bridge.  We also provide a larger example for training and
evaluating a simple MNIST MLP model.

For data scientists or algorithm developers who are trying to extract specifics 
about the state of a model at a certain node, or who want to optimize a model 
at a more granular level, we provide an example for how to :doc:`import` and 
run inference after it has been exported from a DL framework.    

This section is under development; we'll continually populate it with more 
articles geared toward data scientists, algorithm designers, framework developers, 
backend engineers, and others. We welcome ideas and contributions from the 
community.  

