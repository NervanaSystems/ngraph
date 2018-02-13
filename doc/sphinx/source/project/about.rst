.. about: 

About
=====

Welcome to the Intel nGraph project, an open source C++ library for developers
of :abbr:`Deep Learning (DL)` (DL) systems. Here you will find a suite of 
components, APIs, and documentation that can be used to compile and run  
:abbr:`Deep Neural Network (DNN)` models defined in a variety of frameworks.  

.. figure:: ../graphics/ngraph-hub.png  

The nGraph library translates a framework’s representation of computations into 
an :abbr:`Intermediate Representation (IR)` designed to promote computational 
efficiency on target hardware. Initially-supported backends include Intel 
Architecture CPUs, the Intel® Nervana Neural Network Processor™ (NNP), 
and NVIDIA\* GPUs. Currently-supported compiler optimizations include efficient 
memory management and data layout abstraction. 

The *nGraph core* uses a strongly-typed and platform-neutral stateless graph 
representation for computations. Each node, or *op*, in the graph corresponds
to one step in a computation, where each step produces zero or more tensor
outputs from zero or more tensor inputs.

There is a *framework bridge* for each supported framework which acts as 
an intermediary between the *ngraph core* and the framework. A *transformer* 
plays a similar role between the ngraph core and the various execution 
platforms.

Transformers compile the graph using a combination of generic and 
platform-specific graph transformations. The result is a function that
can be executed from the framework bridge. Transformers also allocate
and deallocate, as well as read and write tensors under direction of the
bridge.
  
We developed Intel nGraph to simplify the realization of optimized deep 
learning performance across frameworks and hardware platforms. You can
read more about design decisions and what is tentatively in the pipeline
for development in our `SysML conference paper`_.

.. _frontend: http://neon.nervanasys.com/index.html/
.. _SysML conference paper: https://arxiv.org/pdf/1801.08058.pdf