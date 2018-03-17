.. about: 

About
=====

Welcome to Intel® nGraph™, an open source C++ library and compiler. This 
project enables modern compute platforms to run and train 
:abbr:`Deep Neural Network (DNN)` models. It is framework-neutral and supports 
a variety of backends used by :abbr:`Deep Learning (DL)` frameworks. 

.. figure:: graphics/ngraph-ecosystem.png
   :width: 585px
  
The nGraph library translates a framework’s representation of computations into 
an :abbr:`Intermediate Representation (IR)` designed to promote computational 
efficiency on target hardware. Initially-supported backends include Intel 
Architecture CPUs, the Intel® Nervana Neural Network Processor™ (NNP), 
and NVIDIA\* GPUs. Currently-supported compiler optimizations include efficient 
memory management and data layout abstraction. 

Why is this needed?
--------------------

When Deep Learning (DL) frameworks first emerged as the vehicle for training 
and inference models, they were designed around kernels optimized for a 
particular platform. As a result, many backend details were being exposed in 
the model definitions, making the adaptability and portability of DL models 
to other or more advanced backends inherently complex and expensive.

The traditional approach means that an algorithm developer cannot easily adapt 
his or her model to different backends. Making a model run on a different 
framework is also problematic because the user must separate the essence of 
the model from the performance adjustments made for the backend, translate 
to similar ops in the new framework, and finally make the necessary changes 
for the preferred backend configuration on the new framework.

We designed the Intel nGraph project to substantially reduce these kinds of 
engineering complexities. While optimized kernels for deep-learning primitives 
are provided through the project and via libraries like Intel® Math Kernel 
Library (Intel® MKL) for Deep Neural Networks (Intel® MKL-DNN), there are 
several compiler-inspired ways in which performance can be further optimized.

=======

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
.. _MXNet: http://mxnet.incubator.apache.org/
.. _TensorFlow: https://www.tensorflow.org/

