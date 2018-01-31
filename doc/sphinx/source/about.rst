.. about: 

About
=====

Welcome to the Intel nGraph project, an open source C++ library for developers
of :abbr:`Deep Learning (DL)` (DL) systems and frameworks. Here you will find 
a suite of components, documentation, and APIs that can be used with 
:abbr:`Deep Neural Network (DNN)` models defined in a variety of frameworks.  

The nGraph library translates a framework’s representation of computations into 
a neutral-:abbr:`Intermediate Representation (IR)` designed to promote 
computational efficiency on target hardware; it works on Intel and non-Intel 
platforms. 

.. figure:: graphics/fig.jpeg  

The *nGraph core* uses a strongly-typed and platform-neutral stateless graph 
representation for computations. Each node, or *op*, in the graph corresponds
to one step in a computation, where each step produces zero or more tensor
outputs from zero or more tensor inputs.

There is a *framework bridge* for each supported framework which acts as 
an intermediary between the *ngraph core* and the framework. A *transformer* 
plays a similar role between the ngraphcore and the various execution 
platforms.

Transformers compile the graph using a combination of generic and 
platform-specific graph transformations. The result is a function that
can be executed from the framework bridge. Transformers also allocate
and deallocate, as well as read and write, tensors under direction of the
bridge.

For this early |release| release, we provide framework integration guides 
to

* :ref:`mxnet_intg`,
* :ref:`tensorflow_intg`, and
* Try neon™ `frontend`_ framework for training GPU-performant models.
  
Integration guides for each of these other frameworks is tentatively
forthcoming and/or open to the community for contributions and sample
documentation:

* `Chainer`_, 
* `PyTorch`_, 
* `Caffe2`_, and 
* Frameworks not yet written (for algorithms that do not yet exist). 

.. _Caffe2: https://github.com/caffe2/
.. _PyTorch: http://pytorch.org/
.. _Chainer: https://chainer.org/
.. _frontend: http://neon.nervanasys.com/index.html/
