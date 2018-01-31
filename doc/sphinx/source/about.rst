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

For this early release, we provide framework integration guides to: 

* :ref:`mxnet_intg`
* :ref:`tensorflow_intg`

.. figure:: graphics/fig.jpeg  

A *framework bridge* for each supported framework acts as an intermediary 
between the *nGraph core* and the framework. A *transformer* plays a similar 
role to the various execution platforms. And the *nGraph core* uses a 
strongly-typed and platform-neutral stateless graph representation for 
computations. Each node, or :term:`op`, in the graph corresponds to one 
:term:`step` in a computation, where each step produces zero or more tensor 
outputs from zero or more tensor inputs.

Transformers compile the graph using a combination of generic and 
platform-specific graph transformations. The result is a function that
can be executed from the framework bridge. Under the direction of the 
bridge, transformers may also allocate and deallocate, as well as read 
and write tensors.
