.. about: 

About
=====

Intel nGraph library provides a suite of components to promote computational 
efficiency and enable portability of Deep Neural Network (DNN) models defined in 
a variety of frameworks. The library translates a framework's representation of 
computations into a neutral-:abbr:`Intermediate Representation (IR)` designed 
to run better on target hardware: both Intel and non-Intel platforms.

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
