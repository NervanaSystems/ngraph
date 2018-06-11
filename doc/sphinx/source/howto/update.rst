.. update.rst

###########################
Make a stateful computation
###########################

In this section, we show how to make a stateful computation from
nGraph's stateless operations. The basic idea is that any computation
with side-effects can be factored into a stateless function that
transforms the old state into the new state.

An example from C++
===================

Let's start with a simple C++ example, a function ``count`` that
returns how many times it has already been called:

.. literalinclude:: ../../../examples/update/update.cpp
   :language: cpp
   :lines: 20-24
   :caption: update.cpp

The static variable ``counter`` provides state for this function. The
state is initialized to 0. Every time ``count`` is called, the current
value of ``counter`` is returned and ``counter`` is incremented. To
convert this to use a stateless function, define a function that
takes the current value of ``counter`` as an argument and returns the
updated value.

.. literalinclude:: ../../../examples/update/update.cpp
   :language: cpp
   :lines: 26-29

To use this version of counting,

.. literalinclude:: ../../../examples/update/update.cpp
   :language: cpp
   :lines: 36-48

Update in nGraph
================

In working with nGraph-based construction of graphs, updating takes 
the same approach. During training, we include all the weights as 
arguments to the training function and return the updated weights 
along with any other results. For more complex forms of training, 
such as those using momentum, we would add the momementum tensors 
as additional arguments and include their updated values as additional 
results. A simple case is illustrated in the documentation for how 
to :doc:`derive-for-training`.
