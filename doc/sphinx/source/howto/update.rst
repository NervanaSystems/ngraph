.. update.rst

###########################
Make a stateful computation
###########################

In this section, we show how to make a stateful computation from
nGraphs stateless operations. The basic idea is that any computation
with side-effects can be factored into a stateless function that
transforms the old state into the new state.

An Example from C++
===================

Let's start with a simple C++ example, a function ``count`` that
returns how many times it has already been called:

.. literalinclude:: ../../../examples/update.cpp
   :language: cpp
   :lines: 20-24

The static variable ``counter`` provides state for this function. The
state is initialized to 0. Every time ``count`` is called, the current
value of ``counter`` is returned and ``counter`` is incremented. To
convert this to use a stateless function, we make a function that
takes the current value of ``counter`` as an argument and returns the
updated value.

.. literalinclude:: ../../../examples/update.cpp
   :language: cpp
   :lines: 26-29

To use this version of counting,

.. literalinclude:: ../../../examples/update.cpp
   :language: cpp
   :lines: 36-48

Update in nGraph
================

We use the same approach with nGraph. During training, we include all
the weights as arguments to the training function and return the
updated weights along with any other results. If we are doing a more
complex form of training, such as using momentum, we would add the
momementum tensors are additional arguments and add their updated
values as additional results. The simple case is illiustrated in the
trainable model how to.
