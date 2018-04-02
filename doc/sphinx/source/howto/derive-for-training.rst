.. derive-for-training.rst

#########################
Derive a trainable model 
#########################

Documentation in this section describes one of the ways to derive a trainable 
model from an inference model.

We can derive a trainable model from any graph that has been constructed with 
weight-based updates. For this example named ``mnist_mlp.cpp``, we start with a hand-designed inference model and convert it to a model that can be trained 
with nGraph. 

Additionally, and to provide a more complete walk-through that *also* trains the 
"trainable" model, our example includes the use of a simple data loader for the 
MNIST data.

* :ref:`model_overview`
* :ref:`code_structure`
  - :ref:`inference`
  - :ref:`loss`
  - :ref:`backprop`
  - :ref:`update`


.. _model_overview:

Model overview
==============

The nGraph API was designed for automatic graph construction under direction of 
a framework. Without a framework, the process is somewhat tedious, so the example 
selected is a relatively simple model: a fully-connected topology with one hidden 
layer followed by ``Softmax``.

Since the graph is stateless, parameters for the input and the variables must be
derived from something; in this case, they will be derived from the "original"
weights. The training function will then return the tensors for the updated 
variables. Note that this is not the same as *constructing* the training model 
directly, which would be significantly more work.   


.. _code_structure:

Code Structure
==============


.. _inference:

Inference
---------

We begin by building the graph, starting with the input parameter 
``X``. We define a fully-connected layer, including a parameter for
weights and bias.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 124-136


We repeat the process for the next layer, which we
normalize with a ``softmax``.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 138-149


.. _loss:

Loss
----

We use cross-entropy to compute the loss. nGraph does not currenty
have a cross-entropy op, so we implement it directly, adding clipping
to prevent underflow.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 151-166


.. _backprop:

Backprop
--------

To compute the updates, we need a computation that computes an
adjustment for the weights from an adjustment to the loss. In nGraph,
``loss`` is the computation that computes the loss; it is equivalent
to what some descriptions of the autodiff algorithm call "the tape."
If each step of a computation between a weight and the loss has a
derivative, we can use the reverse mode autodiff to compute an update
for the weight from an update for the loss; in fact, we can compute
updates for all the weights, sharing much of the update computation,
and this is what some frameworks do. But it is just as easy for us to
instead create the update computations for all of the weights, which
lets compilation optimize across steps in the computation.

We'll call the adjustment to the loss

.. code-block:: cpp

   auto delta = -learning_rate * loss;

For any node ``N``, if the update for ``loss`` is ``delta``, the
update computation for ``N`` will be given by the node

.. code-block:: cpp

   auto update = loss->backprop_node(N, delta);

The different update nodes will share intermediate computations. So to
get the updated value for the weights we just say:

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 168-178

.. _update:

Update
------

Since nGraph is stateless, we train by making a function that has the
original weights among its inputs and the updated weights among the
results. For training, we'll also need the labeled training data as
inputs, and we'll return the loss as an additional result.  We'll also
want to track how well we are doing; this is a function that returns
the loss and has the labeled testing data as input. Although we can
use the same nodes in different functions, nGraph currently does not
allow the same nodes to be compiled in different functions, so we
compile clones of the nodes.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 248-260

