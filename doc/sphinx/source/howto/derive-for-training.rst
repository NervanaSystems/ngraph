.. derive-for-training.rst

#########################
Derive a trainable model 
#########################

Documentation in this section describes one of the ways to derive and run a
trainable model from an inference model.

We can derive a trainable model from any graph that has been constructed with 
weight-based updates. For this example named ``mnist_mlp.cpp``, we start with 
a hand-designed inference model and convert it to a model that can be trained 
with nGraph. 

Additionally, and to provide a more complete walk-through that *also* trains the 
model, our example includes the use of a simple data loader for uncompressed
MNIST data.

* :ref:`model_overview`
* :ref:`code_structure`

  - :ref:`inference`
  - :ref:`loss`
  - :ref:`backprop`
  - :ref:`update`


.. _understanding_ml_ecosystem:

Understanding the ML ecosystem
===============================

In a :abbr:`Machine Learning (ML)` ecosystem, it makes sense to take advantage 
of automation and abstraction as much as possible. As such, nGraph was designed 
to integrate wtih graph construction endpoints (AKA *ops*) handed down to it 
from a framework. Our graph-construction API, therefore, needs to operate at a 
fundamentally lower level than a typical framework's API. For this reason, 
writing a model directly in nGraph would be somewhat akin to programming in 
assembly language: not impossible, but not exactly the easiest thing for humans 
to do. 


.. _model_overview:

Model overview 
===============

Due to the lower-level nature of the graph-construction API, the example we've 
selected to document here is a relatively simple model: a fully-connected 
topology with one hidden layer followed by ``Softmax``.

Remember that in nGraph, the graph is stateless; values for the weights must
be provided as parameters along with the normal inputs. Starting with the graph
for inference, we will use it to create a graph for training. The training
function will return tensors for the updated weights. 

.. note:: This example illustrates how to convert an inference model into one 
   that can be trained. Depending on the framework, bridge code may do something 
   similar, or the framework might do this operation itself. Here we do the 
   conversion with nGraph because the computation for training a model is 
   significantly larger than for inference, and doing the conversion manually 
   is tedious and error-prone.


.. _code_structure:

Code structure
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

We want to reduce the loss by adjusting the weights. We compute the
adjustments using the reverse mode autodiff algorithm, commonly
referred to as "backprop" because of the way it is implemented in
interpreted frameworks. In nGraph, we augment the loss computation
with computations for the weight adjustments. This allows the
calculations for the adjustments to be further optimized.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 172


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

