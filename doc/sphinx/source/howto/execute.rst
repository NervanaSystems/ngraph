.. execute.rst

#######################
Executing a Computation
#######################

This section explains how to manually perform the steps that would normally be 
performed by a framework :term:`bridge` to execute a computation. In order to 
successfully run a computation, the entity (framework or user) must be able to 
do all of these things:

.. contents:: 

.. _define_cmp:

Define a Computation
====================

To a framework, a computation is simply a transformation of inputs to outputs.
To a user, a computation is a function whose body is a dataflow graph. A 
*framework bridge* would normally programmatically construct the graph from 
the framework's representation of the computation. Since nGraph is targeted 
toward automatic construction, manual construction can be somewhat tedious. 
Here we deconstruct how this happens. 

Most of the public portion of the nGraph API is in the ``ngraph`` namespace,
so we will omit the namespace. Use of namespaces other than ``std`` will be 
namespaces in ``ngraph``. For example, the ``op::Add`` refers to 
``ngraph::op::Add``.

A computation's graph is constructed from ops; each a member of a
subclass of ``op::Op``, which, in turn, is a subclass of ``Node``. Not
all graphs are computation, but all graphs are composed entirely of
instances of ``Node``.  Computation graphs are only contain ``op::Op``
nodes.

We mostly use shared pointers for nodes,
i.e. ``std::shared_ptr<Node>``. This allows nodes to be deallocated
when the are no longer referenced. The one exception to this rule is
that there can not be a circular path through shared pointers, as this
would prevent the reference counts from every going to 0.

Every node has zero or more inputs, zero or more outputs, and zero or
more attributes. For our purposes, nodes should be thought of as
essentially immutable, so when we construct a node, we need to supply
all of its inputs. Thus, to get this process started, we need some
nodes that have no inputs.

We use ``op::Parameter`` to specify the tensors that will be passed to
the computation. They receive their values from outside of the graph,
so they have no inputs. They have attributes for the element type
and the shape of the tensor that will be passed to them.

.. code-block:: cpp
	
   Shape s{2, 3};
   auto a = std::make_shared<op::Parameter>(element::f32, s);
   auto b = std::make_shared<op::Parameter>(element::f32, s);
   auto c = std::make_shared<op::Parameter>(element::f32, s);


Here we have made three parameter nodes, each a 32-bit float of shape
``(2, 3)`` using a row-major element layout.

We can create a graph for ``(a+b)*c)`` by creating an ``op::Add`` node
with inputs from ``a`` and ``b``, and an ``op::Multiply`` node from
the add node and ``c``:

.. code-block:: cpp

   auto t0 = std::make_shared<op::Add>(a, b);
   auto t1 = std::make_shared<op::Multiply(t0, c);

When the ``op::Add`` op is constructed, it will check that the element
types and shapes of its inputs match; to support multiple frameworks,
ngraph does not do automatic type conversion or broadcasting. In this
case, they match, and the shape of the unique output of ``t0`` will be
a 32-bit float with shape ``(2, 3)``. Similarly, ``op::Multiply``
checks that its inputs match and sets the element type and shape of
its unique output.


.. _specify_bkd:

Specify the backend upon which to run the computation
=====================================================

.. TODO
 
Describe how to specify nGraph++ backends.


.. _compile_cmp:

Compile the computation 
=======================

.. TODO

Describe how to compile a computation with nGraph++ ops/libs/etc. How to avoid
unnecessary compiler actions that might otherwise happen by default. 


.. _allocate_bkd_storage:

Allocate backend storage for the inputs and outputs
===================================================

.. TODO

Explain how transformer(s) do(es) neat things.


.. _initialize_inputs:

Initialize the inputs
=====================

.. TODO

Action that initializes inputs? 


.. _invoke_cmp:

Invoke the computation
======================

.. TODO

.. _access_output



Access the outputs
==================

.. TODO



