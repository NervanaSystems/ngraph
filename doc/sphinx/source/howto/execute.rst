.. execute.rst

#######################
Executing a Computation
#######################

In this section, we manually perform the steps that would normally be
performed by a framework bridge to execute a computation:

- Define the computation
- Specify the backend that will execute the computation
- Compile the computation
- Allocate backend storage for the inputs and outputs
- Initialize the inputs
- Invoke the computation
- Access the outputs

Defining a Computation
======================

A computation is described as a function whose body is a dataflow
graph.  In normal operation, a *framework bridge* would
programmatically construct the graph from the framework's
representation of the computation. Here we will do all the bridge
steps manually. Unfortunately, things that make by hand graph
construction simpler tend to make automatic construction more
difficult, and vice versa.  Since nGraph is targeted towards automatic
construction, manual construction is a little tedious.

Since most of the public portion of the nGraph API is in the ``ngraph``
namespace; we will omit the namespace. Use of namespaces other than
``std`` will be namespaces in ``ngraph``. For example, the ``op::Add``
refers to ``ngraph::op::Add``.

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
