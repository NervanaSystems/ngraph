.. graph-basics:

############
Graph Basics
############


Concepts
========

This section explains some of the primary concepts needed to understand graph
construction. 


Framework Bridges
-----------------

In the nGraph ecosystem, a framework is any "frontend" that helps a data 
scientist model or simulate a specific (usually large-scale) computational 
problem through the use of a high-level, data science-oriented language.  

A framework :term:`bridge` is a concept we use in nGraph to describe how that 
computational problem gets presented to the nGraph 
:abbr:`Intermediate Representation (IR)` for execution on an nGraph backend. 
On the nGraph side, we perform graph transformations that replace subgraphs 
of the computation with more optimal (in terms of machine code) subgraphs. 
Throughout this process, ``ops`` represent tensor operations. 

Either the framework can provide its own graph of functions to compiled and 
optimized via :abbr:`Ahead-of-Time (AoT)` compilation to send back to the 
framework, or an entity (framework or user) who requires the flexibility of 
shaping ops directly can use our graph construction functions to experiment with 
building runtime APIs for their framework. 

See the section on :doc:`howto/execute` for a detailed walk-through describing 
how this translation can be programmed to happen automatically via a framework. 

Transformers
------------

A framework bridge can use bridge-specific ops, as long as they can be 
converted to transformer ops, usually by being converted to core ops on 
the way. For example, a framework might have a `PaddedCell` op and use 
nGraph pattern replacement facilities to convert that into one of our core 
ops. 


Tensors
-------

*Tensors* are maps from coordinates to scalar values, all of the same
type, called the *element type* of the tensor. Coordinates are tuples
of non-negative integers; all the coordinates for a tensor have the
same length, called the *rank* of the tensor. We often use
:math:`n`-tensor for tensors with rank :math:`n`.

The :term:`shape` of a tensor is a tuple of non-negative integers that 
represents an exclusive upper bound for coordinate values. A tensor has an 
element for every coordinate less than the shape, so the *size* of the tensor 
is the product of the values in the shape.

An :math:`n`-dimensional array is the usual implementation for a
tensor, and the two terms are often used interchangeably, but a tensor
could just as easily be represented by a function that returns 0 for
every coordinate or a function that adds the elements of two other
tensors at the same coordinate and returns that sum.

Ops
---

A computation graph is a composition of tensor computations, called
``ops``, which are nodes in the graph. In the graph, every :term:`op`
*input* must be associated with an op *output*, and every op output
must have a fixed element type and shape to correspond with the
tensors used in the computation. Every op has zero or more inputs and
zero or more outputs.  The outputs represent tensors that will be
provided during execution. Ops may also have additional attributes
that do not change during execution.

Every `op` is a `Node`, but not all nodes are ops. This is because
pattern graphs are another kind of graph that includes ops combined
with nodes that describe how to match subgraphs during graph
optimization.

Constructed ops have element types and shapes for each of their outputs, which 
are determined during op construction from the element types and shapes 
associated with the inputs, as well as additional attributes of the ops. For 
example, tensor addition is defined for two tensors of the same shape and size 
and results in a tensor with the same element type and shape:

.. math::

  (A+B)_I = A_I + B_I

Here, :math:`X_I` means the value of a coordinate :math:`I` for the tensor 
:math:`X`. So the value of sum of two tensors is a tensor whose value at a 
coordinate is the sum of the elements are that coordinate for the two inputs. 
Unlike many frameworks, it says nothing about storage or arrays.

An ``Add`` op is used to represent an elementwise tensor sum. To
construct an Add op, each of the two inputs of the ``Add`` must be
assigned some output of some already-created op. All outputs of
constructed ops have element types and shapes, so when the Add is
constructed, it verifies that the two input tensors have the same
element type and shape and then sets its output to have the same
element type and shape.

Since all nodes supplying outputs for inputs to a new node must exist
before the new node can be created, it is impossible to construct a
cyclic graph.  Furthermore, type-checking is performed as the ops are
constructed.


Functions
---------

Ops are grouped together in a ``Function``, which describes a
computation that can be invoked on tensor arguments to compute tensor
results. When called by a bridge, the bridge provides tensors in the
form of row-major arrays for each argument and each computed
result. The same array can be used for more than one argument, but
each result must use a distinct array, and argument arrays cannot be
used as result arrays.

Function definition begins with creating one or more ``Parameter``
ops, which represent the tensors that will be supplied as arguments to
the function.  Parameters have no inputs and attributes for the
element type and shape of the tensor that will be provided as an
argument. The unique output of the ``Parameter`` will have the
provided element type and shape.

A ``Function`` has ``Parameters``, a vector of ``Parameter`` ops,
where no ``Parameter`` op may appear more than once in the vector.  A
``Parameter`` op has no inputs and attributes for its shape and
element type; arrays passed to the function must have the same shape
and element type as the corresponding parameter.  The ``Function``
also has ``Nodes``, a vector of ops that are the results being
computed.

During execution, the output of the nth ``Parameter`` op will be the tensor
corresponding to the array provided as the nth argument, and the outputs
of all result ops will be written into the result arrays in row-major
order.


An Example
----------

::

   #include <memory>
   #include <ngraph.hpp>

   using ngraph;

   // f(a, b, c) = (a + b) * c
   void make_function()
   {

       // First construct the graph
       Shape shape{32, 32};
       auto a = std::make_shared<op::Parameter>(element::f32, shape);
       auto b = std::make_shared<op::Parameter>(element::f32, shape);
       auto c = std::make_shared<op::Parameter>(element::f32, shape);
       auto t0 = std::make_shared<op::Add>(a, b);
       auto t1 = std::make_shared<op::Multiply>(t0, c);

       auto f = std::make_shared<Function>(Nodes{t1}, Parameters{a, b, c});
   }

We use shared pointers for all ops. For each parameter, we need to
element type and shape attributes. When the function is called, each
argument must conform to the corresponding parameter element type and
shape.

During typical graph construction, all ops have one output and some
number of inputs, which makes it easy to construct the graph by
assigning each unique output of a constructor argument node to an
input of the op being constructed.  For example, `Add` need to supply
node outputs to each of its two inputs, which we supply from the
unique outputs of the parameters `a` and `b`.

We do not perform any implicit element type coercion or shape
conversion (such as broadcasts) since these can be
framework-dependent, so all the shapes for the add and multiply must
be the same. If there is a mismatch, the constructor will throw an
exception.

After the graph is constructed, we create the function, passing the
`Function` constructor the nodes that are results and the parameters
that are arguments.

