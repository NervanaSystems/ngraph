.. graph-basics:

Graph Basics

*Tensors* are maps from *coordinates* to
scalar values, all of the same type, called the *element type*
of the tensor.
Coordinates are tuples of non-negative integers;
all the coordinates for a tensor have the same length, called
the *rank* of the tensor. We often use :math:`n`-tensor for
tensors with rank :math:`n`.
The *shape* of a tensor is a tuple
of non-negative integers that is an exclusive upper bound for
coordinate values. A tensor has an element for every coordinate 
less than the shape, so the *size* of the tensor is the product
of the values in the shape.

An :math:`n`-dimensional array is a common implementation of a
tensor, and the two terms are often used interchangeably, but 
a tensor could just as easily be a function that returns 0
for every coordinate.

A graph function describes a computation whose inputs and outputs are all 
tensors. The graph is a composition of simpler
tensor computations, called *ops*, which are nodes in the graph.
Every op has zero or more inputs and zero or more outputs which
represent tensors that will be provided during execution. In the graph,
every op input must be associated with an op output, and every op
output must have a constant element type and shape that will
correspond to the tensors used in the computation.
=======
=============

Ops
---

The graph is a composition of tensor computations, called ``ops``, which are 
nodes in the graph. In the graph, every :term:`op` *input* must be associated 
with an op *output*, and every op output must have a constant element type and 
shape to correspond with the tensors used in the computation. Every op has zero 
or more inputs and zero or more outputs representing tensors that will be 
provided during execution. 

Tensors
-------

*Tensors* are maps from coordinates to scalar values, all of the same type, 
called the *element type* of the tensor. Coordinates are tuples of non-negative 
integers; all the coordinates for a tensor have the same length, called the 
*rank* of the tensor. We often use :math:`n`-tensor for tensors with rank 
:math:`n`. An :math:`n`-dimensional array is a common implementation of a tensor, 
and the two terms are often used interchangeably. However, a tensor could just 
as easily be a function that returns 0 for every coordinate.

The :term:`shape` of a tensor is a tuple of non-negative integers that represents an  
exclusive upper bound for coordinate values. A tensor has an element for every 
coordinate less than the shape, so the *size* of the tensor is the product of 
the values in the shape.

A graph function describes a computation whose inputs and outputs are all 
tensors. 
Ops may also have additional attributes that do not change during
execution.

Function definition begins with creating one or more ``Parameter`` ops,
which represent 
the tensors that will be supplied as arguments to the function.
Parameters have no inputs and attributes for the element type and 
shape of the tensor that will be provided as an argument.
The unique output of the ``Parameter`` will have the provided
element type and shape.

Constructed ops have element types and shapes for each of their outputs,
which are determined during op construction from the element types and
shapes associated with the inputs, as well as additional attributes of
the ops. For example, tensor addition is defined for two tensors of the
same shape and size and results in a tensor with the same element type
and shape:

.. math::

  (A+B)_I = A_I + B_I

Here, :math:`X_I` means the value of a coordinate :math:`I` 
for the tensor :math:`X`. So the value of sum of two tensors
is a tensor whose value at a coordinate is the sum of the
elements are that coordinate for the two inputs. Unlike many
frameowrks, it says nothing about storage or arrays.

An ``Add`` op is used to represent a tensor sum. To construct an Add op,
two op outputs are needed. For example, two parameters could be used,
or the same parameter twice. All outputs of constructed ops have
element types and shapes, so when the Add is constructed, it verifies
that the two outputs associated with its two inputs have the same
element type and shape and sets its output to have the same element
type and shape.

Since all nodes supplying outputs for inputs to a new node must exist
before the new node can be created, it is impossible to construct a 
cyclic graph. Furthermore, type-checking can be performed as the ops 
are constructed.

.. TODO add basic semantics 

.. important:: During graph building, most of the storage associated 
   with values is *implicit*. During compilation, *explicit* storage 
   will be assigned in the form *value descriptors*; this storage will 
   be referred to as the inputs and outputs of those calls.


Sources of values
-----------------

.. note:: The nGraph library includes a number of *built-in ops*. A :
   ref:`built-in op` is like a templated function in C++, in that it 
   can be used with a variety of argument types. Similarly, when the 
   types of each argument are known in a call, the op must be able to 
   verify that the arguments are compatible, and it must be able to 
   determine the ``type`` of the returned value. 

The function graph is strongly typed. Every source of a value in the graph 
must be associated with a type. In a graph, values come from many possible
sources: *literals*, *calls* to ops (built-in ops or user-defined ops AKA 
*functions*), and *parameters* of user-defined functions.  

#. *Literals* A value type is associated with each literal, and must be 
   consistent with the literal's value. 

#. *Calls* to **ops**. When called with appropriate arguments, an *op* 
   produces a return value. All arguments not fixed at compile time 
   must be values. In the nGraph API, the term :term:`parameter` refers 
   to what "stands in" for an argument in an ``op`` definition, and :term:`result` 
   refers to what "stands in" for the returned *value*. 
   
   For example, the ``add`` **op** is a built-in op with two run-time 
   parameters that **must have the same value type**. It produces a 
   result with the same value type as its parameters. 

   Another example of a built-in **op** is the ``tuple`` **op** which, has 
   zero or more run-time parameters of *arbitrary* value types and a result 
   whose type is the tuple type of the types of the parameters. 

   #. **Functions*** are user-defined ops.
      - A user-defined function is "external" if it can be called externally.
      - The result is a graph node that depends only on parameters.
     - The result's type of call to a function is determined from the types of the arguments.
     - Any external function interacting with the graph at the level of user-defined op must specify a type for each of its parameters.

#. *Parameters* of user-defined *functions* may also be a source of a graph's
   values. Externally-callable functions must specify a type for each parameter.


Building a Graph
================

The function graph is composed of instances of the class ``Node``. Nodes are
created by helpers described below. 

.. note:: method ``dependents()`` is a vector of nodes that must be computed 
   before the result of ``Node`` can be used.

User-defined functions
----------------------

When building a function graph with values derived from "custom" or user-defined 
functions, use the following syntax to: 

* create a user-defined function: ``make_shared<Function>()`` 

  * get the specified parameter of the function: \* method:``parameter(index)``

     * return the type: \* method ``type()``

     * set the type to `t`:  \* method ``type(ValueType t)``

     * set the type to a ``TensorViewType``: \* method ``type(ElementType element_type, Shape shape)`` 

  * get the function's result: \* method ``result()``

    * return the node providing the value:  \* method ``value()``

    * set the node that will provide the value: \* method ``value(Node node)``

Type methods are available as with parameters. A user-defined function is 
callable, and can be used to add a call to it in the graph.


Built-in Ops
------------

Calls to built-in ops are created with helper functions generally in the
``op`` namespace. Ops are generally callable singletons that build
calls. When building a function graph with built-in ops, 

- ``op::tuple()`` produces an empty tuple 
- to add a value to a tuple, use the overload ``Tuple(list<Value>)``
    * to add a value to the tuple operation: \* method ``push_back(value)`` 
    * to return the specified component, call  \* method ``get(index)``   
      - where ``index`` is a compile-time value.


Example
-------

::

    // Function with 4 parameters
    auto cluster_0 = make_shared<Function>(4);
    cluster_0->result()->type(element_type_float, Shape {32, 3});
    cluster_0->parameter(0)->type(element_type_float, Shape {Shape {7, 3}});
    cluster_0->parameter(1)->type(element_type_float, Shape {Shape {3}});
    cluster_0->parameter(2)->type(element_type_float, Shape {Shape {32, 7}});
    cluster_0->parameter(3)->type(element_type_float, Shape {Shape {32, 7}});
    auto arg3 = cluster_0->parameter(3);
    // call broadcast op on arg3, broadcasting on axis 1.
    auto broadcast_1 = op::broadcast(arg3, 1);
    auto arg2 = cluster_0->parameter(2);
    auto arg0 = cluster_0->parameter(0);
    // call dot op
    auto dot = op::dot(arg2, arg0);
    // Function returns tuple of dot and broadcast_1.
    cluster_0->result()->value(dot);

Defining built-in ops
=====================

This section is WIP.

Built-in ops are used for several purposes: 

- Constructing call nodes in the graph. 
  * Checking type-consistency of arguments 
  * Specifying the result type for a call 
- Indicating preliminary tensor needs
  * Index operations are aliased views 
  * Tuples are unboxed into tensor views 
  * Remaining ops given vectors of inputs and outputs 
- Constructing patterns that will match sub-graphs 
- Pre-transformer code generation 
- Debug streaming of call descriptions

The general ``Node`` class provides for dependents and node type. The
class ``Call`` subclasses ``Node``. Built-in op implementations can
subclass ``Call`` to provide storage for compile-time parameters, such
as broadcast indices.

The plan is that the abstract class ``Op`` will have methods to be
implemented by built-in ops. Each built-in op corresponds to a callable
singleton (in the ``ngraph::op`` namespace) that constructs the
appropriate ``Call``. As a singleton, the op can conveniently be used as
a constant in patterns. Call objects will be able to find their related
op.

