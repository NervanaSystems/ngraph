.. build-a-functiongraph:

Defining a function graph on the nGraph library
###############################################

Graph Basics
============

To build a function graph with the nGraph library, first understand the ways
that the library will handle graph values before and during compilation. Since
it can be fairly easy to confuse C++ terms with their counterparts in the 
``ngraph`` function (and with the lower-level C++ representations of those 
counterparts), we provide this reference.  

Descriptions of ngraph values
-----------------------------

-  *Element values* are integers, floats, etc.

   -  Each ``type`` of element value is described by an ``ElementType``.
   -  A C++ :cpp:type:`type` is required for referencing literals during
      compilation.
   -  The :cpp:type:`type`'s ``value`` may be represented differently in a 
      compiled compilation. For example, a 32-bit float can hold a 16-bit float.

-  A *value* in a graph is either a tensor view or a tuple.

   -  A **tensor view** is an indexed collection of element values, all of
      the same element type. An element value is not a graph value; a 0-rank 
      tensor holds one element value and serves the same purpose.
   -  A **tuple** is 0 or more values, which can consist of tuples and
      tensor views.

-  Analogous to the value are "value types", also defined recursively.

   -  **Tensor view types** These types describe indexed collections of
      primitive types. They are specified by a shape and an primitive
      type for the elements.

      .. TODO add Doxy links corresponding to these tensor view types'
         APIs or use the literalinclude better 

   -  **Tuple types** These are cartesian product types for tuples of
      tuples and tensors, described by a sequence of tuple types and
      tensor view types.

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

   - *functions* are user-defined ops.
     - A user-defined function is "external" if it can be called externally.   
     - The result is a graph node that  depends only on parameters. 
     - The result's ``type`` of a call to a function is determined from the 
       types of the arguments.

  .. important::  Any external function interacting with the graph 
     at the level of user-defined ``op`` must specify a type for each 
     of its parameters.   

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

     * set the type to `t`:  \* method ``type(ValueType t)`

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
      - where ``index``is a compile-time value.


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

