.. graph-basics:

#############
Graph Basics
#############

Overview
========

This section provides a brief overview of some concepts used in the nGraph 
Library. It also introduces new ideas regarding our unique departure from the 
first generation of deep learning software design. 

The historical dominance of GPUs at the beginning of the current 
:abbr:`DL (Deep Learning)` boom means that many framework authors made 
GPU-specific design decisions at a very deep level. Those assumptions created 
an "ecosystem" of frameworks that all behave essentially the same at the
framework's hardware abstraction layer: 

* The framework expects to own memory allocation. 
* The framework expects the execution device to be a GPU. 
* The framework expects complete control of the GPU, and that the device doesn't 
  need to be shared. 
* The framework expects that developers will write things in a `SIMT-friendly`_ 
  manner, thus requring only a limited set of data layout conventions.    
  
Some of these design decisions have implications that do not translate well to 
the newer or more demanding generation of **adaptable software**. For example, 
most frameworks that expect full control of the GPU devices experience their 
own per-device inefficiency for resource utilization whenever the system 
encounters a bottleneck. 

Most framework owners will tell you to refactor the model in order to remove the 
unimplemented copy, rather than attempt to run multiple models in parallel, or 
attempt to figure out how to build graphs more efficiently. In other words, if 
a model requires any operation that hasn't been implemented on GPU, it must wait 
for copies to propagate from the CPU to the GPU(s). An effect of this 
inefficiency is that it slows down the system. Data scientists who are facing a 
large curve of uncertainty in how large (or how small) the compute-power needs 
of their model will be, investing heavily in frameworks reliant upon GPUs may 
not be the best decision.  

Meanwhile, the shift toward greater diversity in deep learning **hardware devices** 
requires that these assumptions be revisited. Incorporating direct support for 
all of the different hardware targets out there, each of which has its own 
preferences when it comes to the above factors, is a very heavy burden 
on framework owners.

Adding the nGraph compiler to the system lightens that burden by raising the 
abstraction level, and by letting any hardware-specific backends make these 
decisions automatically. The nGraph Compiler is designed to be able to take into 
account the needs of each target hardware platform, and to achieve maximum 
performance.

This makes things easier on framework owners, but also (as new models are developed) 
on data scientists, who will not have to keep in mind nearly as many low-level 
hardware details when architecting their models with layers of complexity for 
anything other than a :abbr:`Just-in-Time (JIT)` compilation.     

While the first generation frameworks tended to need to make a tradeoff between 
being "specialized" and "adaptable" (the trade-off between training and inference), 
nGraph Library permits algorithms implemented in a DNN to be both specialized 
and adaptable. The new generation of software design in and around AI ecosystems 
can and should be much more flexible.   


* :ref:`framework_bridges`
* :ref:`about_transformers`
* :ref:`graph_shaping`
 


.. _framework_bridges:

Framework bridges
=================

In the nGraph ecosystem, a framework is what the data scientist uses to solve 
a specific (and usually large-scale) deep learning computational problem with 
the use of a high-level, data science-oriented language. 

A framework :term:`bridge` is a software layer (typically a plugin *for* or an 
extension *to* a framework) that translates the data science-oriented language 
into a compute-oriented language called a :term:`data-flow graph`. The bridge 
can then present the problem to the nGraph :abbr:`Abstraction Layer (AL)` which 
is responsible for execution on an optimized backend by performing graph 
transformations that replace subgraphs of the computation with more optimal 
(in terms of machine code) subgraphs. Throughout this process, ``ops`` represent 
tensor operations. 

Either the framework can provide its own graph of functions to be compiled and 
optimized via :abbr:`Ahead-of-Time (AoT)` compilation to send back to the 
framework, or an entity (framework or user) who requires the flexibility of 
shaping ops directly can use our graph construction functions to experiment with 
building runtime APIs for their framework, thus exposing more flexible multi-theaded compute 
power options to 

See the section on :doc:`howto/execute` for a detailed walk-through describing 
how this translation can be programmed to happen automatically via a framework. 


.. _about_transformers:

Transformer ops
================

A framework bridge may define its own bridge-specific ops, as long as they can be 
converted to transformer ops. This is usually achieved by them first being 
converted to core ops on the way. For example, if a framework has a 
``PaddedCell`` op, nGraph pattern replacement facilities can be used to convert 
it into one of our core ops.  More detail on transformer ops will be coming soon.  


.. _graph_shaping:

Graph shaping
=============

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
:math:`X`. So the value of the sum of two tensors is a tensor whose value at a 
coordinate is the sum of the elements' two inputs. Unlike many frameworks, it 
says nothing about storage or arrays.

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
==========

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


.. _SIMT-friendly: https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads