.. execute-cmp.rst

######################
Execute a computation
######################

This section explains how to manually perform the steps that would normally be 
performed by a framework :term:`bridge` to execute a computation. nGraph graphs 
are targeted toward automatic construction; it is far easier for a processor 
(a CPU, GPU, or `purpose-built silicon`_) to execute a computation than it is 
for a human to map out how that computation happens. Unfortunately, things 
that make by-hand graph construction simpler tend to make automatic construction 
more difficult, and vice versa.

Nevertheless, it can be helpful to break down what is happening during graph 
construction. The documetation that follows explains two approaches frameworks 
can use to compile with nGraph operations:

* :ref:`Using complete shapes <scenario_one>`
* :ref:`Using partial shapes <scenario_two>`

The nGraph :abbr:`Intermediate Representation (IR)` uses a strong, dynamic 
type system, including static shapes. This means that at compilation, every 
tensor (or, equivalently, every node output) in the graph is assigned 
**complete shape information**; that is, one and only one shape. The static 
process by which this assignment takes place is called :term:`shape propagation`.

In the :ref:`first scenario <scenario_one>`, the :term:`model description` 
walk-through is based on the :file:`abc.cpp` code in the ``/doc/examples/abc`` 
directory, and it deconstructs the steps that must happen (either programmatically 
or manually) in order to successfully execute a computation given complete 
shape information.  

.. _scenario_one:

Scenario One: Using Complete Shapes
===================================

A step-by-step example of how a framework might execute with complete shape 
information is provided here. For a step-by-step example using dynamic 
shapes, see :ref:`scenario_two`.

* :ref:`define_cmp`
* :ref:`specify_backend`
* :ref:`compile_cmp`
* :ref:`allocate_backend_storage`
* :ref:`initialize_inputs`
* :ref:`invoke_cmp`
* :ref:`access_outputs`


.. _define_cmp:

Define the computation
----------------------

To a :term:`framework`, a computation is simply a transformation of inputs to 
outputs. While a :term:`bridge` can programmatically construct the graph 
from a framework's representation of the computation, graph construction can be 
somewhat more tedious when done manually. For anyone interested in specific 
nodes (vertices) or edges of a computation that reveal "what is happening where", 
it can be helpful to think of a computation as a zoomed-out and *stateless* 
:term:`data-flow graph` where all of the nodes are well-defined tensor 
operations and all of the edges denote use of an output from one operation as an 
input for another operation.

Most of the public portion of the nGraph API is in the ``ngraph`` namespace, so 
we will omit the namespace. Use of namespaces other than ``std`` will be 
namespaces in ``ngraph``. For example, the ``op::Add`` is assumed to refer to 
``ngraph::op::Add``. A computation's graph is constructed from ops; each is a 
member of a subclass of ``op::Op``, which, in turn, is a subclass of ``Node``. 
Not all graphs are computation, but all graphs are composed entirely of 
instances of ``Node``.  Computation graphs contain only ``op::Op`` nodes.

We mostly use :term:`shared pointers<shared pointer>` for nodes, i.e.
``std::shared_ptr<Node>``, so that they will be automatically deallocated when 
they are no longer needed. More detail on shared pointers is given in the 
glossary.

Every node has zero or more *inputs*, zero or more *outputs*, and zero or more 
*attributes*. 

The specifics for each ``type`` permitted on a core ``Op``-specific basis can be 
discovered in our :doc:`../../ops/index` docs. For our purpose to 
:ref:`define a computation <define_cmp>`, nodes should be thought of as essentially 
immutable; that is, when constructing a node, we need to supply all of its 
inputs. We get this process started with ops that have no inputs, since any op 
with no inputs is going to first need some inputs.

``op::Parameter`` specifes the tensors that will be passed to the computation. 
They receive their values from outside of the graph, so they have no inputs. 
They have attributes for the element type and the shape of the tensor that will 
be passed to them.

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 25-29

The above code makes three parameter nodes where each is a 32-bit float of 
shape ``(2, 3)`` and a row-major element layout.

To create a graph for ``(a + b) * c``, first make an ``op::Add`` node with inputs 
from ``a`` and ``b``, and an ``op::Multiply`` node from the add node and ``c``:

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 31-32

When the ``op::Add`` op is constructed, it will check that the element types and 
shapes of its inputs match; to support multiple frameworks, ngraph does not do 
automatic type conversion or broadcasting. In this case, they match, and the 
shape of the unique output of ``t0`` will be a 32-bit float with shape ``(2, 3)``. 
Similarly, ``op::Multiply`` checks that its inputs match and sets the element 
type and shape of its unique output.

Once the graph is built, we need to package it in a ``Function``:

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 35-36

The first argument to the constuctor specifies the nodes that the function will 
return; in this case, the product. An ``OutputVector`` is a vector of references to 
outputs of ``op::Node``.  The second argument specifies the parameters of the 
function, in the order they are to be passed to the compiled function. A 
``ParameterVector`` is a vector of shared pointers to ``op::Parameter``. 

.. important:: The parameter vector must include **every** parameter used in 
   the computation of the results.


.. _specify_backend:

Specify the backend upon which to run the computation
-----------------------------------------------------

For a framework bridge, a *backend* is the environment that can perform the 
computations; it can be done with a CPU, GPU, or `purpose-built silicon`_. A 
*transformer* can compile computations for a backend, allocate and deallocate 
tensors, and invoke computations.

Factory-like managers for classes of backend managers can compile a ``Function`` 
and allocate backends. A backend is somewhat analogous to a multi-threaded
process.

There are two backends for the CPU: the optimized ``"CPU"`` backend, which uses 
the `DNNL`_, and the ``"INTERPRETER"`` backend, which runs reference 
versions of kernels that favor implementation clarity over speed. The 
``"INTERPRETER"`` backend can be slow, and is primarily intended for testing. 
See the documentation on :doc:`runtime options for various backends <../../backends/index>` 
for additional details.

To continue with our original example and select the ``"CPU_Backend"``: 

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 38-39


.. _compile_cmp:

Compile the computation 
-----------------------

Compilation triggers something that can be used as a factory for producing a 
``CallFrame`` which is a *function* and its associated *state* that can run 
in a single thread at a time. A ``CallFrame`` may be reused, but any particular 
``CallFrame`` must only be running in one thread at any time. If more than one 
thread needs to execute the function at the same time, create multiple 
``CallFrame`` objects from the ``ExternalFunction``.


.. _allocate_backend_storage:

Allocate backend storage for the inputs and outputs
---------------------------------------------------

At the graph level, functions are stateless. They do have internal state related 
to execution, but there is no user-visible state. Variables must be passed as 
arguments. If the function updates variables, it must return the updated 
variables.

To invoke a function, tensors must be provided for every input and every output. 
At this time, a tensor used as an input cannot also be used as an output. If 
variables are being updated, you should use a double-buffering approach where 
you switch between odd/even generations of variables on each update.

Backends are responsible for managing storage. If the storage is off-CPU, caches 
are used to minimize copying between device and CPU. We can allocate storage for 
the three parameters and the return value.

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 41-46

Each tensor is a shared pointer to a :term:`Tensorview`, which is the interface 
backends implement for tensor use. When there are no more references to the 
tensor view, it will be freed when convenient for the backend. See the 
:doc:`../../backends/cpp-api` documentation for details on how to work 
with ``Tensor``.


.. _initialize_inputs:

Initialize the inputs
---------------------

Next we need to copy some data into the tensors.

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 48-55

The ``runtime::Tensor`` interface has ``write`` and ``read`` methods for 
copying data to/from the tensor.

.. _invoke_cmp:

Invoke the computation
----------------------

To invoke the function, we simply pass argument and resultant tensors to the 
call frame:

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 57-58


.. _access_outputs:

Access the outputs
------------------

We can use the ``read`` method to access the result:

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 60-77

.. _sshp:

Compiling with Complete Shape Information
-----------------------------------------

.. literalinclude:: ../../../../examples/abc/abc.cpp
   :language: cpp
   :linenos:
   :caption: "The (a + b) * c example for executing a computation on nGraph"


.. _scenario_two:

Scenario Two: Known Partial Shape
=================================

The :ref:`second scenario <scenario_two>` involves the use of dynamic tensors. 
A :term:`dynamic tensor` is a tensor whose shape can change from one "iteration" 
to the next. When a dynamic tensor is created, a framework :term:`bridge` might 
supply only *partial* shape information: it might be **all** the tensor 
dimensions, **some** of the tensor dimensions, or **none** of the tensor 
dimensions; furthermore, the rank of the tensor may be left unspecified. 
The "actual" shape of the tensor is not specified until some function writes 
some value to it. The actual shape can change when the value of the tensor 
is overwritten. It is the backend’s responsibility to set the actual shape. 
The :term:`model description` for the second scenario based on the 
:file:`partial_shape.cpp` code in the ``/doc/examples/dynamic_tensor`` 
directory, and it deconstructs the steps that must happen (either 
programmatically or manually) in order to successfully retreive shape data.

* :ref:`create_dyn_tensor`
* :ref:`call_graph_vw_`
* :ref:`dyn_ten_result`
* :ref:`kpsh`


Create and compile a graph where the provided info of shape ``x`` is ``(2,?)``:

.. literalinclude:: ../../../../examples/dynamic_tensor/partial_shape.cpp
   :language: cpp
   :lines: 35-40


.. _create_dyn_tensor:

Create a dynamic tensor
-----------------------

Create a dynamic tensor of shape ``(2,?)``

.. literalinclude:: ../../../../examples/dynamic_tensor/partial_shape.cpp
   :language: cpp
   :lines: 43-46

At this point, ``t_out->get_shape()`` would throw an exception, while 
``t_out->get_partial_shape()`` would return ``"(2,?)"``.


.. _call_graph_vw_:

Initialize input of shape
-------------------------

.. literalinclude:: ../../../../examples/dynamic_tensor/partial_shape.cpp
   :language: cpp
   :lines: 57-62

At this point, ``t_out->get_shape()`` would return ``Shape{2,3}``,
while ``t_out->get_partial_shape()`` would return ``"(2,?)"``.


.. _dyn_ten_result:

Get the result
--------------

.. literalinclude:: ../../../../examples/dynamic_tensor/partial_shape.cpp
   :language: cpp
   :lines: 64-80

At this point, ``t_out->get_shape()`` would return ``Shape{2,20}``,
while ``t_out->get_partial_shape()`` would return ``"(2,?)"``.


.. _kpsh:

Compiling with Known Partial Shape
----------------------------------

.. literalinclude:: ../../../../examples/dynamic_tensor/partial_shape.cpp
   :language: cpp
   :linenos:
   :caption: "Full code for compiling with dynamic tensors and partial shape"


.. _purpose-built silicon: https://www.intel.ai/nervana-nnp
.. _DNNL: https://intel.github.io/mkl-dnn/ 


