.. execute-cmp.rst

######################
Execute a computation
######################

This section explains how to manually perform the steps that would normally be 
performed by a framework :term:`bridge` to execute a computation. The nGraph 
library is targeted toward automatic construction; it is far easier for a 
processing unit (GPU, CPU, or an `Intel Nervana NNP`_) to run a computation than 
it is for a user to map out how that computation happens. Unfortunately, things 
that make by-hand graph construction simpler tend to make automatic construction 
more difficult, and vice versa.

Here we will do all the bridge steps manually. The :term:`model description` 
we're explaining is based on the :file:`abc.cpp` file in the ``/doc/examples/`` 
directory. We'll be deconstructing the steps that an entity (framework or 
user) must be able to carry out in order to successfully execute a computation:

* :ref:`define_cmp`
* :ref:`specify_bkd`
* :ref:`compile_cmp`
* :ref:`allocate_bkd_storage`
* :ref:`initialize_inputs`
* :ref:`invoke_cmp`
* :ref:`access_outputs`

The final code is at the :ref:`end of this page <all_together>`.


.. _define_cmp:

Define the computation
======================

To a :term:`framework`, a computation is simply a transformation of inputs to 
outputs. While a *framework bridge* can programmatically construct the graph 
from a framework's representation of the computation, graph construction can be 
somewhat more tedious for users. To a user, who is usually interested in 
specific nodes (vertices) or edges of a computation that reveal "what is 
happening where", it can be helpful to think of a computation as a zoomed-out 
and *stateless* dataflow graph where all of the nodes are well-defined tensor 
operations and all of the edges denote use of an output from one operation as 
an input for another operation.

.. TODO

.. image for representing nodes and edges of (a+b)*c


Most of the public portion of the nGraph API is in the ``ngraph`` namespace, so 
we will omit the namespace. Use of namespaces other than ``std`` will be 
namespaces in ``ngraph``. For example, the ``op::Add`` is assumed to refer to 
``ngraph::op::Add``.

A computation's graph is constructed from ops; each is a member of a subclass of 
``op::Op``, which, in turn, is a subclass of ``Node``. Not all graphs are 
computation, but all graphs are composed entirely of instances of ``Node``.  
Computation graphs contain only ``op::Op`` nodes.

We mostly use :term:`shared pointers<shared pointer>` for nodes, i.e.
``std::shared_ptr<Node>`` so that they will be automatically
deallocated when they are no longer needed. A brief summary of shared
pointers is given in the glossary.

Every node has zero or more *inputs*, zero or more *outputs*, and zero or more 
*attributes*.  The specifics for each ``type`` permitted on a core ``Op``-specific 
basis can be discovered in our :doc:`../ops/index` docs. For our 
purpose to :ref:`define a computation <define_cmp>`, nodes should be thought of 
as essentially immutable; that is, when constructing a node, we need to supply 
all of its inputs. We get this process started with ops that have no inputs, 
since any op with no inputs is going to first need some inputs.

``op::Parameter`` specifes the tensors that will be passed to the computation. 
They receive their values from outside of the graph, so they have no inputs. 
They have attributes for the element type and the shape of the tensor that will 
be passed to them.

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 26-29

Here we have made three parameter nodes, each a 32-bit float of shape ``(2, 3)`` 
using a row-major element layout.

We can create a graph for ``(a+b)*c`` by creating an ``op::Add`` node with inputs 
from ``a`` and ``b``, and an ``op::Multiply`` node from the add node and ``c``:

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 31-32

When the ``op::Add`` op is constructed, it will check that the element types and 
shapes of its inputs match; to support multiple frameworks, ngraph does not do 
automatic type conversion or broadcasting. In this case, they match, and the 
shape of the unique output of ``t0`` will be a 32-bit float with shape ``(2, 3)``. 
Similarly, ``op::Multiply`` checks that its inputs match and sets the element 
type and shape of its unique output.

Once the graph is built, we need to package it in a ``Function``:

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 35-36

The first argument to the constuctor specifies the nodes that the function will 
return; in this case, the product. A ``NodeVector`` is a vector of shared 
pointers of ``op::Node``.  The second argument specifies the parameters of the 
function, in the order they are to be passed to the compiled function. A 
``ParameterVector`` is a vector of shared pointers to ``op::Parameter``. 

.. important:: The parameter vector must include **every** parameter used in 
   the computation of the results.


.. _specify_bkd:

Specify the backend upon which to run the computation
=====================================================

For a framework bridge, a *backend* is the environment that can perform the 
computations; it can be done with a CPU, GPU, or an Intel Nervana NNP. A 
*transformer* can compile computations for a backend, allocate and deallocate 
tensors, and invoke computations.

Factory-like managers for classes of backend managers can compile a ``Function`` 
and allocate backends. A backend is somewhat analogous to a multi-threaded
process.

There are two backends for the CPU: the optimized ``"CPU"`` backend, which uses 
the `Intel MKL-DNN`_, and the ``"INTERPRETER"`` backend, which runs reference 
versions of kernels that favor implementation clarity over speed. The 
``"INTERPRETER"`` backend can be slow, and is primarily intended for testing.  

To select the ``"CPU"`` backend,

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 39-40


.. _compile_cmp:

Compile the computation 
=======================

Compilation triggers something that can be used as a factory for producing a 
``CallFrame`` which is a *function* and its associated *state* that can run 
in a single thread at a time. A ``CallFrame`` may be reused, but any particular 
``CallFrame`` must only be running in one thread at any time. If more than one 
thread needs to execute the function at the same time, create multiple 
``CallFrame`` objects from the ``ExternalFunction``.

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 43-44


.. _allocate_bkd_storage:

Allocate backend storage for the inputs and outputs
===================================================

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
the three parameters and the return value as follows:

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 46-51

Each tensor is a shared pointer to a ``runtime::TensorView``, the interface 
backends implement for tensor use. When there are no more references to the 
tensor view, it will be freed when convenient for the backend.

.. _initialize_inputs:

Initialize the inputs
=====================

Next we need to copy some data into the tensors.

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 53-60

The ``runtime::TensorView`` interface has ``write`` and ``read`` methods for 
copying data to/from the tensor.

.. _invoke_cmp:

Invoke the computation
======================

To invoke the function, we simply pass argument and resultant tensors to the 
call frame:

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 63


.. _access_outputs:

Access the outputs
==================

We can use the ``read`` method to access the result:

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 65-67

.. _all_together:

Put it all together
===================

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :caption: "The (a + b) * c example for executing a computation on nGraph"




.. _Intel MKL-DNN: https://01.org/mkl-dnn
.. _Intel Nervana NNP: https://ai.intel.com/intel-nervana-neural-network-processors-nnp-redefine-ai-silicon/
