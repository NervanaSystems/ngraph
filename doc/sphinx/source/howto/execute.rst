.. execute.rst

#######################
Executing a Computation
#######################

This section explains how to manually perform the steps that would normally be 
performed by a framework :term:`bridge` to execute a computation. In order to 
successfully run a computation, the entity (framework or user) must be able to 
do all of these things:

* :ref:`define_cmp`
* :ref:`specify_bkd`
* :ref:`compile_cmp`
* :ref:`allocate_bkd_storage`
* :ref:`initialize_inputs`
* :ref:`invoke_cmp`
* :ref:`access_outputs`


.. _define_cmp:

Define a Computation
====================

To a :term:`framework`, a computation is simply a transformation of inputs to 
outputs. To a user, a computation is a function whose body is a dataflow graph. 
While a *framework bridge* can programmatically construct the graph from a 
framework's representation of the computation, graph construction can be somewhat 
more tedious for users.  Since nGraph is targeted toward automatic construction, 
we deconstruct here how this happens. 

Most of the public portion of the nGraph API is in the ``ngraph`` namespace, so 
we will omit the namespace. Use of namespaces other than ``std`` will be 
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

Once the graph is built, we need to package it in a ``Function``:

.. code-block:: cpp

   auto f = make_shared<Function>(NodeVector{t1}, ParameterVector{a, b, c});

The first argument to the constuctor specifies the nodes that the
function will return; in this case, the product. A ``NodeVector`` is a
vector of shared pointers of ``op::Node``.  The second argument
specifies the parameters of the function, in the order they are to be
passed to the compiled function. A ``ParameterVector`` is a vector of
shared pointers to ``op::Parameter``. *The parameter vector must
include* **every** *parameter used in the computation of the results.*

.. _specify_bkd:

Specify the backend upon which to run the computation
=====================================================

A *backend* is an environment that can execute computations, such as
the CPU or an NNP. A *transformer* can compile computations for a
backend, allocate and deallocate tensors, and invoke computations.

The current selection process is showing signs of age, and will be
changed. The general idea is that there are factory-like managers for
classes of backend, Managers can compile a ``Function`` and allocate
backends. A backend is somewhat analogous to a multi-threaded
process.

There are two backends for the CPU, the optimized "CPU" backend, which
makes use of mkl-dnn, and the "INTERPRETER" backend which runs
reference versions of kernels where implementation clarity is favored
over speed. The "INTERPRETER" backend is mainly used for testing.

To select the "CPU" backend,

.. code-block:: cpp

   auto manager = runtime::Manager::get("CPU");
   auto backend = manager->allocate_backend();

.. _compile_cmp:

Compile the computation 
=======================

Compilation produces something misnamed an ``ExternalFunction``, which
is a factory for producing a ``CallFrame``, a function and associated
state that can run in a single thread at a time. A ``CallFrame`` may
be reused, but any particular ``CallFrame`` must only be running in
one thread at any time. If more than one thread needs to execute the
function at the same time, create multiple ``CallFrame`` objects from
the ``ExternalFunction``.

.. code-block:: cpp

   auto external = manager->compile(f);
   auto cf = backend->make_call_frame(external);

.. _allocate_bkd_storage:

Allocate backend storage for the inputs and outputs
===================================================

At the graph level, functions are stateless. They do
have internal state related to execution, but there is no user-visible
state. Variables must be passed as arguments. If the function updates
variables, it must return the updated variables.

To invoke a function, tensors must be provided for every input and
every output. At this time, a tensor used as an input cannot also be
used as an output. If variables are being updated, you should use a
double-buffering approach where you switch between odd/even
generations of variables on each update.

Backends are responsible for managing storage. If the storage is
off-CPU, caches are used to minimize copying between device and
CPU. We can allocate storage for the three parameters and return value
as follows:

.. code-block:: cpp

   auto t_a = backend->make_primary_tensor_view(element::f32, shape);
   auto t_b = backend->make_primary_tensor_view(element::f32, shape);
   auto t_c = backend->make_primary_tensor_view(element::f32, shape);
   auto t_result = backend->make_primary_tensor_view(element::f32, shape);

Each tensor is a shared pointer to a ``runtime::TensorView``, the
interface backends implement for tensor use. When there are no more
references to the tensor view, it will be freed when convenient for
the backend.

.. _initialize_inputs:

Initialize the inputs
=====================

Normally the framework bridge reads/writes bytes to the tensor,
assuming a row-major element layout. To simplify writing unit tests,
we have developed a class for making tensor literals. We can use these
to initialize our tensors:

.. code-block:: cpp

   copy_data(t_a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
   copy_data(t_b, test::NDArray<float, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());
   copy_data(t_c, test::NDArray<float, 2>({{1, 0, -1}, {-1, 1, 2}}).get_vector());

The ``test::NDArray`` needs to know the element type (``float``) and
rank (``2``) of the tensors, and figures out the shape during template
expansion.

The ``runtime::TensorView`` interface has ``write`` and ``read``
methods for copying data to/from the tensor.

.. _invoke_cmp:

Invoke the computation
======================

To invoke the function, we simply pass argument and result tensors to
the call frame:

.. code-block:: cpp

   cf->call({t_a, t_b, t_c}, {t_result});

.. _access_outputs:

Access the outputs
==================

We can use the ``read`` method to access the result:

.. code-block:: cpp

   float r[2,3];
   t_result->read(&r, 0, sizeof(r));



