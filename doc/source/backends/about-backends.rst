.. about-backends.rst

About Backends
##############


The Intel® nGraph™ library provides a number of options for running computations
on hardware-specific backends. 


The term *transformer* refers to all operations related to running computations: 
from compilation to execution. These operations are defined by a "graph" on a 
backend; or more specifically they represent a function graph. 

Since any subset of the graph represents a potential computation, the graph 
should be thought of as a template for computations related to some particular 
model. For example, the core of a model is its inference graph. An optimizer 
extends the inference graph by adding the derivative computations and variable 
updates. The extended graph includes the computations used for both inference 
and for training.

The transformer method ``computation`` is used to specify a subgraph to be 
computed. This method needs one or more result graph nodes that need to be 
computed; their values will be returned when the computation is executed. The 
computation also needs a parameter list of nodes that will receive the arguments 
when the computation is called. This parameter list must include all the 
``placeholder`` nodes that contribute to the computed nodes. You can include 
include additional nodes, such as variables, by setting the ``is_input`` 
attribute to ``True``. The ``computation`` method returns a function that 
expects tensors for each parameter and returns tensors for each value-producing 
result node.

Transformers are currently provided for the following backends:

- CPUs (via NumPy and MKL-DNN)
- NVIDIA* GPUs (via PyCUDA)

Transformer creation
====================

You should create transformers using the factory interface in 
``ngraph.transformers.base``, as shown:

.. code-block:: python

    from ngraph.transformers import make_transformer
    transformer = make_transformer()

This creates a transformer using the default transformer factory (CPU). You can 
then manually set the transformer factory to control the target backend. The 
transformer API provides functionality to enumerate the available transformers 
to assist with this, as shown below:

.. code-block:: python

    import ngraph.transformers as ngt
    available_transformers = ngt.transformer_choices()
    if 'gpu' in available_transformers:
        factory = ngt.make_transformer_factory('gpu')
        ngt.set_transformer_factory(factory)

    transformer = ngt.make_transformer()

The example above first checks whether the GPU transformer is available 
(which depends on CUDA and PyCUDA being installed). If the GPU transformer **is** 
available, this example sets the transformer factory to generate GPU transformers. 
The call to ``make_transformer`` then returns a GPU transformer if one is 
available. Otherwise, the ``make_transformer`` call returns a CPU transformer.

Computations
============

Computation objects are created by the transformer and provide an interface to 
evaluate a subset of the graph. The format of the executable used for evaluation 
depends on which transformer created the computation object. For example, the CPU 
transformer generates Python NumPy code that is called to evaluate the 
computation, while the GPU transformer generates a series of CUDA kernels that 
can be called to evaluate the computation.

Computation creation
--------------------

Computations are created with the ``Transformer.computation`` method. When 
creating a computation, users must specify a list of results that should be 
evaluated by the computation. These results should be Intel® Nervana™ ``Op`` s. 
The transformer can traverse the graph backwards from these results to determine 
the entire subset of graph nodes that are required to evaluate these results. 
This means that is not necessary for users to specify the entire subset of nodes 
to execute. However, users must set a list of graph nodes as inputs to the 
computation. Typically these are placeholder tensors. Continuing from the code 
example above, let's create a simple graph and computation:

.. code-block:: python

    import ngraph as ng

    a = ng.constant(4)
    b = ng.placeholder(())
    c = ng.placeholder(())
    d = ng.multiply(a, b)
    e = ng.add(d, c)

    example_comp = transformer.computation(e, b, c)

This example creates a simple graph to evaluate the function ``e = ((a * b) + c)``. 
The first argument is the result of the computation, and the remaining arguments 
are inputs to the computation. To create the computation, the only result that 
we need to specify is ``e``, since ``d`` is discovered when the transformer 
traverses the graph. In this example, ``a`` is a constant so it does not need to 
be passed in as an input, but ``b`` and ``c`` are placeholder tensors that must 
be specified as inputs.

After any computation is created, the ``Transformer.initialize`` method must be 
called to finalize transformation and allocate all device memory for tensors 
(the ``Transformer.initialize`` method is called automatically if a computation 
is called before you manually call ``initialize``). 

Computation Execution
---------------------

Our example computation object can be executed with its ``__call__`` method by 
specifying the inputs ``b`` and ``c``.

.. code-block:: python

    result_e = example_comp(2, 7)

The return value of this call is the resulting value of ``e``, which should 
be ((4 * 2) + 7) = 15.

Computations with multiple results
----------------------------------

In real-world use cases, the goal is often to create computations that return 
multiple results. For example, a single training iteration might compute both 
the cost value and the weight updates. Multiple results can be passed to 
computation creation in a list. After execution, the computation returns a tuple 
of the results:

.. code-block:: python

    example_comp2 = transformer.computation([d, e], b, c)
    result_d, result_e = example_comp2(2, 7)

In addition to returning the final result as seen above, this example also sets 
``result_d`` to the result of the ``d`` operation, which should be 8.

Transformer/Backend state
-------------------------

A computation is compiled and installed on the backend device the first time the 
computation is called. Any new persistent tensors (such as variables) will be 
initialized at this time. Persistent tensors that were also used in previously 
defined computations will retain their states unless they have been listed among 
the computation's arguments. If some persistent tensors are listed among the 
computation's arguments, their values will be set when the computation is 
invoked. For example, variables updated by a training computation will retain 
their values for an inference computation. You can manually save variables by 
defining a computation that returns their values, and can store variables by 
using them as arguments for a computation.


Transformer implementation
===========================

This section gives an overview of how the base transformer and computation are 
implemented, using the CPU and GPU transformers as examples.


Transformer creation
--------------------

The base transformer constructor initializes a set of all computations and 
results associated with the transformer. These sets are populated as computation 
objects are created. Additionally the transformer constructor can build a list 
of passes to run on the op graph when initialization and transformation is 
executed.

Specific transformer implementations may use the constructor to initialize code 
generators (as in the CPU transformer) or initialize the target device and 
determine device capabilities (as for the GPU transformer).

Computation creation
--------------------

To create a computation, users call the transformer's ``computation`` method. 
This is a relatively lightweight operation that creates a new ``Computation`` 
object and stores it in the set of all computations. The ``Computation`` 
constructor updates the transformer's set of results and builds a set of all ops 
that are dependencies of the results by traversing the graph backwards from the 
result nodes. 

Computations can only be created through a transformer before the transformer 
has been initialized. This is partially because the transformer in its current 
state modifies the graph through passes and tensor description initialization, 
but this will likely change in the future.

Transformer initialization
==========================

The ``Transformer.initialize`` method of the transformer is responsible for 
running passes to augment the graph, generating code or executables to evaluate 
ops, allocating buffers and tensors, and initializing tensors. This method can 
be manually called by users, but will be automatically called upon the first 
evaluation of a computation if the user has not manually called it.

Passes and op transformation are called from 
``Transformer._transform_computations``. Device buffer and tensor allocation are 
called from ``Transformer.allocate_storage`` which must be implemented by each 
transformer. Constant tensor initialization is called from 
``Transformer.allocate``, and other initialization is performed in a special 
computation called by ``Transformer.initialize``.

.. _transformer_passes:

Transformer Passes
------------------

Transformer passes are run in ``Transformer._transform_computations`` here:

.. code-block:: python

    def _transform_computations(self):
        """
        Transform computation graphs to a form that can be run.
        """

        # Run passes on the computation graphs
        self.run_registered_graph_passes(self.all_results)

Transformer passes are used to replace ops in the graph, remove ops from the 
graph, or splice ops into the graph. These passes can be used to simplify the 
graph (see ``SimplePrune`` for an example). Passes can also be used to alter the 
graph to meet device-specific constraints or to optimize ops for exection on 
specific devices. Currently the only pass that falls into this category is the 
``CPUTensorShaping`` pass, which reduces the dimensionality of tensors to 2D 
for reduction elementwise operations and 1D for all other elementwise operations. 
This pass simplifies the requirements placed on code generation to handle 
multidimensional tensors. In the future, we will likely make this device 
specific (for example, the GPU kernel generator can handle up to three 
dimensions efficiently).

All passes inherit from the ``GraphPass`` class, which requires that a child 
class implements ``do_pass``. Currently all implemented passes are instances of 
``PeepholeGraphPass``. A peephole graph pass is a specific type of pass that 
traverses the graph one node at a time, calling ``PeepholeGraphPass.visit`` on 
each node and which builds a mapping of ops to be replaced. Implementors can 
define ``visit`` methods for relevant op types and call 
``PeepholeGraphPass.replace_op`` to replace the visited op with another op. An 
example from the ``SimplePrune`` pass is shown below:

.. code-block:: python

    @visit.on_type(Add)
    def visit(self, op):
        x, y = op.args
        rep = None
        if x.is_scalar and x.is_constant:
            if x.const == 0:
                rep = y
        elif y.is_scalar and y.is_constant:
            if y.const == 0:
                rep = x
        if rep is not None:
            self.replace_op(op, rep)

The first line stating 

.. code-block:: python 

    @visit.on_type(Add) 

indicates that this method will be called when an ``Add`` op is encountered 
during graph traversal. This implementation of ``visit`` checks if either of 
the arguments to ``Add`` is 0. Since adding 0 to a value is essentially a 
no-op, an op meeting this condition can be replaced with its nonzero argument.

Passes presented a major opportunity for performance optimization that we make
extensive use of in the nGraph library. Examples include device-specific fusion 
(of operations) that allows generation of kernels to execute multiple ops at 
once, as well as buffer sharing that will allow nonoverlapping operations to 
share device memory. These passes would improve execution time and memory usage, 
respectively. We currently have machinery for doing fusion and buffer sharing in 
``ngraph.analysis``, but it is not working in the prerelease and has been 
disabled until we can refactor it to utilize passses.

Tensor description initialization
---------------------------------

Tensor descriptions for all ops are initialized in ``Transformer.initialize_tensor_descriptions``. This calls into the transformer to create ``DeviceBufferStorage`` and ``DeviceTensor`` instances for each op. Each transformer must define implementations of ``DeviceBufferStorage`` and ``DeviceTensor``.

The ``DeviceBufferStorage`` class represents a memory allocation on the transformer's device (for example, this will be allocated with PyCUDA for the GPU transformer). This buffer can be used as storage by one or more tensors. When a ``DeviceBufferStorage`` object is created, the buffer is not allocated yet, but the object is added to the ``Transformer.device_buffers`` member for later allocation.

The ``DeviceTensor`` class represents a tensor view on top of a device memory allocation, including a base address offset, shape, strides, and data type. A ``DeviceTensor`` object is created for every ``TensorDescription`` in the graph during ``Transformer.initialize_tensor_descriptions``. When a ``DeviceTensor`` object is created, the individual transformer can handle it in multiple ways. The CPU and GPU transformers both tag ``DeviceTensor`` objects to their underlying ``DeviceBufferStorage`` objects so that they can be allocated at the same time as the device allocation. Each transformer's ``DeviceTensor`` implementation must support some simple operations including copying to and from NumPy arrays. This is used to set argument values in the graph and get result values from the graph.

After all tensor descriptions are initialized and have created their device buffers and tensors, their allocation is transformed as shown:

.. code-block:: python

        self.start_transform_allocate()
        for device_buffer in self.device_buffers:
            device_buffer.transform_allocate()
        self.finish_transform_allocate()

What this means is that the actual allocation of buffers and tensors is transformed into an executable format similar to computations so that it can be run later. This transformed allocation operation is eventually executed by the ``Transformer.allocate_storage`` method.

Computation transformation
--------------------------

Finally, computation objects are transformed into an executable format after allocations are transformed in ``Transformer._transform_computations``:

.. code-block:: python

        for computation in self.computations:
            computation.transform()

The ``Computation.transform`` method first gets the set of all ops needed to evaluation the computation. Since graph passes might have replaced ops by updating their forward pointers, this method gets the fully forwarded set of ops. Then the ops are ordered in such a way that all execution dependencies are met using ``Digraph.can_reach``.

Each transformer implements a ``Transformer.transform_ordered_ops``, which accepts a list of ordered ops and transforms them into an executable format. The CPU transformer implements this by generating a Python function containing one or more NumPy calls for each op. Individual ops are handled in the CPU transformer with the corresponding ``CPUCodeGenerator.generate_op`` implementation. The GPU transformer implements this by generating a ``GPUKernelGroup`` containing a set of ``GPUKernel`` objects that can be executed to evaluate each op. Individual ops are handled in the GPU transformer with the corresponding ``GPUKernelGroup.add_kernel`` implementation or ``ElementWiseKernel.add_op`` implementation. The ElementWiseKernel generates CUDA C code to evaluate most op types. Other more complex ops have hand-written GPU kernels, such as convolution and GEMM. These are handled in different ``GPUKernel`` implementations.

When transformation of computations has finished, the transformer implementation must set the ``Computation.executor`` member to either a function or callable object which will serve as the entry point for computation evaluation.

Computation execution
=====================

Computations are executed by calling the ``Computation.executor`` member. For the CPU transformer this is a function pointer to the corresponding function in the generated Python NumPy code. For the GPU transformer this is the corresponding ``GPUKernelGroup`` object which implements the ``__call__`` method.


