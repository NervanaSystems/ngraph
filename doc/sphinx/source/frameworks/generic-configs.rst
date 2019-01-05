.. frameworks/generic-configs.rst:


Configurations available to any framework
#########################################


Enabling Deep Learning paradigms  
================================

Framework architects or engineers who can't quite find what they need among 
the existing DL tools may need to build something new off a "stock" framework, 
or someting entirely from scratch. For this category of developer, we have 
:doc:`documented several ways <../howto/index>` you can incorporate built-in 
compiler support for users of your framework; this includes out-of-box support 
for things like Intel® MKL-DNN and PlaidML when your framework supports nGraph 
as a "backend" or engine. 

   .. important:: nGraph does not provide an interface for "users" of frameworks 
      (for example, we cannot dictate or control how Tensorflow* or MXNet* presents 
      interfaces to users). Please keep in mind that designing and documenting 
      the :abbr:`User Interface (UI)` of step 3 above is entirely in the realm 
      of the framework owner or developer and beyond the scope of the nGraph 
      Compiler stack. However, any framework can be designed to make direct use 
      of nGraph Compiler stack-based features and then expose an accompanying UI, 
      output message, or other detail to a user.
 
The nGraph :abbr:`IR Intermediate Representation` is format that can understand 
inputs from a framework. Today, there are two primary tasks that can be accomplished 
in the “bridge code” space of the nGraph IR: 

#. Compiling a dataflow graph 
#. Executing a pre-compiled graph. 

See the :doc:`../framework-integration-guides` for how we built bridges with our 
initially-supported frameworks. For more in-depth help in writing things like 
graph optimizations and bridge code, we provide articles on how to 
:doc:`../fusion/index`, and programmatically :doc:`../howto/execute` that can 
target various compute resources using nGraph when a framework provides some 
inputs to be computed.

.. note:: Configuration options can be added manually on the command line or via 
   scripting. Please keep in mind that fine-tuning of parameters is as much of 
   an art as it is a science; there are virtually limitless ways to do so and 
   our documentation provides only a sampling.  


Integrating nGraph with new frameworks
======================================

This section details some of the *configuration options* and some of the 
*environment variables* that can be used to tune for optimal performance when 
your system already has a version of nGraph installed with one of our supported
backends. 

.. csv-table::
   :header: "Backend", "Current nGraph support", "Future nGraph support"
   :widths: 35, 10, 10

   Intel® Architecture Processors (CPUs), Yes, Yes
   Intel® Nervana™ Neural Network Processor™ (NNPs), Yes, Yes
   NVIDIA\* CUDA (GPUs), Yes, Some 
   :abbr:`Field Programmable Gate Arrays (FPGA)` (FPGAs), Coming soon, Yes
   `Movidius`_, Not yet, Yes
   Other, Not yet, Ask


Regardless of the framework, after the :doc:`../buildlb` step, a good place 
to start usually involves making the libraries available to the framework. On 
Linux\* systems built on Intel® Architecture, that command tends to looks 
something like: 

.. code-block:: console

   export NGRAPH_CPP_BUILD_PATH=path/to/ngraph_dist/
   export LD_LIBRARY_PATH=path/to/ngraph_dist/lib/



FMV
===

FMV stands for :abbr:`Function Multi-Versioning`, and it can also provide a 
number of generic ways to patch or bring architecture-based optimizations to 
the :abbr:`Operating System (OS)` that is handling your ML environment. See 
the `GCC wiki for details`_.

If your nGraph build is a Neural Network configured on Clear Linux* OS 
for Intel® Architecture, and it includes at least one older CPU, the 
`following article may be helpful`_.



Training Deep Neural Networks
==============================

Before tweaking various environment variables, be aware that how the computation 
gets executed depends upon the ordering of the data format that the model is 
using. ``NHWC`` and ``NCHW`` are the two more common layouts in Deep Learning 
models. Your ultimate runtime can vary greatly -- even when all other factors 
are exactly the same -- when this detail is overlooked.

For CPU (and most cuDNN) backends, the preferred layout is currently ``NCHW``.

* **N** -- Number of images per batch
* **C** -- Channel of the image (expressed as a number like 3 for RGB and 1 
  for grayscale)
* **H** -- Height of the image
* **W** -- Width of the image

Intel® Math Kernel Library for Deep Neural Networks 
---------------------------------------------------

-The following `KMP options`_ were originally optimized for models using the 
Intel® `MKL-DNN`_ to train models with the ``NCHW`` data layout; however, other 
configurations can be explored. MKL-DNN is automatically enabled as part of an 
nGraph compilation; you do *not* need to add MKL-DNN separately or as an 
additional component to be able to use these configuration settings.   

* ``KMP_BLOCKTIME`` Sets the time, in milliseconds, that a thread should wait 
  after completing the execution of a parallel region, before sleeping.
* ``KMP_AFFINITY`` Enables the runtime library to bind threads to physical 
  processing units. 
* ``KMP_SETTINGS`` Enables (``true``) or disables (``false``) the printing of 
  OpenMP\* runtime library environment variables during program execution.
* ``OMP_NUM_THREADS`` Specifies the number of threads to use.


nGraph-enabled Intel® Xeon® 
============================

The list below includes recommendations on data layout, parameters, and 
application configuration to achieve best performance running DNN workloads on 
Intel® Xeon® (CPU processor) systems.

Threading 
---------

The number of threads set by ``OMP_NUM_THREADS`` ought not exceed the number of 
physical cores. The threads should be pinned to their respective physical cores 
and activated as follows:

* When ``HT=off``, ``KMP_AFFINITY=compact,granularity=fine``

* When ``HT=on``, ``KMP_AFFINITY=compact,1,0,granularity=fine``


Memory allocation 
-----------------

Buffer pointers should be aligned on 64-byte boundaries. NUMA policy should be 
configured for local memory allocation (``numactl --localloc``). 



Convolution shapes
^^^^^^^^^^^^^^^^^^

* When **running inference, or training for forward-propagation and weight 
  updates**, for best performance:
  
  - the number of input channels should be 1, 3, or a multiple of SIMD-width (8 
    for AVX2 systems, 16 for AVX512 systems). 
  - the number of output channels should be a multiple of SIMD-width (8 for AVX2 
    systems, 16 for AVX512 systems).

* When **training backward propagation**, the number of input and output 
  channels should be a multiple of SIMD-width (8 for AVX2 systems, 16 for AVX512 
  systems),
  
  - padding should not exceed :math:`0.5x` where :math:`x` is the kernel size.
  - kernel width should be less than 14.


``OMP_NUM_THREADS``
^^^^^^^^^^^^^^^^^^^

The best resource for this configuration option is the `gnu.org site`_ 
``OMP_NUM_THREADS`` defaults to the number of logical cores. To check the 
number of cores on your system, you can run the following on the command-line to 
see the details of your CPU: 

.. code-block:: console

   $ lscpu


Intra-op and inter-op parallelism 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``intra_op_parallelism_threads``
* ``inter_op_parallelism_threads``

Some frameworks, like TensorFlow\*, use these settings to improve performance; 
however, they are often not sufficient for optimal performance. Framework-based 
adjustments cannot access the underlying NUMA configuration in multi-socket 
Intel® Xeon® processor-based platforms, which is a key requirement for 
many kinds of inference-engine computations. See the next section on NUMA 
performance to learn more about this performance feature available to systems 
utilizing nGraph. 
   

NUMA performance 
~~~~~~~~~~~~~~~~~

NUMA stands for :abbr:`Non-Uniform Memory Access (NUMA)`. It indicates how each 
CPU can access memory attached to each socket. 

Without the "knowledge" of CPU socket and NUMA configuration, a simple thread 
affinity (as in the case of thread pool) does not lead to optimal performance. 
In fact, it can sometimes prohibitively decrease throughput; a core from socket 
0 might have to continually access cache lines from the memory bank of socket 1, 
increasing bandwidth demands on the Intel® Ultra-Path Interconnect (Intel® UPI). 
This situation is exacerbated with larger number of sockets found in 4, 8, and 
16-socket systems. We believe that users need to be aware of system level 
optimizations in addition to framework specific configuration parameters to 
achieve the best performance for NN workloads on CPU platforms. The nGraph 
Compiler stack runs on transformers handled by Intel® Architecture (IA), and 
thus can make more efficient use of the underlying hardware.




.. _KMP options: https://software.intel.com/en-us/node/522691
.. _MKL-DNN: https://github.com/intel/mkl-dnn
.. _gnu.org site: https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html
.. _Movidius: https://www.movidius.com/
.. _GCC wiki for details: https://gcc.gnu.org/wiki/FunctionMultiVersioning
.. _following article may be helpful: https://clearlinux.org/documentation/clear-linux/tutorials/fmv
