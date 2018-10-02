.. frameworks/generic.rst


Working with generic frameworks
###############################

An engineer may want to work with a deep learning framework that does not yet 
have bridge code written. For non-supported or “generic” frameworks, it is 
expected that engineers will use the nGraph library to create custom bridge code, 
and/or to design and document a user interface (UI) with specific runtime 
options for whatever custom use case they need. 

The two primary tasks that can be accomplished in the “bridge code” space of the 
nGraph Abstraction layer are: (1) compiling a dataflow graph and (2) executing 
a pre-compiled graph. See the :doc:`../framework-integration-guides` for how we 
have built bridges with other frameworks. For more in-depth help in writing 
graph optimizations and bridge code, we provide articles on how to 
:doc:`../fusion/index`, and programmatically :doc:`../howto/execute` that can 
target various compute resources using nGraph when a framework provides some 
inputs to be computed.


Activate nGraph |trade| on generic frameworks
=============================================

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


Regardless of the framework, after the :doc:`../buildlb`, a good place to start 
usually involves making the libraries available to the framework. On Linux\* 
systems, that command tends to looks something like: 

.. code-block:: console

   export NGRAPH_CPP_BUILD_PATH=path/to/ngraph_dist/
   export LD_LIBRARY_PATH=path/to/ngraph_dist/lib/


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

MKL-DNN
-------

The following `KMP options`_ were originally optimized for `MKLDNN`_ projects 
running models with the ``NCHW`` data layout; however, other configurations can 
be explored. MKL-DNN is automatically enabled as part of an nGraph build; you do 
*not* need to add MKL-DNN separately or as an additional component to be able to 
use these configuration settings.   

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
achieve the best performance for NN workloads on CPU platforms. 


.. _KMP options: https://software.intel.com/en-us/node/522691
.. _MKLDNN: https://github.com/intel/mkl-dnn
.. _gnu.org site: https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html
.. _Movidius: https://www.movidius.com/
