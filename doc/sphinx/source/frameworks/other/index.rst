.. frameworks/other/index.rst:

.. _fw_other: 

.. contents::

Integrating other frameworks
============================

This section details some of the *configuration options* and some of the 
*environment variables* that can be used to tune for optimal performance when 
your system already has a version of nGraph installed with one or more of our 
supported :doc:`../../backends/index`.

Regardless of the framework, after the :doc:`../../buildlb` step, a good place 
to start usually involves making the libraries available to the framework. On 
Linux\* systems built on Intel® Architecture, that command tends to looks 
something like: 

.. code-block:: console

   export NGRAPH_CPP_BUILD_PATH=path/to/ngraph_dist/
   export LD_LIBRARY_PATH=path/to/ngraph_dist/lib/


Find or display version
-----------------------

If you're working with the :doc:`../../python_api/index`, the following command 
may be useful:

.. code-block:: console

   python3 -c "import ngraph as ng; print('nGraph version: ',ng.__version__)";

To manually build a newer version than is available from the latest `PyPI`_
(:abbr:`Python Package Index (PyPI)`), see our nGraph Python API `BUILDING.md`_ 
documentation.


Activate logtrace-related environment variables
-----------------------------------------------

Another configuration option is to activate ``NGRAPH_CPU_DEBUG_TRACER``,
a runtime environment variable that supports extra logging and debug detail. 

This is a useful tool for data scientists interested in outputs from logtrace 
files that can, for example, help in tracking down model convergences. It can 
also help engineers who might want to add their new ``Backend`` to an existing 
framework to compare intermediate tensors/values to references from a CPU 
backend.

To activate this tool, set the ``env`` var ``NGRAPH_CPU_DEBUG_TRACER=1``.
It will dump ``trace_meta.log`` and ``trace_bin_data.log``. The names of the 
logfiles can be customized.

To specify the names of logs with those flags:

:: 

  NGRAPH_TRACER_LOG = "meta.log"
  NGRAPH_BIN_TRACER_LOG = "bin.log"

The meta_log contains::
 
  kernel_name, serial_number_of_op, tensor_id, symbol_of_in_out, num_elements, shape, binary_data_offset, mean_of_tensor, variance_of_tensor

A line example from a unit-test might look like::

  K=Add S=0 TID=0_0 >> size=4 Shape{2, 2} bin_data_offset=8 mean=1.5 var=1.25

The binary_log line contains::

  tensor_id, binary data (tensor data)

A reference for the implementation of parsing these logfiles can also be found 
in the unit test for this feature.


FMV
---

FMV stands for :abbr:`Function Multi-Versioning`, and it can also provide a 
number of generic ways to patch or bring architecture-based optimizations to 
the :abbr:`Operating System (OS)` that is handling your ML environment. See 
the `GCC wiki for details`_.

If your nGraph build is a Neural Network configured on Clear Linux\* OS 
for Intel® Architecture, and it includes at least one older CPU, the 
`following article may be helpful`_.


Training Deep Neural Networks
-----------------------------

Before tweaking various environment variables, be aware that how the computation 
gets executed depends  on the data layout that the model is using. ``NHWC`` and 
``NCHW`` are common layouts in Deep Learning models. Your ultimate 
runtime can vary greatly -- even when all other factors are exactly the same -- 
when this detail is overlooked.

For CPU (and most cuDNN) backends, the preferred layout is currently ``NCHW``.

* **N** -- Number of images per batch
* **C** -- Channel of the image (expressed as a number like 3 for RGB and 1 
  for grayscale)
* **H** -- Height of the image
* **W** -- Width of the image


Intel® Math Kernel Library for Deep Neural Networks 
---------------------------------------------------

.. important:: Intel® MKL-DNN is automatically enabled as part of an
   nGraph default :doc:`build <../../buildlb>`; you do *not* need to add it 
   separately or as an additional component to be able to use these 
   configuration settings.

The following `KMP`_ options were originally optimized for models using the 
Intel® `MKL-DNN`_ to train models with the ``NCHW`` data layout; however, other 
configurations can be explored.    

* ``KMP_BLOCKTIME`` Sets the time, in milliseconds, that a thread should wait 
  after completing the execution of a parallel region, before sleeping.
* ``KMP_AFFINITY`` Enables the runtime library to bind threads to physical 
  processing units. A useful article that explains more about how to use this 
  option for various CPU backends is here: https://web.archive.org/web/20190401182248/https://www.nas.nasa.gov/hecc/support/kb/Using-Intel-OpenMP-Thread-Affinity-for-Pinning_285.html
* ``KMP_SETTINGS`` Enables (``true``) or disables (``false``) the printing of 
  OpenMP\* runtime library environment variables during program execution.
* ``OMP_NUM_THREADS`` Specifies the number of threads to use.


nGraph-enabled Intel® Xeon® 
---------------------------

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

The best resource for this configuration option is the Intel® OpenMP\* docs 
at the following link: `Intel OpenMP documentation`_. ``OMP_NUM_THREADS`` 
defaults to the number of logical cores. To check the number of cores on your 
system, you can run the following on the command-line to see the details 
of your CPU:

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

.. _PyPI: https://pypi.org/project/ngraph-core
.. _KMP: https://software.intel.com/en-us/node/522691
.. _MKL-DNN: https://github.com/intel/mkl-dnn
.. _Intel OpenMP documentation: https://www.openmprtl.org/documentation
.. _Movidius: https://www.movidius.com/
.. _BUILDING.md: https://github.com/NervanaSystems/ngraph/blob/master/python/BUILDING.md
.. _GCC wiki for details: https://gcc.gnu.org/wiki/FunctionMultiVersioning
.. _following article may be helpful: https://clearlinux.org/documentation/clear-linux/tutorials/fmv
