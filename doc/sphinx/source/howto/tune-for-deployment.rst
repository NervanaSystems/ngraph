.. tune-for-deployment.rst


############################################
Fine-tune code deployment to nGraph devices  
############################################

When deploying code to nGraph-enabled devices, there are some additional 
configuration options that can be incorporated. Fine-tuning an nGraph-enabled
device is as much of an art as it is a science; there are virtually limitless
ways to command your training or inference computations to run the way you want.  

In this section, which was written for framework architects or engineers who are 
optimizing a new or less widely-supported framework, we provide some of our 
learnings from the work we've done in developing our custom bridge code, such as
that for our `ngraph tensorflow bridge`_ code. 

Indeed, how this has worked :doc:`for many <../framework-integration-guides>` 
of the "direct optimizations" we've shared with the developer community, 
`engineering teams carefully tune the workload to extract best performance`_ 
from a specific :abbr:`DL (Deep Learning)` model embedded in a specific framework 
that is training a specific dataset. Some of the ways we attempt to improve 
performance include: 

* Testing and recording the results of various system-level configuration options
  or enabled or disabled flags,
* Compiling with a mix of custom environment variables, 
* Finding semi-related comparisons for benchmarking [#1]_, and 
* Tuning lower levels of the system so that the machine-learning algorithm can 
  learn faster or more accurately that it did on previous runs, such as 
  with :doc:`../ops/index`. 

In nearly every case, the "best" mix of configuration options boils down to 
something unique to that framework-model topology. The larger and more complex a 
framework is, the harder it becomes to extract the best performance; 
configuration options that are enabled by "default" from the framework side can 
sometimes slow down compilation time without the developer being any the wiser. 
Sometimes only `a few small`_ adjustments can increase performance. Likewise, a 
minimally-designed framework could offer significant performance-improvement 
opportunities by lowering overhead.

For this reason, we're providing some of the more commonly-used options for 
fine-tuning various kinds code deployments to the nGraph-enabled devices we 
currently support. Watch this section for new updates. 


Training Deep Neural Networks
==============================

This section details some of the *configuration options* and *environment variables* 
that can be used to tune for optimal performance when your system already has a
version of nGraph installed.  

Before tweaking various environment variables, be aware that how the computation 
gets executed depends upon the data format that the model is using. NHWC or NCHW
are the two more common layouts, and the ultimate runtime can vary greatly -- 
even when all other factors are the same -- when this detail is overlooked.

For CPU (and most cuDNN) backends, the preferred layout is ``NCHW``.

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
*not* need to add MKL-DNN separately on the framework side to use these 
configuration settings.  

* ``KMP_BLOCKTIME`` Sets the time, in milliseconds, that a thread should wait 
  after completing the execution of a parallel region, before sleeping.
* ``KMP_AFFINITY`` Enables the runtime library to bind threads to physical 
  processing units. 
* ``KMP_SETTINGS`` Enables (``true``) or disables (``false``) the printing of 
  OpenMP* runtime library environment variables during program execution.
* ``OMP_NUM_THREADS`` Specifies the number of threads to use.


Code deployment to nGraph-enabled Intel® Xeon®
==============================================

The list below includes recommendations on :term:`data layouts`, parameters, and 
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

Buffer pointers should be aligned at the 64-byte boundary. NUMA policy should be 
configured for local memory allocation (``numactl --localloc``)

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

The best resource for this configuration option is the `gnu org site`_ 
``OMP_NUM_THREADS`` defaults to the number of logical cores. To chekc the 
number of cores on your system, you can run the following on the command-line to 
see the details of your CPU: 

.. code-block:: console

   $ lscpu  


Intra-op and inter-op parallelism 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``intra_op_parallelism_threads``
* ``inter_op_parallelism_threads``

Some frameworks, like Tensorflow, use these settings to improve performance; 
however, they are often not sufficient to achieve optimal performance. 
Framework-based adjustments cannot access the underlying  NUMA configuration in 
multi-socket Intel Xeon processor-based platforms, which is a key requirement for
many kinds of inference-engine computations.  See the next section on 
NUMA performance to learn more about this performance feature available to systems
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




.. rubric:: Footnotes

.. [#1] Benchmarking performance of DL systems is a young discipline; it is a
   good idea to be vigilant for results based on atypical distortions in the 
   configuration parameters. Every topology is different, and performance 
   increases or slowdowns can be attributed to multiple means.    

.. _ngraph tensorflow bridge: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html#tensorflow
.. _engineering teams carefully tune the workload to extract best performance: https://ai.intel.com/accelerating-deep-learning-training-inference-system-level-optimizations
.. _a few small: https://software.intel.com/en-us/articles/boosting-deep-learning-training-inference-performance-on-xeon-and-xeon-phi
.. _KMP options: https://software.intel.com/en-us/node/522691
.. _MKLDNN: https://github.com/intel/mkl-dnn
.. _gnu org site: https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html




