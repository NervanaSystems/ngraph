.. inspection/debug_core.rst:

.. contents::

.. _debug_core:

Diagnostics
###########

.. important:: Many of the following flags may be experimental only and subject to change.

Build nGraph with various compile flags and environment variables to diagnose performance
and memory issues.  See also :doc:`profiling`.


Compile Flags
=============

.. csv-table::
   :header: "Compile Flag", "Description", "Default Value"
   :widths: 20, 35, 5
   :escape: ~

   ``NGRAPH_UNIT_TEST_ENABLE``, Control the building of unit tests,``TRUE``
   ``NGRAPH_TOOLS_ENABLE``, Control the building of tools,``TRUE``
   ``NGRAPH_CPU_ENABLE``, Control the building of the CPU backend,``TRUE``
   ``NGRAPH_INTERPRETER_ENABLE``, Control the building of the ``INTERPRETER`` backend,``TRUE``
   ``NGRAPH_NOP_ENABLE``, Control the building of the NOP backend,``TRUE``
   ``NGRAPH_ENABLE_CPU_CONV_AUTO``, Enable mkldnn convolution_auto for CPU,``TRUE``
   ``NGRAPH_JSON_ENABLE``, Enable JSON based serialization and tracing features,``TRUE``
   ``NGRAPH_DYNAMIC_COMPONENTS_ENABLE``, Enable dynamic loading of components,``TRUE``
   ``NGRAPH_FAST_MATH_ENABLE``, Enable fast math, ``ON``
   ``NGRAPH_TBB_ENABLE``, "Only if (``NGRAPH_CPU_ENABLE``) Control usage of TBB for CPU backend",``TRUE``
   ``NGRAPH_USE_PREBUILT_LLVM``, ,``FALSE``
   ``NGRAPH_ONNX_IMPORT_ENABLE``, Enable ONNX importer,``FALSE``
   ``NGRAPH_DEBUG_ENABLE``, Enable output for ``NGRAPH_DEBUG`` statements,``FALSE``
   ``NGRAPH_DEX_ONLY``, Build CPU DEX without codegen,``FALSE``
   ``NGRAPH_MLIR_ENABLE``, Control the building of MLIR backend,``FALSE``
   ``NGRAPH_INTELGPU_ENABLE``, Control the building of the Intel GPU backend with clDNN,``FALSE``
   ``NGRAPH_GPU_ENABLE``, Control the building of the GPU backend,``FALSE``
   ``NGRAPH_GPUH_ENABLE``, Control the building of the Hybrid GPU backend,``FALSE``
   ``NGRAPH_GENERIC_CPU_ENABLE``, Enable build ``NGRAPH`` for generic CPU backend,``FALSE``
   ``NGRAPH_DEPRECATED_ENABLE``, Enable compiler deprecation pragmas for deprecated APIs (recommended only for development use),``FALSE``
   ``NGRAPH_CODE_COVERAGE_ENABLE``, Enable code coverage data collection,``FALSE``
   ``NGRAPH_LIB_VERSIONING_ENABLE``, Enable shared library versioning,``FALSE``
   ``NGRAPH_PYTHON_BUILD_ENABLE``, Enable build of ``NGRAPH`` python package wheel,``FALSE``
   ``NGRAPH_PLAIDML_ENABLE``, Enable the PlaidML backend, ``${PLAIDML_FOUND}``
   ``NGRAPH_STATIC_LIB_ENABLE``, Enable build ``NGRAPH`` static library,``FALSE``
   ``NGRAPH_INTERPRETER_STATIC_LIB_ENABLE``, Enable build INTERPRETER backend static library,``FALSE``
   ``NGRAPH_CPU_STATIC_LIB_ENABLE``, Enable build CPU backend static library,``FALSE``
   ``NGRAPH_DISTRIBUTED_ENABLE``, Enable distributed training using MLSL/OpenMPI,``OFF``
   ``NGRAPH_DISTRIBUTED_MLSL_ENABLE``, Use MLSL ,``OFF``
   ``NGRAPH_HALIDE``, , ``OFF``
   ``NGRAPH_DOC_BUILD_ENABLE``, Automatically build documentation ,``OFF``


Environment Variables
=====================

.. important:: Many of the following flags may be experimental only and subject to change.


.. csv-table::
   :header: "Environmental Variable", "Description"
   :widths: 20, 35
   :escape: ~

   ``NGRAPH_MLIR_OPT_LEVEL``, 
   ``NGRAPH_MLIR_MAX_CYCLE_DEPTH``, 
   ``NGRAPH_MLIR``, 
   ``NGRAPH_ENABLE_VISUALIZE_TRACING``,
   ``NGRAPH_VISUALIZE_TRACING_FORMAT``, Default format is ``.svg``
   ``NGRAPH_VISUALIZE_EDGE_LABELS``, Set it to 1 in ``~/.bashrc``
   ``NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE``, Calculated in code; helps prevent *long* edges between two nodes very far apart
   ``NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES``, Set it to 1 in ``~/.bashrc``
   ``NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES``, Set it to 1 in ``~/.bashrc``
   ``NGRAPH_ENABLE_SERIALIZE_TRACING``, Creates serialized files to be run with ``nbench`` for localized execution rather than whole stack execution
   ``NGRAPH_PROFILE_PASS_ENABLE``, Per-pass time taken to compile
   ``NGRAPH_PASS_ENABLES``, Enable or disable a pass: either core or backend
   ``NGRAPH_PASS_ATTRIBUTES``, Enable or disable attributes related to a pass; see also `pass config`_
   ``NGRAPH_CPU_TRACING``, Generate Timelines for CPU to check in ``chrome://tracing``
   ``NGRAPH_ENABLE_TRACING``,
   ``NGRAPH_SERIALIZER_OUTPUT_SHAPES``,
   ``NGRAPH_COMPILER_DEBUGINFO_ENABLE``,
   ``NGRAPH_COMPILER_DIAG_ENABLE``,
   ``NGRAPH_COMPILER_REPORT_ENABLE``,
   ``NGRAPH_DISABLE_LOGGING``,
   ``NGRAPH_FAIL_MATCH_AT``,
   ``NGRAPH_PROVENANCE_ENABLE``,
   ``NGRAPH_CODEGEN``,
   ``NGRAPH_DEX_DEBUG``,
   ``NGRAPH_CPU_CONCURRENCY``,
   ``NGRAPH_CPU_DEBUG_TRACER``, See also :ref:`debug_tracer`
   ``NGRAPH_CPU_TRACER_LOG``, See also :ref:`debug_tracer`
   ``NGRAPH_CPU_BIN_TRACER_LOG``, See also :ref:`debug_tracer`
   ``NGRAPH_CPU_USE_REF_KERNELS``, 
   ``OMP_NUM_THREADS``, See `OpenMPI Runtime Library Documentation`_
   ``NGRAPH_INTRA_OP_PARALLELISM``, See also :ref:`interop_intraop`
   ``NGRAPH_INTER_OP_PARALLELISM``, See also :ref:`interop_intraop`
   ``NGRAPH_CPU_EIGEN_THREAD_COUNT``,
   ``NGRAPH_CPU_CHECK_PARMS_AND_CONSTS``,
   ``NGRAPH_CPU_NAN_CHECK``,
   ``NGRAPH_CPU_INF_CHECK``,
   ``NGRAPH_DECONV_FUSE``, "Default ``FALSE``; when ``TRUE`` it enables fusion for deconvolution.  Only available with CPU."
   ``NGRAPH_PASS_CPU_LAYOUT_ELTWISE``,



.. _debug_tracer:

Debug Tracer
------------

Another diagnostic configuration option is to activate ``NGRAPH_CPU_DEBUG_TRACER``,
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


.. _interop_intraop:

Intra-op and inter-op parallelism
---------------------------------

* ``intra_op_parallelism_threads``
* ``inter_op_parallelism_threads``

Some frameworks, like TensorFlow\*, use these settings to improve performance; 
however, they are often not sufficient for optimal performance. Framework-based 
adjustments cannot access the underlying NUMA configuration in multi-socket 
Intel® Xeon® processor-based platforms, which is a key requirement for 
many kinds of inference-engine computations.

The meta_log contains::
 
  kernel_name, serial_number_of_op, tensor_id, symbol_of_in_out, num_elements, shape, binary_data_offset, mean_of_tensor, variance_of_tensor

A line example from a unit-test might look like::

  K=Add S=0 TID=0_0 >> size=4 Shape{2, 2} bin_data_offset=8 mean=1.5 var=1.25

The binary_log line contains::

  tensor_id, binary data (tensor data)

A reference for the implementation of parsing these logfiles can also be found 
in the unit test for this feature.


.. _pass config: https://github.com/NervanaSystems/ngraph/blob/a4a3031bb40f19ec28704f76de39762e1f27e031/src/ngraph/pass/pass_config.cpp#L54
.. _OpenMPI Runtime Library Documentation: https://www.openmprtl.org/documentation