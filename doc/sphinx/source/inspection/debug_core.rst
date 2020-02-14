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

   ``NGRAPH_CODE_COVERAGE_ENABLE``, Enable code coverage data collection, ``FALSE``
   ``NGRAPH_DEBUG_ENABLE``, Enable output for ``NGRAPH_DEBUG`` statements, ``FALSE``
   ``NGRAPH_DEPRECATED_ENABLE``, Enable compiler deprecation pragmas for deprecated APIs (recommended only for development use), ``FALSE``
   ``NGRAPH_DEX_ONLY``, Build CPU DEX without codegen, ``FALSE``
   ``NGRAPH_DISTRIBUTED_ENABLE``, Enable distributed training using MLSL/OpenMPI, ``OFF``
   ``NGRAPH_DISTRIBUTED_MLSL_ENABLE``, Use MLSL, ``OFF``
   ``NGRAPH_DOC_BUILD_ENABLE``,  Automatically build documentation, ``OFF``
   ``NGRAPH_FAST_MATH_ENABLE``,  Enable fast math, ``ON``
   ``NGRAPH_HALIDE``,  ,``OFF``
   ``NGRAPH_INTERPRETER_ENABLE``, Control the building of the ``INTERPRETER`` backend,  ``TRUE``
   ``NGRAPH_INTERPRETER_STATIC_LIB_ENABLE``, Enable build INTERPRETER backend static library, ``FALSE``
   ``NGRAPH_JSON_ENABLE``, Enable JSON based serialization and tracing features, ``TRUE``
   ``NGRAPH_LIB_VERSIONING_ENABLE``, Enable shared library versioning, ``FALSE``
   ``NGRAPH_MLIR_ENABLE``, Control the building of MLIR backend, ``FALSE``
   ``NGRAPH_NOP_ENABLE``,  Control the building of the NOP backend,  ``TRUE``
   ``NGRAPH_ONNX_IMPORT_ENABLE``, Enable ONNX importer, ``FALSE``
   ``NGRAPH_PLAIDML_ENABLE``, Enable the PlaidML backend,  ``${PLAIDML_FOUND}``
   ``NGRAPH_PYTHON_BUILD_ENABLE``, Enable build of ``NGRAPH`` python package wheel, ``FALSE``
   ``NGRAPH_STATIC_LIB_ENABLE``, Enable build ``NGRAPH`` static library, ``FALSE``
   ``NGRAPH_TBB_ENABLE``, Only if (``NGRAPH_CPU_ENABLE``) Control usage of TBB for CPU backend, ``TRUE``
   ``NGRAPH_TOOLS_ENABLE``, Control the building of tools, ``TRUE``
   ``NGRAPH_UNIT_TEST_ENABLE``,  Control the building of unit tests, ``TRUE``
   ``NGRAPH_USE_PREBUILT_LLVM``, Use a precompiled LLVM, ``FALSE``
   ``NGRAPH_USE_PREBUILT_MLIR``, Use the `precompiled MLIR`_,``FALSE``


Environment Variables
=====================

.. important:: Many of the following flags may be experimental only and subject to change.


.. csv-table::
   :header: "Environment Variable", "Description"
   :widths: 20, 35
   :escape: ~

   ``NGRAPH_DISABLE_LOGGING``,	Disable printing all logs irrespective of build type
   ``NGRAPH_DISABLED_FUSIONS``,	Disable specified fusions. Specified as `;` separated list and supports regex
   ``NGRAPH_ENABLE_REPLACE_CHECK``,	Enables strict type checking in copy constructor copy_with_new_args
   ``NGRAPH_ENABLE_SERIALIZE_TRACING``, generates 1 ``json`` file per pass to run with ``nbench`` for localized execution rather than whole stack execution
   ``NGRAPH_ENABLE_TRACING``, Enables creating graph execution timelines to be viewed in ``chrome://tracing`` see also :doc:`viz_tools`.
   ``NGRAPH_ENABLE_VISUALIZE_TRACING``,	Enables creating visual graph for each pass ``.svg`` files by default; see also :doc:`viz_tools`
   ``NGRAPH_FAIL_MATCH_AT``, Allows one to specify node name patterns to abort pattern matching at particular nodes. Helps debug an offending fusion
   ``NGRAPH_GTEST_INFO``, Enables printing info about a specific test
   ``NGRAPH_INTER_OP_PARALLELISM``, See :ref:`interop_intraop`
   ``NGRAPH_INTRA_OP_PARALLELISM``, See :ref:`interop_intraop`
   ``NGRAPH_PASS_ATTRIBUTES``, Specify pass-specific attributes as a semi-colon separated list to be enabled or disabled. Naming of pass attributes is up to the backends and see also `pass config`_
   ``NGRAPH_PASS_ENABLES``,	Specify a semi-colon separated list to enable or disable a pass on core or backend. This will override the default enable/disable values
   ``NGRAPH_PROFILE_PASS_ENABLE``, Dump the name and execution time of each pass; shows per-pass time taken to compile
   ``NGRAPH_PROVENANCE_ENABLE``, Enable adding provenance info to nodes. This will also be added to serialized files.
   ``NGRAPH_SERIALIZER_OUTPUT_SHAPES``,	Enable adding output shapes in the serialized graph
   ``NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE``,	Calculated in code; helps prevent *long* edges between two nodes very far apart
   ``NGRAPH_VISUALIZE_EDGE_LABELS``, Set it to 1 in ``~/.bashrc``; adds label to a graph edge when NGRAPH_ENABLE_VISUALIZE_TRACING=1
   ``NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES``, Set it to 1 in ``~/.bashrc``; adds output shape of a node when NGRAPH_ENABLE_VISUALIZE_TRACING=1
   ``NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES``, Set it to 1 in ``~/.bashrc``; adds output type of a node when NGRAPH_ENABLE_VISUALIZE_TRACING=1
   ``NGRAPH_VISUALIZE_TRACING_FORMAT``, Default format is ``.svg``. See also :doc:`viz_tools` 
   ``OMP_NUM_THREADS``, See: `OpenMPI Runtime Library Documentation`_



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
.. _precompiled MLIR: https://github.com/IntelAI/mlir

Looking at graph objects
------------------------

A number of nGraph objects can print themselves on streams. For example,``cerr << a + b`` produces
``v0::Add Add_2(Parameter_0[0]:f32{2,3}, Parameter_1[0]:f32{2,3}):(f32{2,3})`` indicating the
specific version of the op, its name, arguments, and outputs.
