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
   ``NGRAPH_DISTRIBUTED_ENABLE``, Enable distributed training using MLSL/OpenMPI, ``OFF``
   ``NGRAPH_DISTRIBUTED_MLSL_ENABLE``, Use MLSL, ``OFF``
   ``NGRAPH_DOC_BUILD_ENABLE``,  Automatically build documentation, ``OFF``
   ``NGRAPH_FAST_MATH_ENABLE``,  Enable fast math, ``ON``
   ``NGRAPH_INTERPRETER_ENABLE``, Control the building of the ``INTERPRETER`` backend,  ``TRUE``
   ``NGRAPH_INTERPRETER_STATIC_LIB_ENABLE``, Enable build INTERPRETER backend static library, ``FALSE``
   ``NGRAPH_JSON_ENABLE``, Enable JSON based serialization and tracing features, ``TRUE``
   ``NGRAPH_LIB_VERSIONING_ENABLE``, Enable shared library versioning, ``FALSE``
   ``NGRAPH_NOP_ENABLE``,  Control the building of the NOP backend,  ``TRUE``
   ``NGRAPH_ONNX_IMPORT_ENABLE``, Enable ONNX importer, ``FALSE``
   ``NGRAPH_PLAIDML_ENABLE``, Enable the PlaidML backend,  ``${PLAIDML_FOUND}``
   ``NGRAPH_PYTHON_BUILD_ENABLE``, Enable build of ``NGRAPH`` python package wheel, ``FALSE``
   ``NGRAPH_STATIC_LIB_ENABLE``, Enable build ``NGRAPH`` static library, ``FALSE``
   ``NGRAPH_TOOLS_ENABLE``, Control the building of tools, ``TRUE``
   ``NGRAPH_UNIT_TEST_ENABLE``,  Control the building of unit tests, ``TRUE``


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



.. _pass config: https://github.com/NervanaSystems/ngraph/blob/a4a3031bb40f19ec28704f76de39762e1f27e031/src/ngraph/pass/pass_config.cpp#L54
.. _OpenMPI Runtime Library Documentation: https://www.openmprtl.org/documentation

Looking at graph objects
------------------------

A number of nGraph objects can print themselves on streams. For example,``cerr << a + b`` produces
``v0::Add Add_2(Parameter_0[0]:f32{2,3}, Parameter_1[0]:f32{2,3}):(f32{2,3})`` indicating the
specific version of the op, its name, arguments, and outputs.
