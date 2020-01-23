.. inspection/profiling.rst:

.. _profiling: 

Performance testing with ``nbench``
###################################

The nGraph Compiler stack includes the ``nbench`` tool which 
provides additional methods of assessing or debugging performance 
issues.

If you follow the build process under :doc:`../buildlb`, the 
``NGRAPH_TOOLS_ENABLE`` flag defaults to ``ON`` and automatically 
builds ``nbench``. As its name suggests, ``nbench`` can be used 
to benchmark any nGraph-serialized model with a given backend.

To benchmark an already-serialized nGraph ``.json`` model with, for 
example, a ``CPU`` backend, run ``nbench`` as follows.

.. code-block:: console

   $ cd ngraph/build/src/tools
   $ nbench/nbench -b CPU - i 1 -f <serialized_json file>

Samples for testing can be found under  ``ngraph/test/models``.

.. _nbench:

``nbench``
==========

.. code-block:: none

    Benchmark and nGraph JSON model with a given backend.
    
    SYNOPSIS
        nbench [-f <filename>] [-b <backend>] [-i <iterations>]
    OPTIONS
        -f|--file                 Serialized model file
        -b|--backend              Backend to use (default: CPU)
        -d|--directory            Directory to scan for models. All models are benchmarked.
        -i|--iterations           Iterations (default: 10)
        -s|--statistics           Display op statistics
        -v|--visualize            Visualize a model (WARNING: requires Graphviz installed)
        --timing_detail           Gather detailed timing
        -w|--warmup_iterations    Number of warm-up iterations
        --no_copy_data            Disable copy of input/result data every iteration
        --dot                     Generate Graphviz dot file
        --double_buffer           Double buffer inputs and outputs

.. _nbench_tf:

Use ``nbench`` to ease end-to-end debugging for TensorFlow\*
------------------------------------------------------------

Rather than run a TensorFlow\* model "end-to-end" all the time, 
developers who notice a problem with performance or memory usage 
can generate a unique serialized model for debugging by using 
``NGRAPH_ENABLE_SERIALIZE=1``. This serialized model can then be 
run and re-run with ``nbench`` to efficiently experiment with any 
changes in ``ngraph`` space; developers can make changes and test 
changes without the overhead of a complete end-to-end compilation 
for each change.

Find or display version
-----------------------

If you're working with the :doc:`../../python_api/index`, the following command 
may be useful:

.. code-block:: console

   python3 -c "import ngraph as ng; print('nGraph version: ',ng.__version__)";

To manually build a newer version than is available from the latest `PyPI`_
(:abbr:`Python Package Index (PyPI)`), see our nGraph Python API `BUILDING.md`_ 
documentation.


.. _PyPI: https://pypi.org/project/ngraph-core/
.. _BUILDING.md: https://github.com/NervanaSystems/ngraph/blob/master/python/BUILDING.md
