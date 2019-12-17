.. inspection/profiling.rst:

.. _profiling: 

Profiling Performance
#####################

The nGraph Compiler stack provides the ``nbench`` tool to assess 
and debug.

If you follow the build process under :doc:`../buildlb`, the 
``NGRAPH_TOOLS_ENABLE`` flag defaults to ``ON``, and automatically 
builds ``nbench``.  ``nbench`` can be used to benchmark any nGraph 
``.json`` model with a given backend.

Some samples can be found under  ``ngraph/test/models``.

.. note:: To get your own serialized files from a framework, 
   like TensorFlow\*, use ``NGRAPH_ENABLE_SERIALIZE=1``.

Example
-------

To benchmark an already-serialized nGraph ``.json`` model with, for 
example, a ``CPU`` backend, run ``nbench`` as follows:

.. code-block:: console

   $ cd ngraph/build/src/tools
   $ nbench/nbench -b CPU - i 1 -f <serialized_json file>


.. _nbench:

``nbench``
==========

Options 
-------

::
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

