.. inspection/profiling.rst:

.. _profiling: 

Profiling Performance
#####################

nGraph currently uses ``nbench`` to assess and debug performance. 

This tool can benchmark any nGraph ``.json`` model with a given backend.

Compile and run with:

.. code-block:: console

   $ g++ ./nbench.cpp
             -std=c++11
             -I$HOME/ngraph_dist/include
             -L$HOME/ngraph_dist/lib
             -lngraph
             -o nbench
   $ env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib env NGRAPH_INTERPRETER_EMIT_TIMING=1 ./nbench


See also: ``ngraph/test/models``.
