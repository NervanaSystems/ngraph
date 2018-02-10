.. testing-libngraph:

########################
Test the nGraph library
########################

The |InG| library code base uses the `GTest framework`_ for unit tests. CMake 
automatically downloads a copy of the required GTest files when configuring the 
build directory.

To perform the unit tests:

#. Create and configure the build directory as described in our 
   :doc:`installation` guide.

#. Enter the build directory and run ``make check``:
   
   .. code-block:: console

      $ cd build/
      $ make check


Compiling a framework with ``libngraph``
========================================

After building and installing the nGraph library to your system, the next 
logical step is to compile a framework that you can use to run a 
training/inference model with one of the backends that are now enabled.

For this early |release| release, we're providing integration guides for:

* `MXNet`_,  
* `TensorFlow`_, and
* neon™ `frontend framework`_.

Integration guides for other frameworks are tentatively forthcoming.

.. _GTest framework: https://github.com/google/googletest.git
.. _MXNet: http://mxnet.incubator.apache.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _frontend framework: http://neon.nervanasys.com/index.html/

