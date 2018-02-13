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

For this early |release| release, we're providing :doc:`framework-integration-guides`, 
for:

<<<<<<< HEAD
* :doc:`MXNet<framework-integration-guides>` framework,  
* :doc:`Tensorflow<framework-integration-guides>` framework, and
=======
* :doc:`framework-integration-guides` framework,  
* :doc:`framework-integration-guides` framework, and
>>>>>>> master
* neon™ `frontend framework`_.

Integration guides for other frameworks are tentatively forthcoming.

.. _GTest framework: https://github.com/google/googletest.git
.. _frontend framework: http://neon.nervanasys.com/index.html/

