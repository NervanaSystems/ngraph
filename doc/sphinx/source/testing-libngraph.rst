.. testing-libngraph:


Testing `libngraph`
###################

The `libngraph` code base uses the GTest framework for unit tests. CMake 
automatically downloads a copy of the required GTest files when configuring the 
build directory.

To perform the unit tests:

#. Create and configure the build directory as described in our :doc:`installation` guide.
#. Enter the build directory and run ``make check``.   
   
   .. code-block:: console

      $ cd build/
      $ make check


Compiling MXNet with ``libngraph``
==================================

.. TODO:  add Matt B's relevant wiki documentation here


Using ``libngraph`` from Tensorflow as XLA plugin
=================================================

.. TODO:  add Avijit's presentation info and process here 

.. warning:: Section below is a Work in Progress.

#. Get the `ngraph` fork of TensorFlow from this repo: ``git@github.com:NervanaSystems/ngraph-tensorflow.git``
#. Etc.
#. Go to the end near the following snippet

   ::

      native.new_local_repository(
      name = "ngraph_external",
      path = "/your/home/directory/where/ngraph_is_installed",
      build_file = str(Label("//tensorflow/compiler/plugin/ngraph:ngraph.BUILD")),
      )

   and modify the following line in the :file:`tensorflow/workspace.bzl` file to 
   provide an absolute path to ``~/ngraph_dist``
   
   ::
     
     path = "/directory/where/ngraph_is_installed"


#. Now run :command:`configure` and follow the rest of the TF build process.



Maintaining ``libngraph``
=========================
TBD


