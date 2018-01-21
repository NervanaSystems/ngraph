.. installation:

Building the Intel® nGraph™ library 
####################################

Build Environments
==================

The |release| version of |project| supports a Linux\*-based system, 
one updated with the following packages and system prerequisites: 

.. csv-table::
   :header: "Operating System", "Compiler", "Build System", "Status", "Additional Packages"
   :widths: 25, 15, 25, 20, 25
   :escape: ~

   Ubuntu 16.04 (LTS) 64-bit, CLang 3.9, CMake 3.5.1 + GNU Make, supported, `build-essential cmake clang-3.9 libtinfo-dev`
   Ubuntu 16.04 (LTS) 64-bit, CLang 4.0, CMake 3.5.1 + GNU Make, officially unsupported, `build-essential cmake clang-4.0 libtinfo-dev`


Installation Steps
==================

.. note:: If you are developing |nGL| projects on macOS*\, please be 
   aware that this platform is officially unsupported; see the section 
   `macOS Development Prerequisites`_ below.

To build ``libngraph`` on one a platform of your choice, the following 
procedure will  


#. Decide where you want your library to live and create a top-level directory
   for it.     

#. Clone the `NervanaSystems` ``ngraph-cpp`` repo.

   .. code-block:: console

      $ git clone git@github.com:NervanaSystems/private-ngraph-cpp.git
      $ cd private-ngraph-cpp


#. Create a build directory outside of ``ngraph-cpp/src`` directory tree; 
   something like  ``ngraph-cpp/build`` should work.

.. code-block:: console

      $ git clone git@github.com:NervanaSystems/private-ngraph-cpp.git
   


#. ``$ cd`` to the build directory.
#. Generate the GNUMakefiles in the customary manner from within your ``build``
   directory:

   .. code-block:: console

      $ cmake ../

#. Run ``$ make -j8``.
#. Run ``$ make install`` to install ``libngraph.so`` and the header files to 
   ``$HOME/ngraph_dist``.
#. (Optional, requires `doxygen`_.) Run ``$ make doc`` from within your ``build`` 
   directory to generate API documentation.
#. (Optional, requires `Sphinx`_.)  Run ``make html`` inside the  
   ``doc/sphinx`` directory to build HTML docs for the nGraph library.    

.. macOS Development Prerequisites:

macOS Development Prerequisites
-------------------------------

The repository includes two scripts (``maint/check-code-format.sh`` and 
``maint/apply-code-format.sh``) that are used respectively to check adherence 
to `libngraph` code formatting conventions, and to automatically reformat code 
according to those conventions. These scripts require the command 
``clang-format-3.9`` to be in your ``PATH``. Run the following commands 
(you will need to adjust them if you are not using bash):

.. code-block:: bash

  $ brew install llvm@3.9
  $ mkdir -p $HOME/bin
  $ ln -s /usr/local/opt/llvm@3.9/bin/clang-format $HOME/bin/clang-format-3.9
  $ echo 'export PATH=$HOME/bin:$PATH' >> $HOME/.bash_profile


Testing `libngraph`
===================

The `libngraph` code base uses the GTest framework for unit tests. CMake 
automatically downloads a copy of the required GTest files when configuring the 
build directory.

To perform the unit tests:

#. Create and configure the build directory as described above.
#. Enter the build directory.
#. Run ``$ make check``.

Using `libngraph`
=================

From Tensorflow as XLA plugin
------------------------------

.. warning:: Section below is a Work in Progress.

#. Get the following fork of the TF from this repo: ``git@github.com:NervanaSystems/ngraph-tensorflow.git``
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



System Requirements
====================
TBD

External library requirements
==============================
TBD

Maintaining ``libngraph``
=========================
TBD

Code formatting
================

All C/C++ source code in the ``libngraph`` repository, including the test code 
when practical, should adhere to the project's source-code formatting guidelines.

The script ``maint/apply-code-format.sh`` enforces that formatting at the C/C++ 
syntactic level. 

The script at ``maint/check-code-format.sh`` verifies that the formatting rules 
are met by all C/C++ code (again, at the syntax level.)  The script has an exit 
code of ``0`` when this all code meets the standard; and non-zero otherwise.  
This script does *not* modify the source code.


.. _doxygen: https://www.stack.nl/~dimitri/doxygen/
.. _Sphinx:  http://www.sphinx-doc.org/en/stable/
.. _NervanaSystems: https://github.com/NervanaSystems/private-ngraph-cpp/blob/master/README.md

