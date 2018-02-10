.. installation:

###################################
Install the Intel® nGraph™ library 
###################################

Build Environments
==================

The |release| version of |project| supports Linux\* or UNIX-based 
systems which have recent updates of the following packages and 
prerequisites: 

.. csv-table::
   :header: "Operating System", "Compiler", "Build System", "Status", "Additional Packages"
   :widths: 25, 15, 25, 20, 25
   :escape: ~

   CentOS 7.4 64-bit, CLang 3.4, GCC 4.8 + CMake 2.8, supported, ``patch diffutils zlib1g-dev libtinfo-dev`` 
   Ubuntu 16.04 (LTS) 64-bit, CLang 3.9, CMake 3.5.1 + GNU Make, supported, ``build-essential cmake clang-3.9 git libtinfo-dev``
   Ubuntu 16.04 (LTS) 64-bit, CLang 4.0, CMake 3.5.1 + GNU Make, officially unsupported, ``build-essential cmake clang-4.0 git libtinfo-dev``
   Clear Linux\* OS for Intel Architecture, CLang 5.0.1, CMake 3.10.2, experimental, bundles ``machine-learning-basic dev-utils python3-basic python-basic-dev``

On Ubuntu 16.04 with ``gcc-5.4.0`` or ``clang-3.9``, the recommended option 
is to add ``-DNGRAPH_USE_PREBUILT_LLVM=TRUE`` to the :command:`cmake` command. 
This gets a pre-built tarball of LLVM+Clang from `llvm.org`_, and substantially 
reduces build times.

If using ``gcc-4.8``, it may be necessary to add symlinksfrom ``gcc`` to
``gcc-4.8``, and from ``g++`` to ``g++-4.8``, in your :envvar:`PATH`, even 
if you explicitly specify the ``CMAKE_C_COMPILER`` and ``CMAKE_CXX_COMPILER`` 
flags when building. (You should NOT supply the `-DNGRAPH_USE_PREBUILT_LLVM` 
flag in this case, because the prebuilt tarball supplied on llvm.org is not 
compatible with a gcc-4.8 based build.)

Support for macOS is limited; see the macOS development prerequisites 
section at the end of this page for details.


Installation Steps
==================

To build |nGl| on one of the supported systems, the CMake procedure will 
install ``ngraph_dist`` to the installing user's ``$HOME`` directory as
the default location. See the :file:`CMakeLists.txt` file for more 
information about how to change or customize this location.

#.  (Optional) Since most of a developer's interaction with a frontend 
    framework will take place locally through Pythonic APIs to the C++
    library, you can set a reference placeholder for the documented source 
    cloned from the repo. Create something like ``/opt/local`` and (with sudo 
    permissions), give ownership of that directory to your user.  

    .. code-block:: console

       $ cd /opt
       $ sudo mkdir -p local/libraries
       $ sudo chown -R username:username /opt/local

#. Clone the `NervanaSystems` ``ngraph-cpp`` repo to your `/libraries`
   directory.

   .. code-block:: console

      $ cd /opt/local/libraries
      $ git clone git@github.com:NervanaSystems/private-ngraph-cpp.git
      $ cd private-ngraph-cpp

#. Create a build directory outside of the ``private-ngraph-cpp/src`` directory 
   tree; somewhere like ``private-ngraph-cpp/build``, for example.

   .. code-block:: console

      $ mkdir build   

#. ``$ cd`` to the build directory and generate the GNUMakefiles in the 
   customary manner from within your ``build`` directory:

   .. code-block:: console

      $ cd build && cmake ../

#. Run ``$ make -j8`` and ``make install`` to install ``libngraph.so`` and the 
   header files to the default location of ``$HOME/ngraph_dist``.

   .. code-block:: console

      $ make -j8 && make install 


#. (Optional, requires `Sphinx`_.)  Run ``make html`` inside the  
   ``doc/sphinx`` directory to build HTML docs for the nGraph library.    

#. (Optional, requires `doxygen`_.)  Run ``$ make htmldocs`` inside
   the ``doc/sphinx`` directory to build HTML API docs inside the 
   ``/docs/doxygen/`` directory. 


macOS Development Prerequisites
-------------------------------

.. note:: If you are developing |nGl| projects on macOS*\, please be 
   aware that this platform is officially unsupported.

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

.. _doxygen: https://www.stack.nl/~dimitri/doxygen/
.. _Sphinx:  http://www.sphinx-doc.org/en/stable/
.. _NervanaSystems: https://github.com/NervanaSystems/private-ngraph-cpp/blob/master/README.md
.. _llvm.org: https://www.llvm.org 

