.. install:

########
Install 
########

Build Environments
==================

The |release| version of |project| supports Linux\*-based systems  
with the following packages and prerequisites: 

.. csv-table::
   :header: "Operating System", "Compiler", "Build System", "Status", "Additional Packages"
   :widths: 25, 15, 25, 20, 25
   :escape: ~

   CentOS 7.4 64-bit, GCC 4.8, CMake 3.2, supported, ``patch diffutils zlib1g-dev libtinfo-dev`` 
   Ubuntu 16.04 (LTS) 64-bit, CLang 3.9, CMake 3.5.1 + GNU Make, supported, ``build-essential cmake clang-3.9 git zlib1g libtinfo-dev``
   Clear Linux\* OS for Intel Architecture, CLang 5.0.1, CMake 3.10.2, experimental, bundles ``machine-learning-basic dev-utils python3-basic python-basic-dev``

Other configurations may work, but should be considered experimental with
limited support. On Ubuntu 16.04 with ``gcc-5.4.0`` or ``clang-3.9``, for 
example, we recommend adding ``-DNGRAPH_USE_PREBUILT_LLVM=TRUE`` to the 
:command:`cmake` command in step 4 below. This fetches a pre-built tarball 
of LLVM+Clang from `llvm.org`_, and will substantially reduce build time.

If using ``gcc`` version 4.8, it may be necessary to add symlinks from ``gcc`` 
to ``gcc-4.8``, and from ``g++`` to ``g++-4.8``, in your :envvar:`PATH`, even 
if you explicitly specify the ``CMAKE_C_COMPILER`` and ``CMAKE_CXX_COMPILER`` 
flags when building. (**Do NOT** supply the ``-DNGRAPH_USE_PREBUILT_LLVM`` 
flag in this case, because the prebuilt tarball supplied on llvm.org is not 
compatible with a gcc 4.8-based build.)

Support for macOS is limited; see the `macOS development`_ section at the end 
of this page for details.


Installation Steps
==================

The CMake procedure installs ``ngraph_dist`` to the installing user's ``$HOME`` 
directory as the default location. See the :file:`CMakeLists.txt` file for 
details about how to change or customize the install location.

#. (Optional) Create something like ``/opt/libraries`` and (with sudo), 
   give ownership of that directory to your user. Creating such a placeholder 
   can be useful if you'd like to have a local reference for APIs and 
   documentation, or if you are a developer who wants to experiment with 
<<<<<<< HEAD
   how to :doc:`../howto/execute` using resources available through the 
   code base.
=======
   :doc:`../howto/execute` graph computations using resources available 
   through the library.

   .. code-block:: console

      $ sudo mkdir -p /opt/libraries
      $ sudo chown -R username:username /opt/libraries
      $ cd /opt/libraries

#. Clone the `NervanaSystems` ``ngraph`` repo:

   .. code-block:: console

      $ git clone git@github.com:NervanaSystems/ngraph.git
      $ cd ngraph

#. Create a build directory outside of the ``ngraph/src`` directory 
   tree; somewhere like ``ngraph/build``, for example:

   .. code-block:: console

      $ mkdir build && cd build

#. Generate the GNUMakefiles in the customary manner (from within the 
   ``build`` directory). If running ``gcc-5.4.0`` or ``clang-3.9``, remember 
   that you can also append ``cmake`` with the prebuilt LLVM option to 
   speed-up the build:

   .. code-block:: console

      $ cmake ../ [-DNGRAPH_USE_PREBUILT_LLVM=TRUE]

#. Run ``$ make`` and ``make install`` to install ``libngraph.so`` and the 
   header files to ``$HOME/ngraph_dist``:

   .. code-block:: console
      
      $ make   # note: make -j <N> may work, but sometimes results in out-of-memory errors if too many compilation processes are used


#. (Optional, requires `doxygen`_, `Sphinx`_, and `breathe`_). Run ``make html`` 
   inside the ``doc/sphinx`` directory of the cloned source to build a copy of 
   the `website docs`_ locally. The low-level API docs with inheritance and 
   collaboration diagrams can be found inside the ``/docs/doxygen/`` directory.    

.. macos_development: 

macOS development
-----------------

.. note:: The macOS*\ platform is officially unsupported.

The repository includes two scripts (``maint/check-code-format.sh`` and 
``maint/apply-code-format.sh``) that are used respectively to check adherence 
to ``libngraph`` code formatting conventions, and to automatically reformat code 
according to those conventions. These scripts require the command 
``clang-format-3.9`` to be in your ``PATH``. Run the following commands 
(you will need to adjust them if you are not using bash):

.. code-block:: bash

   $ brew install llvm@3.9
   $ mkdir -p $HOME/bin
   $ ln -s /usr/local/opt/llvm@3.9/bin/clang-format $HOME/bin/clang-format-3.9
   $ echo 'export PATH=$HOME/bin:$PATH' >> $HOME/.bash_profile


Test 
====

The |InG| library code base uses GoogleTest's\* `googletest framework`_ 
for unit tests. The ``cmake`` command from the :doc:`install` guide 
automatically downloaded a copy of the needed ``gtest`` files when 
it configured the build directory.

To perform unit tests on the install:

#. Create and configure the build directory as described in our 
   :doc:`install` guide.

#. Enter the build directory and run ``make check``:
   
   .. code-block:: console

      $ cd build/
      $ make check


Compile a framework with ``libngraph``
======================================

After building and installing nGraph++ to your system, the next logical 
step is to compile a framework that you can use to run a DL training or 
inference model. If you've already extracted a model from a framework
by following a tutorial from `ONNX_`, and you have a exported, serialized 
file ready to be imported by the library, see our :doc:`../howto/handle`. 

For this early |release| release,  :doc:`framework-integration-guides`, 
can help you get started with a framework. 

* :doc:`MXNet<framework-integration-guides>` framework,  
* :doc:`TensorFlow<framework-integration-guides>` framework, and
* neon™ `frontend framework`_.

Integration guides for other frameworks are tentatively forthcoming.

.. _doxygen: https://www.stack.nl/~dimitri/doxygen/
.. _Sphinx:  http://www.sphinx-doc.org/en/stable/
.. _breathe: https://breathe.readthedocs.io/en/latest/
.. _llvm.org: https://www.llvm.org 
.. _NervanaSystems: https://github.com/NervanaSystems/ngraph/blob/master/README.md
.. _website docs: http://ngraph.nervanasys.com/index.html/index.html
.. _googletest framework: https://github.com/google/googletest.git
.. _ONNX: http://onnx.ai
.. _frontend framework: http://neon.nervanasys.com/index.html/
