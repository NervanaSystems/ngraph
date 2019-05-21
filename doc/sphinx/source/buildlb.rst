.. buildlb.rst:

###############
Build and Test 
###############

This section details how to build the C++ version of the nGraph Library, which 
is targeted toward developers working on kernel-specific operations, 
optimizations, or on deep learning solutions that leverage custom backends. 

* :ref:`ubuntu`
* :ref:`centos`


Prerequisites
=============

Release |release| of |project| supports Linux\*-based systems with the following 
packages and prerequisites: 

.. csv-table::
   :header: "Operating System", "Compiler", "Build System", "Status", "Additional Packages"
   :widths: 25, 15, 25, 20, 25
   :escape: ~

   CentOS 7.4 64-bit, GCC 4.8, CMake 3.9.0, supported, ``wget zlib-devel ncurses-libs ncurses-devel patch diffutils gcc-c++ make git perl-Data-Dumper`` 
   Ubuntu 16.04 or 18.04 (LTS) 64-bit, Clang 3.9, CMake 3.5.1 + GNU Make, supported, ``build-essential cmake clang-3.9 clang-format-3.9 git curl zlib1g zlib1g-dev libtinfo-dev unzip autoconf automake libtool``
   Clear Linux\* OS for Intel® Architecture version 28880, Clang 8.0, CMake 3.14.2, experimental, bundles ``machine-learning-basic c-basic python-basic python-basic-dev dev-utils``

Other configurations may work, but should be considered experimental with
limited support. On Ubuntu 16.04 with gcc-5.4.0 or clang-3.9, for example, we 
recommend adding ``-DNGRAPH_USE_PREBUILT_LLVM=TRUE`` to the cmake command in 
step 4 below. This fetches a pre-built tarball of LLVM+Clang from llvm.org, 
and it will substantially reduce build time.

If using ``gcc`` version 4.8, it may be necessary to add symlinks from ``gcc`` 
to ``gcc-4.8``, and from ``g++`` to ``g++-4.8``, in your :envvar:`PATH`, even 
if you explicitly specify the ``CMAKE_C_COMPILER`` and ``CMAKE_CXX_COMPILER`` 
flags when building. (**Do NOT** supply the ``-DNGRAPH_USE_PREBUILT_LLVM`` 
flag in this case, because the prebuilt tarball supplied on llvm.org is not 
compatible with a gcc 4.8-based build.)


The ``default`` build
---------------------

Running ``cmake`` with no build flags defaults to the following settings; see
the ``CMakeLists.txt`` file for other experimental options' details: 

.. code-block:: console 

   -- NGRAPH_UNIT_TEST_ENABLE:         ON
   -- NGRAPH_TOOLS_ENABLE:             ON
   -- NGRAPH_CPU_ENABLE:               ON
   -- NGRAPH_INTELGPU_ENABLE:          OFF
   -- NGRAPH_GPU_ENABLE:               OFF
   -- NGRAPH_INTERPRETER_ENABLE:       ON
   -- NGRAPH_NOP_ENABLE:               ON
   -- NGRAPH_GPUH_ENABLE:              OFF
   -- NGRAPH_GENERIC_CPU_ENABLE:       OFF
   -- NGRAPH_DEBUG_ENABLE:             OFF  # Set to "ON" to enable logging
   -- NGRAPH_ONNX_IMPORT_ENABLE:       OFF
   -- NGRAPH_DEX_ONLY:                 OFF
   -- NGRAPH_CODE_COVERAGE_ENABLE:     OFF
   -- NGRAPH_LIB_VERSIONING_ENABLE:    OFF
   -- NGRAPH_PYTHON_BUILD_ENABLE:      OFF
   -- NGRAPH_USE_PREBUILT_LLVM:        OFF
   -- NGRAPH_PLAIDML_ENABLE:           OFF
   -- NGRAPH_DISTRIBUTED_ENABLE:       OFF


.. important:: The default :program:`cmake` procedure (no build flags) will  
   install ``ngraph_dist`` to an OS-level location like ``/usr/bin/ngraph_dist``
   or ``/usr/lib/ngraph_dist``. Here we specify how to build locally to the
   location of ``~/ngraph_dist`` with the cmake target ``-DCMAKE_INSTALL_PREFIX=~/ngraph_dist``. 


All of the nGraph Library documentation presumes that ``ngraph_dist`` gets 
installed locally. The system location can be used just as easily by customizing 
paths on that system. See the :file:`ngraph/CMakeLists.txt` file to change or 
customize the default CMake procedure.


Build steps
-----------

.. _ubuntu:

Ubuntu LTS
~~~~~~~~~~

The process documented here will work on Ubuntu\* 16.04 (LTS) or on Ubuntu 
18.04 (LTS).

#. Clone the `NervanaSystems` ``ngraph`` repo:

   .. code-block:: console

      $ git clone https://github.com/NervanaSystems/ngraph.git
      $ cd ngraph

#. Create a build directory outside of the ``ngraph/src`` directory 
   tree; somewhere like ``ngraph/build``, for example:

   .. code-block:: console

      $ mkdir build && cd build

#. Generate the GNU Makefiles in the customary manner (from within the 
   ``build`` directory). This command enables ONNX support in the library  
   and sets the target build location at ``~/ngraph_dist``, where it can be 
   found easily.  

   .. code-block:: console

      $ cmake .. -DNGRAPH_ONNX_IMPORT_ENABLE=ON  -DCMAKE_INSTALL_PREFIX=~/ngraph_dist

   **Other optional build flags** -- If running ``gcc-5.4.0`` or ``clang-3.9``, 
   remember that you can also append ``cmake`` with the prebuilt LLVM option 
   to speed-up the build.  Another option if your deployment system has Intel® 
   Advanced Vector Extensions (Intel® AVX) is to target the accelerations 
   available directly by compiling the build as follows during the cmake 
   step: ``-DNGRAPH_TARGET_ARCH=skylake-avx512``.  
   
   .. code-block:: console

      $ cmake .. [-DNGRAPH_USE_PREBUILT_LLVM=OFF] [-DNGRAPH_TARGET_ARCH=skylake-avx512]   

#. Run ``$ make`` and ``make install`` to install ``libngraph.so`` and the 
   header files to ``~/ngraph_dist``:

   .. code-block:: console
      
      $ make   # note: make -j <N> may work, but sometimes results in out-of-memory errors if too many compilation processes are used
      $ make install          

#. (Optional, requires `doxygen`_, `Sphinx`_, and `breathe`_). Run ``make html`` 
   inside the ``doc/sphinx`` directory of the cloned source to build a copy of 
   the `website docs`_ locally. The low-level API docs with inheritance and 
   collaboration diagrams can be found inside the ``/docs/doxygen/`` directory. 
   See the :doc:`project/doc-contributor-README` for more details about how to 
   build documentation for nGraph. 


.. _centos: 

CentOS 7.4
~~~~~~~~~~

The process documented here will work on CentOS 7.4.

#. Update the system with :command:`yum` and issue the following commands: 
   
   .. code-block:: console

      $ sudo yum update
      $ sudo yum install zlib-devel install ncurses-libs ncurses-devel patch diffutils wget gcc-c++ make git perl-Data-Dumper


#. Install Cmake 3.4:

   .. code-block:: console
    
      $ wget https://cmake.org/files/v3.4/cmake-3.5.0.tar.gz      
      $ tar -xzvf cmake-3.5.0.tar.gz
      $ cd cmake-3.5.0
      $ ./bootstrap --system-curl --prefix=~/cmake
      $ make && make install     

#. Clone the `NervanaSystems` ``ngraph`` repo via HTTPS and use Cmake 3.5.0 to 
   build nGraph Libraries to ``~/ngraph_dist``. This command enables ONNX 
   support in the library (optional). 

   .. code-block:: console

      $ cd /opt/libraries 
      $ git clone https://github.com/NervanaSystems/ngraph.git
      $ cd ngraph && mkdir build && cd build
      $ ~/cmake/bin/cmake .. -DCMAKE_INSTALL_PREFIX=~/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=ON 
      $ make && sudo make install 


macOS\* development
--------------------

.. note:: Although we do not currently offer full support for the macOS platform, 
   some configurations and features may work.

The repository includes two scripts (``maint/check-code-format.sh`` and 
``maint/apply-code-format.sh``) that are used respectively to check adherence 
to ``libngraph`` code formatting conventions, and to automatically reformat code 
according to those conventions. These scripts require the command 
``clang-format-3.9`` to be in your ``PATH``. Run the following commands 
(you will need to adjust them if you are not using bash):

.. code-block:: bash

   $ brew install llvm@3.9 automake
   $ mkdir -p $HOME/bin
   $ ln -s /usr/local/opt/llvm@3.9/bin/clang-format $HOME/bin/clang-format-3.9
   $ echo 'export PATH=$HOME/bin:$PATH' >> $HOME/.bash_profile

Testing the build 
=================

We use the `googletest framework`_ from Google for unit tests. The ``cmake`` 
command automatically downloaded a copy of the needed ``gtest`` files when 
it configured the build directory.

To perform unit tests on the install:

#. Create and configure the build directory as described in our 
   :doc:`buildlb` guide.

#. Enter the build directory and run ``make check``:
   
   .. code-block:: console

      $ cd build/
      $ make check


.. _doxygen: http://www.doxygen.nl/index.html
.. _Sphinx:  http://www.sphinx-doc.org/en/stable/
.. _breathe: https://breathe.readthedocs.io/en/latest/
.. _llvm.org: https://www.llvm.org 
.. _NervanaSystems: https://github.com/NervanaSystems/ngraph/blob/master/README.md
.. _ONNX: http://onnx.ai
.. _website docs: https://ngraph.nervanasys.com/docs/latest/
.. _googletest framework: https://github.com/google/googletest.git
