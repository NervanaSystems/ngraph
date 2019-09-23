.. buildlb.rst:

###############
Build and Test
###############

* :ref:`default_ngflags`
* :ref:`ngraph_plaidml_backend`

There are a few common paths to take when manually building the |project| 
from source code. Today nGraph supports various developers working on all 
parts of the :abbr:`Deep Learning (DL)` stack, and the way you decide to 
build or install components ought to depend on the capabilities of your 
hardware, and how you intend to use it.

A "from scratch" source-code build of the nGraph Library enables the CPU, 
``Interpreter``, and unit tests by default. See :ref:`default_ngflags` 
for more detail.

A "from scratch" source-code build that defaults to the PlaidML backend 
contains rich algorithm libraries akin to those that were previously available 
only to developers willing to spend extensive time writing, testing, and 
customizing kernels. An ``NGRAPH_PLAIDML`` dist can function like a framework 
that lets developers compose, train, and even deploy :abbr:`DL (Deep Learning)` 
models in their preferred language on neural networks of any size. This is 
a good option if, for example, you are working on a laptop with a high-end 
GPU that you want to use for compute. See :ref:`ngraph_plaidml_backend` 
for instructions on how to build.

In either case, there are some prerequisites that your system will need 
to build from sources.

.. _prerequisites:

Prerequisites
-------------

.. csv-table::
   :header: "Operating System", "Compiler", "Build System", "Status", "Additional Packages"
   :widths: 25, 15, 25, 20, 25
   :escape: ~

   CentOS 7.4 64-bit, GCC 4.8, CMake 3.9.0, supported, ``wget zlib-devel ncurses-libs ncurses-devel patch diffutils gcc-c++ make git perl-Data-Dumper`` 
   Ubuntu 16.04 or 18.04 (LTS) 64-bit, Clang 3.9, CMake 3.5.1 + GNU Make, supported, ``build-essential cmake clang-3.9 clang-format-3.9 git curl zlib1g zlib1g-dev libtinfo-dev unzip autoconf automake libtool``
   Clear Linux\* OS for Intel® Architecture version 28880, Clang 8.0, CMake 3.14.2, experimental, bundles ``machine-learning-basic c-basic python-basic python-basic-dev dev-utils``


.. _default_ngflags:

Building nGraph from source
===========================

.. important:: The default :program:`cmake` procedure (no build flags) will  
   install ``ngraph_dist`` to an OS-level location like ``/usr/bin/ngraph_dist``
   or ``/usr/lib/ngraph_dist``. Here we specify how to build locally to the
   location of ``~/ngraph_dist`` with the cmake target ``-DCMAKE_INSTALL_PREFIX=~/ngraph_dist``. 

All of the nGraph Library documentation presumes that ``ngraph_dist`` gets 
installed locally. The system location can be used just as easily by 
customizing paths on that system. See the :file:`ngraph/CMakeLists.txt` 
file to change or customize the default CMake procedure.

* :ref:`ubuntu`
* :ref:`centos`


.. _ubuntu:

Ubuntu LTS build steps
----------------------

The process documented here will work on Ubuntu\* 16.04 (LTS) or on Ubuntu 
18.04 (LTS).

#. Ensure you have installed the :ref:`prerequisites` for Ubuntu\*.

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

CentOS 7.4 build steps
----------------------

The process documented here will work on CentOS 7.4.

#. Ensure you have installed the :ref:`prerequisites` for CentOS\*, 
   and update the system with :command:`yum`.

   .. code-block:: console

      $ sudo yum update

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


.. _ngraph_plaidml_backend:

Building nGraph-PlaidML from source
===================================

The following instructions will create the ``~/ngraph_plaidml_dist`` 
locally:

#. Ensure you have installed the :ref:`prerequisites` for your OS.

#. Install the prerequisites for the backend. Our hybrid ``NGRAPH_PLAIDML``
   backend works best with Python3 versions. We recommend that you use a 
   virtual environment, due to some of the difficulties that users have 
   seen when trying to install outside of a venv.

   .. code-block:: console

      $ sudo apt install python3-pip
      $ pip install plaidml 
      $ plaidml-setup

#. Clone the source code, create and enter your build directory:

   .. code-block:: console

      $ git clone https://github.com/NervanaSystems/ngraph.git
      $ cd ngraph && mkdir build && cd build

#. Prepare the CMake files as follows: 

   .. code-block:: console

      $ cmake .. -DCMAKE_INSTALL_PREFIX=~/ngraph_plaidml_dist -DNGRAPH_CPU_ENABLE=OFF -DNGRAPH_PLAIDML_ENABLE=ON 

#. Run :command:`make` and ``make install``. Note that if you are building 
   outside a local or user path, you may need to run ``make install`` as the 
   root user.

   .. code-block:: console

      $ make
      $ make install

   This should create the shared library ``libplaidml_backend.so`` and 
   nbench. Note that if you built in a virtual environment and run 
   ``make check`` from it, the Google Test may report failures. Full 
   tests can be run when PlaidML devices are available at the machine 
   level.

For more about working with the PlaidML backend from nGraph, see our 
API documentation :doc:`backends/plaidml-ng-api/index`. 


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
.. _PlaidML: https://github.com/plaidml/plaidml