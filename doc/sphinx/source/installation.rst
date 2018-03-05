.. installation:

########
Install 
########

Build Environments
==================

The |release| version of |project| supports Linux\*-based systems which have 
recent updates of the following packages and prerequisites: 

.. csv-table::
   :header: "Operating System", "Compiler", "Build System", "Status", "Additional Packages"
   :widths: 25, 15, 25, 20, 25
   :escape: ~

   CentOS 7.4 64-bit, GCC 4.8, CMake 3.2, supported, ``patch diffutils zlib1g-dev libtinfo-dev`` 
   Ubuntu 16.04 (LTS) 64-bit, CLang 3.9, CMake 3.5.1 + GNU Make, supported, ``build-essential cmake clang-3.9 git libtinfo-dev``
   Clear Linux\* OS for Intel Architecture, CLang 5.0.1, CMake 3.10.2, experimental, bundles ``machine-learning-basic dev-utils python3-basic python-basic-dev``

Other configurations may work, but aren't tested; on Ubuntu 16.04 with 
``gcc-5.4.0`` or ``clang-3.9``, for example, we recommend adding 
``-DNGRAPH_USE_PREBUILT_LLVM=TRUE`` to the :command:`cmake` command in step 4
below. This gets a pre-built tarball of LLVM+Clang from `llvm.org`_, and will
substantially reduce build time.

If using ``gcc-4.8``, it may be necessary to add symlinks from ``gcc`` to
``gcc-4.8``, and from ``g++`` to ``g++-4.8``, in your :envvar:`PATH`, even 
if you explicitly specify the ``CMAKE_C_COMPILER`` and ``CMAKE_CXX_COMPILER`` 
flags when building. (You **should NOT** supply the ``-DNGRAPH_USE_PREBUILT_LLVM`` 
flag in this case, because the prebuilt tarball supplied on llvm.org is not 
compatible with a gcc-4.8 based build.)

Support for macOS is limited; see the `macOS development`_ section at the end of 
this page for details.


Installation Steps
==================

To build |nGl| on one of the supported systems, the CMake procedure will 
install ``ngraph_dist`` to the installing user's ``$HOME`` directory as
the default location. See the :file:`CMakeLists.txt` file for more 
information about how to change or customize this location.

#.  (Optional) Create something like ``/opt/local`` and (with sudo permissions), 
    give ownership of that directory to your user. Under this directory, you can 
    add a placeholder for ``libraries`` to have a placeholder for the documented 
    source cloned from the repo: 

    .. code-block:: console

       $ cd /opt
       $ sudo mkdir -p local/libraries
       $ sudo chown -R username:username /opt/local

#. Clone the `NervanaSystems` ``ngraph-cpp`` repo to your `/libraries`
   directory.

   .. code-block:: console

      $ cd /opt/local/libraries
      $ git clone git@github.com:NervanaSystems/ngraph-cpp.git
      $ cd ngraph-cpp

#. Create a build directory outside of the ``ngraph-cpp/src`` directory 
   tree; somewhere like ``ngraph-cpp/build``, for example.

   .. code-block:: console

      $ mkdir build   

#. ``$ cd`` to the build directory and generate the GNUMakefiles in the 
   customary manner from within your ``build`` directory (remember to append the 
   command with the prebuilt option, if needed):

   .. code-block:: console

      $ cd build && cmake ../ [-DNGRAPH_USE_PREBUILT_LLVM=TRUE]

#. (Optional) Run ``$ make [-jN]`` where :option:`-jN` specifies the number of 
   cores. The example here uses a configuration of :option:`j8`, which is 
   good for a system install using an Intel® Xeon® (CPU processor). This step 
   is **not recommended** with Docker / VM installs. 

   .. code-block:: console
      
      $ make -j8

#. Run ``make install`` to install ``libngraph.so`` and the header files to the 
   default location of ``$HOME/ngraph_dist``

   .. code-block:: console

      $ make install

#. (Optional, requires `doxygen`_, `Sphinx`_, and `breathe`_). Run ``make html`` 
   inside the ``doc/sphinx`` directory of the cloned source to build a copy of 
   the `website docs`_ locally. The low-level API docs with inheritance diagrams 
   and collaboration diagrams can be found inside the ``/docs/doxygen/`` 
   directory.    

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

.. _doxygen: https://www.stack.nl/~dimitri/doxygen/
.. _Sphinx:  http://www.sphinx-doc.org/en/stable/
.. _breathe: https://breathe.readthedocs.io/en/latest/
.. _llvm.org: https://www.llvm.org 
.. _NervanaSystems: https://github.com/NervanaSystems/ngraph-cpp/blob/master/README.md
.. _website docs: http://ngraph.nervanasys.com/index.html/index.html
