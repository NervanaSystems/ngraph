.. installation:

Building the Intel® nGraph™ library 
####################################

Build Environments
==================

The |release| version of |project| supports Linux\* or UNIX-based 
systems where the system has been recently updated with the following 
packages and prerequisites: 

.. csv-table::
   :header: "Operating System", "Compiler", "Build System", "Status", "Additional Packages"
   :widths: 25, 15, 25, 20, 25
   :escape: ~

   CentOS 7.4 64-bit,, CMake 3.5.1 + GNU Make, supported,
   Ubuntu 16.04 (LTS) 64-bit, CLang 3.9, CMake 3.5.1 + GNU Make, supported, ``build-essential cmake clang-3.9 git libtinfo-dev``
   Ubuntu 16.04 (LTS) 64-bit, CLang 4.0, CMake 3.5.1 + GNU Make, officially unsupported, ``build-essential cmake clang-4.0 git libtinfo-dev``
   Clear Linux\* OS for Intel Architecture, Clang 5.0.1, CMake 3.10.2, experimental, bundles ``machine-learning-basic dev-utils python3-basic python-basic-dev``

Installation Steps
==================

To build |nGl| on one of the supported systems, the default CMake procedure 
will install ``ngraph_dist`` to your user's ``$HOME`` directory as
the default install location.  See the :file:`CMakeLists.txt` file for more 
information.

This guide provides one possible configuration that does not rely on a 
virtual environment. You are, of course, free to use a virtual environment, 
or to set up user directories and permissions however you like. 

#.  Since most of a developer's interaction with a frontend framework 
    will take place locally through Python, set a placeholder directory 
    where Python bindings can interact more efficiently with the nGraph 
    library backend components. Create something like ``/opt/local`` and 
    (presuming you have sudo permissions), give ownership of that local 
    directory to your user. This will make configuring for various ``PATH`` 
    and environmental variables much more simple later. 

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
   tree; something like  ``private-ngraph-cpp/build`` should work.

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

#. (COMING SOON -- optional, requires `doxygen`_.)  TBD



.. macOS Development Prerequisites:

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



External library requirements
==============================
TBD



.. _doxygen: https://www.stack.nl/~dimitri/doxygen/
.. _Sphinx:  http://www.sphinx-doc.org/en/stable/
.. _NervanaSystems: https://github.com/NervanaSystems/private-ngraph-cpp/blob/master/README.md

