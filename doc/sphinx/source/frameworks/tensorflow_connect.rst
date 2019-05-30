.. frameworks/tensorflow_connect.rst:

Getting Started with TensorFlow\*
=================================


No matter what your level of experience with :abbr:`Deep Learning (DL)` systems 
may be, nGraph provides a path to start working with the DL stack. Let's begin 
with the easiest and most straightforward options.

Using ``pip`` install
----------------------

If you are already familiar with TensorFlow\*, the easiest way to get started 
is to use the latest `prebuilt nGraph-TensorFlow bridge`_. You can install 
TensorFlow and nGraph to a virtual environment; otherwise, the code will install 
to a system location.

.. important:: The latest version of TensorFlow may be greater than the version 
   named below.

.. code-block:: console
   
   pip install -U tensorflow==1.13.1
   pip install -U ngraph-tensorflow-bridge

That's it!  Now you can test the installation by running the following command:

.. code-block:: python

   python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);
   import ngraph_bridge; 
   print(ngraph_bridge.__version__)"

Detailed examples are located in the `ngraph_bridge examples`_ directory. 


Build your own nGraph-TensorFlow bridge 
---------------------------------------

A slightly more involved option is to build the latest nGraph with a binary 
TensorFlow; use this option if you know you need features newer than those 
available in the prebuilt binary. Review the latest :doc:`../project/release-notes` 
for the most recent changes. 

To build your own bridge, follow these steps:

#. **Clone the nGraph-TensorFlow bridge repo**

    .. code-block:: console

       git clone https://github.com/tensorflow/ngraph-bridge.git

#. The script we need to run assumes you're running the ``pip`` implementation 
   of ``virtualenv``, on Python3.5 or greater.  The quickest way to get this on 
   your system is:

   .. code-block:: console

       python3 -m pip install virtualenv==16.0.0 --user


#. Run the script 

   .. code-block:: console

      python3 build_ngtf.py --use_prebuilt_tensorflow

#. Now test the build by executing the following commands:

   .. code-block:: console

       source build_cmake/venv-tf-py3/bin/activate
       PYTHONPATH=`pwd` python3 test/ci/buildkite/test_runner.py \
            --backend CPU \
            --artifacts ./build_cmake/artifacts/ \
            --test_resnet

That's it! Now you can take a look at and start experimenting with the detailed 
located in the `ngraph_bridge examples`_ directory. 

.. _prebuilt nGraph-TensorFlow bridge: https://github.com/tensorflow/ngraph-bridge#option-1-use-a-pre-built-ngraph-tensorflow-bridge
.. _Option 2: https://github.com/tensorflow/ngraph-bridge#option-2-build-ngraph-bridge-with-binary-tensorflow-installation
.. _`ngraph_bridge examples: https://github.com/tensorflow/ngraph-bridge/blob/master/examples/README.md
