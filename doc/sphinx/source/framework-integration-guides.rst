.. framework-integration-guides:

#############################
Framework Integration Guides
#############################

.. contents::


.. mxnet_intg:

Compile MXNet with ``libngraph``
================================

.. important:: These instructions pick up from where the :doc:`installation`
   installation instructions left off, so they presume that your system already
   has the library installed at ``$HOME/ngraph_dist`` as the default location.
   If the |nGl| code has not yet been installed to your system, please go back
   and return here to finish compiling MXNet with ``libngraph``.

#. Set the ``LD_LIBRARY_PATH`` path to the location where we built the nGraph 
   libraries:

   .. code-block:: bash

      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib/


#. Add the `MXNet`_ prerequisites to your system, if the system doesn't have them
   already. These requirements are Ubuntu\*-specific.

   .. code-block:: console

      $ sudo apt-get install -y libopencv-dev curl libatlas-base-dev python
      python-pip python-dev python-opencv graphviz python-scipy python-sklearn
      libopenblas-dev


#. Clone the ``ngraph-mxnet`` repository recursively and checkout the
   ``ngraph-integration-dev`` branch:

   .. code-block:: console

      $ git clone --recursive git@github.com:NervanaSystems/ngraph-mxnet.git
      $ cd ngraph-mxnet && git checkout ngraph-integration-dev

#. Edit the ``make/config.mk`` file from the repo we just checked out to set
   the ``USE_NGRAPH`` option (line ``80``) to true with `1` and set the :envvar:`NGRAPH_DIR`
   (line ``81``) to point to the installation location target where the |nGl|
   was installed:

   .. code-block:: bash

      USE_NGRAPH = 1
      NGRAPH_DIR = $(HOME)/ngraph_dist

#. Ensure that settings on the config file are disabled for ``USE_MKL2017``
   (line ``93``) and ``USE_NNPACK`` (line ``100``).

   .. code-block:: bash

      # whether use MKL2017 library
      USE_MKL2017 = 0

      # whether use MKL2017 experimental feature for high performance
      # Prerequisite USE_MKL2017=1
      USE_MKL2017_EXPERIMENTAL = 0

      # whether use NNPACK library
      USE_NNPACK = 0


#. Finally, compile MXNet with |InG|:

   .. code-block:: console

      $ make -j $(nproc)

#. After successfully running ``make``, install the Python integration packages
   that your MXNet build needs to run a training example.

   .. code-block:: console

      $ cd python && pip install -e . && cd ../

#. Confirm a successful integration by running the MNIST training example:

   .. code-block:: console

      $ python example/image-classification/train_mnist.py



.. tensorflow_intg:

Building TensorFlow\* with an XLA plugin to ``libngraph``
=========================================================

.. important:: These instructions pick up where the :doc:`installation` 
   installation instructions left off, so they presume that your system already
   has the |nGl| installed. If the |nGl| code has not yet been installed to
   your system, please go back to complete those steps, and return here when
   you are ready to build TensorFlow\*.


#. Set the ``LD_LIBRARY_PATH`` path to the location where we built the nGraph 
   libraries:

   .. code-block:: bash

      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib/

#. To prepare to build TensorFlow with an XLA plugin capable of running |nGl|, 
   use the standard build process which is a system called "bazel". These 
   instructions were tested with `bazel version 0.5.4`_. 

   .. code-block:: console

      $ wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh
      $ chmod +x bazel-0.5.4-installer-linux-x86_64.sh
      $ ./bazel-0.5.4-installer-linux-x86_64.sh --user

#. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

   .. code-block:: bash
   
      export PATH=$PATH:~/bin

   .. code-block:: console

      $ source ~/.bashrc   

#. Ensure that all the TensorFlow 1.3 dependencies are installed, as per the
   TensorFlow `1.3 installation guide`_:

   .. note:: You do not need CUDA in order to use the nGraph XLA plugin.

#. Once TensorFlow's dependencies are installed, clone the source of the 
   `ngraph-tensorflow`_ repo to your machine; this is the required fork for 
   this integration:

   .. code-block:: console

      $ git clone git@github.com:NervanaSystems/ngraph-tensorflow.git
      $ cd ngraph-tensorflow

#. Now run :command:`configure` and choose `y` when prompted to build TensorFlow
   with XLA just-in-time compiler.

   .. code-block:: console
      :emphasize-lines: 5-6

      . . .

      Do you wish to build TensorFlow with Hadoop File System support? [y/N]
      No Hadoop File System support will be enabled for TensorFlow
      Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] y
      XLA JIT support will be enabled for TensorFlow
      Do you wish to build TensorFlow with VERBS support? [y/N]
      No VERBS support will be enabled for TensorFlow
      Do you wish to build TensorFlow with OpenCL support? [y/N]

      . . .

#. Next build the pip package

   .. code-block:: console

      $ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
      $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

#. Finally install the pip package

   .. code-block:: console

      $ pip install /tmp/tensorflow_pkg/tensorflow-1.3.0-cp27-cp27mu-linux_x86_64.whl


Run MNIST MLP through the TensorFlow / XLA plugin to nGraph
------------------------------------------------------------

To test an example through the TensorFlow / XLA plugin to nGraph, you can use the 
the MNIST softmax regression example script named `mnist_softmax_ngraph.py` that
is available in the `/examples/mnist`_ directory.

This script was modified from the example explained in the TensorFlow\* tutorial;
the following changes were made from the original script:

.. code-block:: python

   def main(_):
   with tf.device('/device:XLA_NGRAPH:0'):
     run_mnist(_)

   def run_mnist(_):
     # Import data
     mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
     ...

To test everything together, set the configuration options:

.. code-block:: bash

   export OMP_NUM_THREADS=4 
   export KMP_AFFINITY=granularity=fine,scatter

And run the script as follows from within the `/examples/mnist`_ directory of 
your cloned version of `ngraph-tensorflow`_:

.. code-block:: console   

   $ python mnist_softmax_ngraph.py


.. _MXNet: http://mxnet.incubator.apache.org/
.. _bazel version 0.5.4: https://github.com/bazelbuild/bazel/releases/tag/0.5.4
.. _1.3 installation guide: https://www.tensorflow.org/versions/r1.3/install/install_sources#prepare_environment_for_linux
.. _ngraph-tensorflow: https://github.com/NervanaSystems/ngraph-tensorflow
.. _/examples/mnist: https://github.com/NervanaSystems/ngraph-tensorflow/tree/develop/tensorflow/compiler/plugin/ngraph/examples/mnist
