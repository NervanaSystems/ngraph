.. framework-integration-guides:

#############################
Framework Integration Guides
#############################

* :ref:`neon_intg`
* :ref:`mxnet_intg`
* :ref:`tensorflow_intg`

.. _neon_intg:

neon |trade|
============

Use ``neon`` as a frontend for nGraph backends
-----------------------------------------------

``neon`` is an open source Deep Learning framework that has a history 
of `being the fastest`_ framework `for training CNN-based models with GPUs`_. 
Detailed info about neon's features and functionality can be found in the 
`neon docs`_. This section covers installing neon on an existing 
system that already has an ``ngraph_dist`` installed. 

.. important:: The numbered instructions below pick up from where 
   the :doc:`install` instructions left off, and they presume that your system 
   already has the ngraph library installed installed at ``$HOME/ngraph_dist`` 
   as the default location. If the |nGl| code has not yet been installed to 
   your system, you can follow the instructions on the `ngraph-neon python README`_ 
   to install everything at once.  


#. Set the ``NGRAPH_CPP_BUILD_PATH`` and the ``LD_LIBRARY_PATH`` path to the 
   location where you built the nGraph libraries. (This example shows the default 
   location):

   .. code-block:: bash

      export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist/
      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib/       

      
#. The neon framework uses the :command:`pip` package manager during installation; 
   install it with Python version 3.5 or higher:

   .. code-block:: console

      $ sudo apt-get install python3-pip python3-venv
      $ python3 -m venv frameworks
      $ cd frameworks 
      $ . bin/activate
      (frameworks) ~/frameworks$ 

#. Go to the "python" subdirectory of the ``ngraph`` repo we cloned during the 
   previous :doc:`install`, and complete these actions: 

   .. code-block:: console

      (frameworks)$ cd /opt/libraries/ngraph/python
      (frameworks)$ git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
      (frameworks)$ pip install -U . 

#. Finally we're ready to install the `neon` integration: 

   .. code-block:: console

      (frameworks)$ git clone git@github.com:NervanaSystems/ngraph-neon
      (frameworks)$ cd ngraph-neon
      (frameworks)$ make install

#. To test a training example, you can run the following from ``ngraph-neon/examples/cifar10``
   
   .. code-block:: console

      (frameworks)$ python cifar10_conv.py



.. _mxnet_intg:

MXNet\* 
========

Compile MXNet with nGraph
--------------------------

.. important:: These instructions pick up from where the :doc:`install`
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
   the ``USE_NGRAPH`` option (line ``100``) to true with `1` and set the :envvar:`NGRAPH_DIR`
   (line ``101``) to point to the installation location target where the |nGl|
   was installed:

   .. code-block:: bash

      USE_NGRAPH = 1
      NGRAPH_DIR = $(HOME)/ngraph_dist

#. Ensure that settings on the config file are disabled for ``USE_MKL2017``
   (line ``113``) and ``USE_NNPACK`` (line ``120``).

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



.. _tensorflow_intg:

TensorFlow\* 
=============

This section describes how install TensorFlow* with the bridge code 
needed to be able to access nGraph backends. Note that you **do not** 
need to have already installed nGraph for this procedure to work.  

Bridge TensorFlow/XLA to nGraph
-------------------------------

#. Prepare your system with the TensorFlow prerequisite, a system called 
   "bazel". These instructions were tested with `bazel version`_ 0.11.0. 

   .. code-block:: console

      $ wget https://github.com/bazelbuild/bazel/releases/download/0.11.0/bazel-0.11.0-installer-linux-x86_64.sh
      $ chmod +x bazel-0.11.0-installer-linux-x86_64.sh
      $ ./bazel-0.11.0-installer-linux-x86_64.sh --user

#. Add and source the ``bin`` path that bazel just created to your ``~/.bashrc`` 
   file in order to be able to call bazel from the user's installation we set up:

   .. code-block:: bash
   
      export PATH=$PATH:~/bin

   .. code-block:: console

      $ source ~/.bashrc   

#. Ensure that all the other TensorFlow dependencies are installed, as per the
   TensorFlow `installation guide`_:

   .. important:: CUDA is not needed. 

#. After TensorFlow's dependencies are installed, clone the source of the 
   `ngraph-tensorflow`_ repo to your machine; this is the required fork for 
   this integration. Many users may prefer to use a Python virtual env from 
   here forward:  

   .. code-block:: console

      $ python3 -m venv frameworks  
      $ cd frameworks 
      $ . bin/activate
      $ git clone git@github.com:NervanaSystems/ngraph-tensorflow.git
      $ cd ngraph-tensorflow
      $ git checkout ngraph-tensorflow-preview-0

#. Now run :command:`./configure` and choose `y` when prompted to build TensorFlow
   with XLA :abbr:`Just In Time (JIT)` support.

   .. code-block:: console
      :emphasize-lines: 6-7

      . . .

      Do you wish to build TensorFlow with Apache Kafka Platform support? [y/N]: n
      No Apache Kafka Platform support will be enabled for TensorFlow.

      Do you wish to build TensorFlow with XLA JIT support? [y/N]: y
      XLA JIT support will be enabled for TensorFlow.

      Do you wish to build TensorFlow with GDR support? [y/N]: 
      No GDR support will be enabled for TensorFlow.

      . . .

#. Build and install the pip package:

   .. code-block:: console

      $ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
      $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
      $ pip install -U /tmp/tensorflow_pkg/tensorflow-1.*whl


      .. note::  The actual name of the Python wheel file will be updated to the official 
         version of TensorFlow as the ngraph-tensorflow repository is synchronized frequently 
         with the original TensorFlow repository.

#. Now clone the ``ngraph-tensorflow-bridge`` repo one level above -- in the 
   parent directory of the ngraph-tensorflow repo cloned in step 4:

   .. code-block:: console

      $ cd ..
      $ git clone https://github.com/NervanaSystems/ngraph-tensorflow-bridge.git
      $ cd ngraph-tensorflow-bridge

#. Finally, build and install ngraph-tensorflow-bridge

   .. code-block:: console

      $ mkdir build
      $ cd build
      $ cmake ../
      $ make install

This final step automatically downloads the necessary version of ngraph and the 
dependencies. The resulting plugin `DSO`_ named ``libngraph_plugin.so`` gets copied 
to the following directory inside the TensorFlow installation directory: 

:: 

   <Python site-packages>/tensorflow/plugins

Once the build and installation steps are complete, you can start experimenting with 
coding for nGraph. 


Run MNIST Softmax with the activated bridge
----------------------------------------------

To see everything working together, you can run MNIST Softmax example with the now-activated 
bridge to nGraph. The script named mnist_softmax_ngraph.py can be found under the 
ngraph-tensorflow-bridge/test directory. It was modified from the example explained 
in the TensorFlow\* tutorial; the following changes were made from the original script:

.. code-block:: python

   def main(_):
   with tf.device('/device:NGRAPH:0'):
     run_mnist(_)

   def run_mnist(_):
     # Import data
     mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
     ...

To test everything together, set the configuration options:

.. code-block:: bash

   export OMP_NUM_THREADS=4 
   export KMP_AFFINITY=granularity=fine,scatter

And run the script as follows from within the `/test`_ directory of 
your cloned version of `ngraph-tensorflow-bridge`_:

.. code-block:: console   

   $ python mnist_softmax_ngraph.py


.. note:: The number-of-threads parameter specified in the ``OMP_NUM_THREADS`` 
   is a function of number of CPU cores that are available in your system. 


.. _MXNet: http://mxnet.incubator.apache.org
.. _bazel version: https://github.com/bazelbuild/bazel/releases/tag/0.11.0
.. _DSO: http://csweb.cs.wfu.edu/%7Etorgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
.. _installation guide: https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux
.. _ngraph-tensorflow: https://github.com/NervanaSystems/ngraph-tensorflow
.. _ngraph-tensorflow-bridge: https://github.com/NervanaSystems/ngraph-tensorflow-bridge
.. _/test: https://github.com/NervanaSystems/ngraph-tensorflow-bridge/tree/master/test
.. _ngraph-neon python README: https://github.com/NervanaSystems/ngraph/blob/master/python/README.md
.. _ngraph-neon repo's README: https://github.com/NervanaSystems/ngraph-neon/blob/master/README.md
.. _neon docs: https://github.com/NervanaSystems/neon/tree/master/doc
.. _being the fastest: https://github.com/soumith/convnet-benchmarks/
.. _for training CNN-based models with GPUs: https://www.microway.com/hpc-tech-tips/deep-learning-frameworks-survey-tensorflow-torch-theano-caffe-neon-ibm-machine-learning-stack/
