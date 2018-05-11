.. framework-integration-guides:

###############################
Integrate Supported Frameworks
###############################

* :ref:`mxnet_intg`
* :ref:`tensorflow_intg`
* :ref:`neon_intg`


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


#. Clone the ``ngraph-mxnet`` repository recursively

   .. code-block:: console

      $ git clone --recursive git@github.com:NervanaSystems/ngraph-mxnet.git

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

See the `ngraph tensorflow bridge README`_ for how to install the 
nGraph-TensorFlow bridge.



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
      (frameworks)$ export PYBIND_HEADERS_PATH=/opt/libraries/ngraph/python/pybind11
      (frameworks)$ pip install -U . 

#. Finally we're ready to install the `neon` integration: 

   .. code-block:: console

      (frameworks)$ git clone git@github.com:NervanaSystems/ngraph-neon
      (frameworks)$ cd ngraph-neon
      (frameworks)$ make install

#. To test a training example, you can run the following from ``ngraph-neon/examples/cifar10``
   
   .. code-block:: console

      (frameworks)$ python cifar10_conv.py




.. _MXNet: http://mxnet.incubator.apache.org
.. _DSO: http://csweb.cs.wfu.edu/%7Etorgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
.. _ngraph-neon python README: https://github.com/NervanaSystems/ngraph/blob/master/python/README.md
.. _ngraph neon repo's README: https://github.com/NervanaSystems/ngraph-neon/blob/master/README.md
.. _neon docs: https://github.com/NervanaSystems/neon/tree/master/doc
.. _being the fastest: https://github.com/soumith/convnet-benchmarks/
.. _for training CNN-based models with GPUs: https://www.microway.com/hpc-tech-tips/deep-learning-frameworks-survey-tensorflow-torch-theano-caffe-neon-ibm-machine-learning-stack/
.. _ngraph tensorflow bridge README: https://github.com/NervanaSystems/ngraph-tensorflow-bridge
