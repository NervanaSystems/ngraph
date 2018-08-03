.. framework-integration-guides:

###############################
Integrate Supported Frameworks
###############################

* :ref:`mxnet_intg`
* :ref:`tensorflow_intg`
* :ref:`neon_intg`

A framework is "supported" when there is a framework :term:`bridge` that can be 
cloned from one of our GitHub repos and built to connect to a supported backend
with nGraph, all the while maintaining the framework's programmatic or user 
interface. Current bridge-enabled frameworks include TensorFlow* and MXNet*. 

Once connected via the bridge, the framework can then run and train a deep 
learning model with various workloads on various backends using nGraph Compiler 
as an optimizing compiler available through the framework.  


.. _mxnet_intg:

MXNet\* bridge
===============

#. See the README on `nGraph-MXNet`_ Integration for how to enable the bridge.

#. (Optional) For experimental or alternative approaches to distributed training
   methodologies, including data parallel training, see the MXNet-relevant sections
   of the docs on :doc:`distr/index` and :doc:`How to <howto/index>` topics like
   :doc:`howto/distribute-train`. 


.. _tensorflow_intg:

TensorFlow\* bridge
===================

See the `ngraph tensorflow bridge README`_ for how to install the `DSO`_ for the 
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

.. important:: As of version |version|, these instructions presume that your 
   system already has the library installed to the default location, as outlined 
   in our :doc:`install` documentation. 


#. Set the ``NGRAPH_CPP_BUILD_PATH`` and the ``LD_LIBRARY_PATH``. You can use 
   the ``env`` command to see if these paths have been set already and if they 
   have not, they can be set with something like: 

   .. code-block:: bash

      export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist/
      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib/
      
#. The neon framework uses the :command:`pip` package manager during installation; 
   install it with Python version 3.5 or higher:

   .. code-block:: console

      $ sudo apt-get install python3-pip python3-venv
      $ python3 -m venv neon_venv
      $ cd neon_venv 
      $ . bin/activate
      (neon_venv) ~/frameworks$ 

#. Go to the "python" subdirectory of the ``ngraph`` repo we cloned during the 
   previous :doc:`install`, and complete these actions: 

   .. code-block:: console

      (neon_venv)$ cd /opt/libraries/ngraph/python
      (neon_venv)$ git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
      (neon_venv)$ export PYBIND_HEADERS_PATH=/opt/libraries/ngraph/python/pybind11
      (neon_venv)$ pip install -U . 

#. Finally we're ready to install the `neon` integration: 

   .. code-block:: console

      (neon_venv)$ git clone git@github.com:NervanaSystems/ngraph-neon
      (neon_venv)$ cd ngraph-neon
      (neon_venv)$ make install

#. To test a training example, you can run the following from ``ngraph-neon/examples/cifar10``
   
   .. code-block:: console

      (neon_venv)$ python cifar10_conv.py

#. (Optional) For experimental or alternative approaches to distributed training
   methodologies, including data parallel training, see the :doc:`distr/index` 
   and :doc:`How to <howto/index>` articles on :doc:`howto/distribute-train`. 


.. _nGraph-MXNet: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/NGRAPH_README.md
.. _MXNet: http://mxnet.incubator.apache.org
.. _DSO: http://csweb.cs.wfu.edu/%7Etorgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
.. _ngraph-neon python README: https://github.com/NervanaSystems/ngraph/blob/master/python/README.md
.. _ngraph neon repo's README: https://github.com/NervanaSystems/ngraph-neon/blob/master/README.md
.. _neon docs: https://github.com/NervanaSystems/neon/tree/master/doc
.. _being the fastest: https://github.com/soumith/convnet-benchmarks
.. _for training CNN-based models with GPUs: https://www.microway.com/hpc-tech-tips/deep-learning-frameworks-survey-tensorflow-torch-theano-caffe-neon-ibm-machine-learning-stack
.. _ngraph tensorflow bridge README: https://github.com/NervanaSystems/ngraph-tf
