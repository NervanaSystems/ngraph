.. framework-integration-guides: 

Framework Integration Guides
############################

.. contents::


Compile MXNet with ``libngraph``
================================

#. Add the `MXNet`_ prerequisites to your system, if the system doesn't have them
   already.  These requirements are Ubuntu\*-specific.  

   .. code-block:: console

      $ sudo apt-get install -y libopencv-dev curl libatlas-base-dev python 
      python-pip python-dev python-opencv graphviz python-scipy python-sklearn 
      libopenblas-dev
   
#. Set the ``LD_LIBRARY_PATH`` path to the location where we built the libraries:

   .. code-block:: bash

      export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib/

#. Clone the ``ngraph-mxnet`` repository recursively and checkout the 
   ``ngraph-integration-dev branch``:

   .. code-block:: console

      $ git clone --recursive git@github.com:NervanaSystems/ngraph-mxnet.git
      $ cd ngraph-mxnet && git checkout ngraph-integration-dev

#. Edit the ``make/config.mk`` file from the repo we just checked out to set 
   the ``USE_NGRAPH`` option to true with `1` and set :envvar:`NGRAPH_DIR` 
   to point to the installation:

   .. code-block:: bash

      USE_NGRAPH = 1
      NGRAPH_DIR = $(HOME)/ngraph_dist

#. Ensure that the config file has disabled nnpack and mklml.

#. Finally, compile MXNet:

   .. code-block:: console

      $ make -j $(nproc)

#. After successfully running ``make``, install the python integration packages 
   your MXNet build needs to run a training example.  

   .. code-block:: console

      $ cd python && pip install -e . && cd ../

#. Confirm a successful integration by running the MNIST training example: 

   .. code-block:: console
      
      $ python example/image-classification/train_mnist.py



Using ``libngraph`` from Tensorflow as XLA plugin
=================================================

.. TODO:  add Avijit's presentation info and process here 

.. warning:: Section below is a Work in Progress.

#. Get the `ngraph` fork of TensorFlow from this repo: ``git@github.com:NervanaSystems/ngraph-tensorflow.git``

#. Go to the end near the following snippet

   ::

      native.new_local_repository(
      name = "ngraph_external",
      path = "/your/home/directory/where/ngraph_is_installed",
      build_file = str(Label("//tensorflow/compiler/plugin/ngraph:ngraph.BUILD")),
      )

   and modify the following line in the :file:`tensorflow/workspace.bzl` file to 
   provide an absolute path to ``~/ngraph_dist``
   
   ::
     
     path = "/directory/where/ngraph_is_installed"


#. Now run :command:`configure` and follow the rest of the TF build process.



Maintaining ``libngraph``
=========================
TBD



.. _MXNet: http://mxnet.incubator.apache.org/


