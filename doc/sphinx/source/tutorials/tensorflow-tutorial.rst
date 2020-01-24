.. _tensorflow_tutorial:

Get Started with nGraph for TensorFlow
######################################

Learn how to use nGraph to speed up training and inference on TensorFlow workloads. 

.. contents::

Overview
========

This tutorial is divided into two parts: 

#. building and installing nGraph for TensorFlow, and 
#. an example of how to use nGraph with TensorFlow.


Build and install nGraph
========================

Software requirements
---------------------

+--------------------------+-----------------------------------------+
| Using pre-built packages | Building from source                    |
+==========================+=========================================+
| Python   3               | Python 3                                |
+--------------------------+-----------------------------------------+
| OpenCL runtime           | OpenCL runtime                          |
+--------------------------+-----------------------------------------+
| TensorFlow   v1.14       |`Bazel <bazel_>`_ 0.25.2                 |
+--------------------------+-----------------------------------------+
|                          | GCC 4.8 (Ubuntu),   Clang/LLVM (macOS)  |
+--------------------------+-----------------------------------------+
|                          | ``cmake`` 3.4 or higher                 |
+--------------------------+-----------------------------------------+
|                          | ``virtualenv`` 16.0                     |
+--------------------------+-----------------------------------------+

`OpenCL runtime <opencl_runtime_>`_ is required only if you plan to use nGraph
with an Intel GPU backend.

Note to macOS users
~~~~~~~~~~~~~~~~~~~

The build and installation instructions are identical for Ubuntu 16.04 and
macOS. However, the Python setup may vary across different versions of macOS.
The TensorFlow build instructions recommend Homebrew but developers often use
Pyenv. Some users prefer Anaconda/Miniconda. Before building nGraph, ensure 
that you can successfully build TensorFlow on macOS with a suitable Python
environment.

Use pre-built packages
----------------------

`nGraph bridge <ngraph_bridge_>`_ enables you to use the nGraph Library with
TensorFlow.  Complete the following steps to install a pre-built nGraph bridge
for TensorFlow.

Install TensorFlow:

::

    pip install -U tensorflow==1.14.0

Install ``ngraph-tensorflow-bridge``:

::

    pip install -U ngraph-tensorflow-bridge

Build from source
-----------------

To use the latest version of nGraph Library, complete the following steps to
build nGraph bridge from source. 

.. note:: The requirements for building nGraph bridge are identical to the
   requirements for building TensorFlow from source. For more information,
   review the `TensorFlow configuration <tensorflow_configuration_>`_ details. 


Before you build
~~~~~~~~~~~~~~~~

Install the following requirements before building ``nGraph-bridge``: ``bazel``, ``cmake``, ``virtualenv``, and ``gcc 4.8``.

Install ``bazel``:

::

    wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh      
    bash bazel-0.25.2-installer-linux-x86_64.sh --user

Add and source the ``bin`` path to your ``~/.bashrc`` file to call
``bazel``:

::

    export PATH=$PATH:~/bin
    source ~/.bashrc   

Install ``cmake``, ``virtualenv``, and ``gcc 4.8``.

Build ngraph-tensorflow-bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the ``ngraph-bridge`` repo:

::

    git clone https://github.com/tensorflow/ngraph-bridge.git
    cd ngraph-bridge
    git checkout master

Run the following Python script to build TensorFlow, nGraph, and the bridge.
Use Python 3.5:

::

    python3 build_ngtf.py --use_prebuilt_tensorflow

When the build finishes, a new ``virtualenv`` directory is created in
``build_cmake/venv-tf-py3``. Build artifacts (i.e., the
``ngraph_tensorflow_bridge-<VERSION>-py2.py3-none-manylinux1_x86_64.whl``) are
created in the ``build_cmake/artifacts`` directory. 

Add the following flags to build PlaidML and Intel GPU backends (optional):

::

    --build_plaidml_backend
    --build_intelgpu_backend

For more build options:

::

    python3 build_ngtf.py --help

Install ngraph-tensorflow-bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the ``ngraph-tensorflow-bridge`` using ``pip``:

::

    (nGraph) $ pip install build_cmake/artifacts/ngraph_tensorflow_bridge-*-none-manylinux1_x86_64.whl
    

Test the installation:

::

    python3 test_ngtf.py

This command runs all C++ and Python unit tests from the ``ngraph-bridge``
source tree. It also runs various TensorFlow Python tests using nGraph.

To use the ``ngraph-tensorflow-bridge``, activate the following ``virtualenv``
to start using nGraph with TensorFlow. 

::

    source build_cmake/venv-tf-py3/bin/activate

Alternatively, you can build TensorFlow and nGraph bridge outside of a
``virtualenv``. The Python ``whl`` files are located in the
``build_cmake/artifacts/`` and ``build_cmake/artifats/tensorflow`` directories,
respectively. 

Select the help option of ``build_ngtf.py`` script to learn more about various
build options and how to build other backends. 


Verify that ``ngraph-bridge`` installed correctly (optional):

::

    python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
        import ngraph_bridge; print(ngraph_bridge.__version__)"

Running this code produces something like:

.. literalinclude:: ../frameworks/getting_started.rst
   :lines: 42-46


.. note:: The pre-built version of the ``ngraph-tensorflow-bridge`` may differ
   from the version built from source. This is due to the delay in the source
   release and publishing of the corresponding Python wheel. 

Classify an image
=================

Once you have installed nGraph bridge, you can use TensorFlow with nGraph to
speed up the training of a neural network or accelerate inference of a trained
model.

Complete the following steps to use TensorFlow with nGraph to classify an image
using a `frozen model <frozen_model_>`_. 

Download the Inception v3 trained model and labels file:

::

    wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz

Extract the frozen model and labels file from the tarball:

::

    tar xvf inception_v3_2016_08_28_frozen.pb.tar.gz       

Download the image file: 

::

    wget https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/data/grace_hopper.jpg

Download the TensorFlow script:

::

    wget https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py      

Modify the downloaded TensorFlow script to run TensorFlow with nGraph optimizations:

::

    import ngraph_bridge
    ...
    config = tf.ConfigProto()
    config_ngraph_enabled = ngraph_bridge.update_config(config)
    sess = tf.Session(config=config_ngraph_enabled) 

Run the classification:

::

    python label_image.py --graph inception_v3_2016_08_28_frozen.pb \
            --image grace_hopper.jpg --input_layer=input \
            --output_layer=InceptionV3/Predictions/Reshape_1 \
            --input_height=299 --input_width=299 \
            --labels imagenet_slim_labels.txt 

This will print the following results:

::

    military uniform 0.8343056
    mortarboard 0.021869544
    academic gown 0.010358088
    pickelhaube 0.008008157
    bulletproof vest 0.005350913

The above instructions are derived from the TensorFlow C++ and Python 
`Image Recognition Demo <image_recognition_demo_>`_. 

All of the above commands are available in the 
`nGraph TensorFlow examples <ngraph_tensorflow_examples_>`_ directory. 
To classify your own images, modify the ``infer_image.py`` file in this
directory.

Add runtime options for a CPU backend
-------------------------------------

Adding runtime options for a CPU backend applies to training and inference.

By default nGraph runs with a CPU backend. To get the best performance of the
CPU backend, add the following option:

::

    OMP_NUM_THREADS=<num_cores> KMP_AFFINITY=granularity=fine,compact,1,0
    \ 
    python label_image.py --graph inception_v3_2016_08_28_frozen.pb 
            --image grace_hopper.jpg --input_layer=input \
            --output_layer=InceptionV3/Predictions/Reshape_1 \
            --input_height=299 --input_width=299 \
            --labels imagenet_slim_labels.txt 

Where ``<num_cores>`` equals the number of cores in your processor. 

**Measure the time**

nGraph is a Just In Time (JIT) compiler, meaning that the TensorFlow
computation graph is compiled to nGraph during the first instance of the
execution. From the second time onwards, the execution speeds up
significantly. 

Add the following Python code to measure the computation time:

.. code-block:: python

    # Warmup
    sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t})
    # Run
    import time
    start = time.time()
    results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
            })      
    elapsed = time.time() - start
    print('Time elapsed: %f seconds' % elapsed)

Observe that the ouput time runs faster than TensorFlow native (without
nGraph).

Use a different backend
-----------------------

You can substitute the default CPU backend with a different backend such as
``INTELGPU`` or ``PLAIDML`` (coming soon). 

For ``INTELGPU`` support, use nGraph TensorFlow bridge version ``0.16``.


To determine what backends are available, use the following API:

::

    ngraph_bridge.list_backends()

**Intel GPU**

To add the ``INTELGPU`` backend:

::

    ngraph_bridge.set_backend('INTELGPU')


Detailed examples on how to use ``ngraph_bridge`` are located in the 
`examples <examples_>`_ directory.

Debugging
=========

During the build, there may be missing configuration steps for building
TensorFlow. If you run into build issues, first ensure that you can build
TensorFlow. 

For debugging runtime issues, see the instructions provided in the
`diagnostics <diagnostics_>`_ directory.

.. _nGraph_bridge: https://github.com/tensorflow/ngraph-bridge.git
.. _Opencl_runtime: https://software.intel.com/en-us/articles/opencl-drivers
.. _tensorflow_configuration: https://www.tensorflow.org/install/source
.. _bazel: https://github.com/bazelbuild/bazel/releases/tag/0.25.2
.. _frozen_model: https://www.tensorflow.org/guide/extend/model_files#freezing
.. _image_recognition_demo: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
.. _nGraph_tensorflow_examples: https://github.com/tensorflow/ngraph-bridge/tree/master/examples
.. _diagnostics: https://github.com/tensorflow/ngraph-bridge/tree/master/diagnostics
.. _examples: https://github.com/tensorflow/ngraph-bridge/tree/master/examples