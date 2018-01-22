.. testing-libngraph:


Testing the nGraph library
##########################

The |InG| library code base uses the `GTest framework`_ for unit tests. CMake 
automatically downloads a copy of the required GTest files when configuring the 
build directory.

To perform the unit tests:

#. Create and configure the build directory as described in our 
   :doc:`installation` guide.

#. Enter the build directory and run ``make check``:
   
   .. code-block:: console

      $ cd build/
      $ make check

#. To build the full Google test suite (required to compile with MXNet):

   .. code-block:: console

      $ git clone git@github.com:google/googletest.git
      $ cd googletest/ && cmake . && make -j$(nproc) && sudo make install      


Compiling a framework with ``libngraph``
========================================

After building and installing the nGraph library to your system, the next 
logical step is to compile a framework that you can use to run a 
training/inference model with one of function-driven backends that are now 
enabled. See our :doc:`model-phases` documentation for more about function-driven
backend design and architecture for algorithms.      

Intel nGraph library supports all of the popular frameworks including `MXNet`_,
`TensorFlow`_, `Caffe2`_, `PyTorch`_, `Chainer`_ and the native `neon`_ frontend
framework. Currently we provide integration guides for MXNet and Tensorflow, as
well as legacy documentation for the `neon`_ framework. Integration guides for 
each of the other frameworks is forthcoming.    


.. _GTest framework: https://github.com/google/googletest.git
.. _MXNet: http://mxnet.incubator.apache.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _Caffe2: https://github.com/caffe2/
.. _PyTorch: http://pytorch.org/
.. _Chainer: https://chainer.org/
.. _neon: http://neon.nervanasys.com/index.html/
