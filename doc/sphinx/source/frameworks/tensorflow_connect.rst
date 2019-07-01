.. frameworks/tensorflow_connect.rst:

Getting Started with TensorFlow\*
=================================

No matter what your level of experience with :abbr:`Deep Learning (DL)` systems 
may be, nGraph provides a path to start working with the DL stack. Let's begin 
with the easiest and most straightforward options.

Using ``pip`` install
----------------------

The easiest way to get started is to use the latest PyPI `ngraph-tensorflow-bridge`_,
which has instructions for Linux* systems, and tips for users of Mac OS X. 

You can install TensorFlow and nGraph to a virtual environment; otherwise, the code 
will install to a system location.

.. code-block:: console
   
   pip install tensorflow
   pip install ngraph-tensorflow-bridge

That's it! Now you can test the installation by running the following command:

.. code-block:: console

   python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);import ngraph_bridge; print(ngraph_bridge.__version__)"


Output will look something like:

:: 

    TensorFlow version:  [version]
    nGraph bridge version: b'[version]'
    nGraph version used for this build: b'[version-rc-hash]'
    TensorFlow version used for this build: v[version-hash]
    CXX11_ABI flag used for this build: 1

More detail in the `ngraph_bridge examples`_ directory. 


Build your own nGraph-TensorFlow bridge 
---------------------------------------

A slightly more involved option is to build the latest nGraph with a binary 
TensorFlow, or to build the bridge from source. You can try these options 
if you know you need features newer than those available in the prebuilt binary.

See the `README`_ on the `ngraph_bridge repo`_ for the latest versioning 
options with instructions on how to build the latest bridge or make a 
custom `DSO`_.


.. _ngraph-tensorflow-bridge: https://pypi.org/project/ngraph-tensorflow-bridge
.. _README: https://github.com/tensorflow/ngraph-bridge/blob/master/README.md
.. _ngraph_bridge repo: https://github.com/tensorflow/ngraph-bridge  
.. _DSO: http://csweb.cs.wfu.edu/%7Etorgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
.. _ngraph_bridge examples: https://github.com/tensorflow/ngraph-bridge/blob/master/examples/README.md
