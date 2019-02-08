.. howto/distribute-train.rst 


Train using multiple nGraph CPU backends with data parallel 
===========================================================

In the :doc:`previous section <../howto/derive-for-training>`, we described the 
steps needed to create a "trainable" nGraph model. Here we demonstrate how to 
train a data parallel model by distributing the graph to more than one device.

As of release version 0.12, the default build is with OpenMPI.  To use the 
`Intel MLSL`_ library, set the following compilation flag at build time: 

``-DNGRAPH_DISTRIBUTED_ENABLE=TRUE``.

To deploy data-parallel training on backends supported by nGraph API, the 
``AllReduce`` op should be added after the steps needed to complete the 
:doc:`backpropagation <../howto/derive-for-training>`.

.. literalinclude:: ../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 180-196
   :emphasize-lines: 8-11

We need to initialize and finalize distributed training with ``Distributed`` object;
see the `full raw code`_. 

Finally, to run the training using two nGraph devices, invoke :command:`mpirun` which 
is distributed with `Intel MLSL`_ library.  This will launch two nGraph CPU backends.


.. code-block:: console 

   $ mpirun -np 2 dist_mnist_mlp


.. _Intel MLSL: https://github.com/intel/MLSL/releases
.. _full raw code: https://github.com/NervanaSystems/ngraph/blob/master/doc/examples/mnist_mlp/dist_mnist_mlp.cpp 
