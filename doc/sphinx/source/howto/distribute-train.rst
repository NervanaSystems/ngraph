.. howto/distribute-train.rst 


Train using multiple nGraph CPU backends with data parallel 
===========================================================

In the :doc:`previous section <../howto/derive-for-training>`, we described the 
steps needed to create a "trainable" nGraph model. Here we demonstrate how to 
train a data parallel model by distributing the graph across devices.

To use this mode of training, first install a supported version of `OpenMPI`_ 
(1.10 or newer). 

Next, create an nGraph build with the cmake flag ``-DNGRAPH_DISTRIBUTED_ENABLE=TRUE``.  

To deploy data-parallel training on backends supported by nGraph API, the 
``AllReduce`` op should be added after the steps needed to complete the 
:doc:`backpropagation <../howto/derive-for-training>`.

.. literalinclude:: ../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 180-196
   :emphasize-lines: 9-12

We need to initialize and finalize distributed training with ``Distributed`` object;
see the `full raw code`_. 

Finally, to run the training using two nGraph devices, invoke :command:`mpirun`. 
This will launch two nGraph CPU backends.


.. code-block:: console 

   $ mpirun -np 2 dist_mnist_mlp


.. _OpenMPI: https://www.open-mpi.org/software/ompi/v3.1
.. _full raw code: https://github.com/NervanaSystems/ngraph/blob/master/doc/examples/mnist_mlp/dist_mnist_mlp.cpp 
