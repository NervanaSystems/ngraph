.. distribute-train.rst 


Distribute training 
===================

In the :doc:`previous section <../howto/derive-for-training>`, we described the 
steps needed to create a "trainable" nGraph model. With that "trainable" nGraph model and a new set of data with which to train, it's also possible to create a
program that will prescriptively distribute a computation graph across devices
with nGraph.

To use this mode of training, first install a supported version of `OpenMPI`_. 

Next, create an nGraph build that uses the cmake flag ``DNGRAPH_DISTRIBUTED_ENABLE=TRUE``.  

To deploy data-parallel training on multi-node/device, the ``AllReduce`` op should be added after the steps needed to complete :doc:`backpropagation <../howto/derive-for-training>`.

.. literalinclude:: ../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 192

And finally, to run the training on two ngraph devices, invoke `mpirun -np 2`.


.. code-block:: console 

   $ mpirun -np 2


.. _OpenMPI: https://www.open-mpi.org/software/ompi/v3.1
