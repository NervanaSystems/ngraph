.. howto/distribute-train.rst 


Train using multiple nGraph CPU backends with data parallel 
===========================================================

In the :doc:`previous section <../howto/derive-for-training>`, we described the 
steps needed to create a "trainable" nGraph model. Here we demonstrate how to 
train a data parallel model by distributing the graph across devices.

To use this mode of training, first install a supported version of `OpenMPI`_ 
(1.10 or newer). 

Next, create an nGraph build with the cmake flag ``-DNGRAPH_DISTRIBUTED_ENABLE=TRUE``.  

To deploy data-parallel training on multi-node/device, the ``AllReduce`` op 
should be added after the steps needed to complete the 
:doc:`backpropagation <../howto/derive-for-training>`.

.. literalinclude:: ../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 188-191

Also since we are using OpenMPI in this example, we need to initialize and 
finalize MPI.

.. literalinclude:: ../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 112

.. literalinclude:: ../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 295

Finally, to run the training on two nGraph devices, invoke :command:`mpirun`. 
This will run on a single machine and launch two processes. 


.. code-block:: console 

   $ mpirun -np 2 dist_mnist_mlp





.. _OpenMPI: https://www.open-mpi.org/software/ompi/v3.1
