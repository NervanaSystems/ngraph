.. howto/distribute-train.rst 


Distribute training across multiple nGraph backends 
===================================================

.. important:: Distributed training is not officially supported in version |version|;
    however, the following configuration options have worked for nGraph devices 
    with mixed or limited success in testing.

In the :doc:`previous section <../constructing-graphs/derive-for-training>`, 
we described the steps needed to create a "trainable" nGraph model. Here we 
demonstrate how to train a data parallel model by distributing the graph to 
more than one device.

Frameworks can implement distributed training with nGraph versions prior to 
`0.13`:

* Use ``-DNGRAPH_DISTRIBUTED_ENABLE=OMPI`` to enable distributed training 
  with OpenMPI. Use of this flag requires that OpenMPI be a pre-existing library 
  in the system. If it's not present on the system, install `OpenMPI`_ version 
  ``2.1.1`` or later before running the compile. 

* Use ``-DNGRAPH_DISTRIBUTED_ENABLE=MLSL`` to enable the option for 
  :abbr:`Intel® Machine Learning Scaling Library (MLSL)` for Linux* OS:

  .. note:: The Intel® MLSL option applies to Intel® Architecture CPUs 
     (``CPU``) and ``Interpreter`` backends only. For all other backends, 
     ``OpenMPI`` is presently the only supported option. We recommend the 
     use of `Intel MLSL` for CPU backends to avoid an extra download step.

Finally, to run the training using two nGraph devices, invoke 

.. code-block:: console 

   mpirun 

To deploy data-parallel training, the ``AllReduce`` op should be added after the 
steps needed to complete the :doc:`backpropagation <../constructing-graphs/derive-for-training>`; 
the new code is highlighted below: 

.. literalinclude:: ../../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 174-189
   :emphasize-lines: 8-11

See the `full code`_ in the ``examples`` folder ``/doc/examples/mnist_mlp/dist_mnist_mlp.cpp``. 

.. code-block:: console 

   mpirun -np 2 dist_mnist_mlp


.. _Intel MLSL: https://github.com/intel/MLSL/releases
.. _OpenMPI: https://www.open-mpi.org/software/ompi/v2.1/  
.. _full code: https://github.com/NervanaSystems/ngraph/blob/master/doc/examples/mnist_mlp/dist_mnist_mlp.cpp 
