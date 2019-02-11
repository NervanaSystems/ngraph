.. howto/distribute-train.rst 


Distribute training across multiple nGraph backends 
===================================================

In the :doc:`previous section <../howto/derive-for-training>`, we described the 
steps needed to create a "trainable" nGraph model. Here we demonstrate how to 
train a data parallel model by distributing the graph to more than one device.

These options are currently supported for available backends; choose the best one 
for your scenario.  

* Use ``-NGRAPH_DISTRIBUTED_OMPI_ENABLE=TRUE`` to enable distributed training with 
  OpenMPI. Use of this flag requires that OpenMPI be a pre-existing library 
  in the system. If it's not present on the system, install Open MPI 2.1.1 before 
  running the compile. 

* Use ``-NGRAPH_DISTRIBUTED_MLSL_ENABLE=TRUE`` to enable the option for 
  :abbr:`Intel® Machine Learning Scaling Library (MLSL)` for Linux* OS:

  .. important:: The Intel® MLSL option applies to ``CPU`` and ``Interpreter`` 
     backends only. For all other backends, ``OpenMPI`` is presently the 
     only supported option. We recommend the use of `Intel MLSL` if there 
     are CPU only backends.

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
