.. distr/index: 

Distributed Training 
====================
Documentation here shows how to distribute a computation graph across
devices with nGraph.

Requirements:
- Install OpenMPI 1.10 or newer openmpi_download_.
.. _openmpi_download:https://www.open-mpi.org/software/ompi/v3.1/

- Build nGraph with cmake flag `-DNGRAPH_DISTRIBUTED_ENABLE=TRUE`

We showed how to ``Derive a trainable model``. If one would like to do data
parallel training on multi-node/device, ``Allreduce`` op should be added
after the ``backprop``.

.. literalinclude:: ../../../examples/mnist_mlp/dist_mnist_mlp.cpp
   :language: cpp
   :lines: 192

Then to run on two ngraph devices, we start the executable with `mpirun -np 2`.
