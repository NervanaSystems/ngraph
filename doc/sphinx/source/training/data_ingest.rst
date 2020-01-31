.. training/data_ingest.rst:

Data Ingestion
##############


Using TensorFlow
----------------

.. include:: tf_dist.rst


Using PaddlePaddle
------------------

.. include:: paddle_dist.rst


Using a custom framework
------------------------

.. include:: ../core/constructing-graphs/distribute-train.rst

To synchronize gradients across all workers, the essential operation for data
parallel training, due to its simplicity and scalability over parameter servers,
is ``allreduce``. The AllReduce op is one of the nGraph Library’s core ops. To
enable gradient synchronization for a network, we simply inject the AllReduce op
into the computation graph, connecting the graph for the autodiff computation
and optimizer update (which then becomes part of the nGraph graph). The
nGraph Backend will handle the rest.

