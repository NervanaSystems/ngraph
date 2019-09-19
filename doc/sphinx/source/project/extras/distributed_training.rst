.. project/extras/distributed_training.rst:

Distributed training with nGraph
================================

.. important:: Distributed training is not officially supported as of version
   |version|; however, some configuration options have worked for nGraph 
   devices in testing environments.


How? (Generic frameworks)
-------------------------

See also: :doc:`../../core/constructing-graphs/distribute-train`

To synchronize gradients across all workers, the essential operation for data
parallel training, due to its simplicity and scalability over parameter servers,
is ``allreduce``. The AllReduce op is one of the nGraph Library’s core ops. To
enable gradient synchronization for a network, we simply inject the AllReduce op
into the computation graph, connecting the graph for the autodiff computation
and optimizer update (which then becomes part of the nGraph graph). The
nGraph Backend will handle the rest.

Data scientists with locally-scalable rack or cloud-based resources will likely
find it worthwhile to experiment with different modes or variations of
distributed training. Deployments using nGraph Library with supported backends
can be configured to train with data parallelism and will soon work with model
parallelism. Distributing workloads is increasingly important, as more data and
bigger models mean the ability to :doc:`../../core/constructing-graphs/distribute-train`
work with larger and larger datasets, or to work with models having many layers
that aren't designed to fit to a single device.

Distributed training with data parallelism splits the data and each worker
node has the same model; during each iteration, the gradients are aggregated
across all workers with an op that performs "allreduce", and applied to update
the weights.

Using multiple machines helps to scale and speed up deep learning. With large 
mini-batch training, one could train ResNet-50 with Imagenet-1k data to the
*Top 5* classifier in minutes using thousands of CPU nodes. See
`arxiv.org/abs/1709.05011`_.

.. _arxiv.org/abs/1709.05011: https://arxiv.org/format/1709.05011