.. project/extras.rst


#######
Extras
#######


* :ref:`homomorphic_encryption`
* :ref:`distributed_training`

This section contains extra tools and tips for working with up-and-coming 
features of the nGraph Compiler stack.


.. _homomorphic_encryption:

Homomorphic Encryption (HE)
===========================

* **Encryption with Intel® HE transformer for nGraph™** 

  * The `Intel HE_transformer`_ enables deep encryption with nGraph Backends.

  * `Blog post`_ with `examples`_

    .. note:: Some implementations using TensorFlow* may also work with the  
       `nGraph Bridge repo`_ if older versions of ``ngraph-tf`` are not 
       available.



.. _distributed_training:

Distributed training with nGraph
================================

.. important:: Distributed training is not officially supported as of version
   |version|; however, some configuration options have worked for nGraph 
   devices in testing environments.


How? (Generic frameworks)
-------------------------

See also: :doc:`../core/constructing-graphs/distribute-train`

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
bigger models mean the ability to :doc:`../core/constructing-graphs/distribute-train`
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



.. _nGraph Bridge repo: https://github.com/tensorflow/ngraph-bridge
.. _Intel HE_transformer: https://github.com/NervanaSystems/he-transformer
.. _Blog post: https://www.intel.ai/he-transformer-for-ngraph-enabling-deep-learning-on-encrypted-data/
.. _examples: https://github.com/NervanaSystems/he-transformer#examples
.. _arxiv.org/abs/1709.05011: https://arxiv.org/format/1709.05011
.. _based on the synchronous: https://arxiv.org/format/1602.06709 
