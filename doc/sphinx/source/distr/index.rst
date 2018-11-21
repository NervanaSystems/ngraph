.. distr/index: 

Distributed Training in nGraph
==============================

Why distributed training?
-------------------------

A tremendous amount of data is required to train DNNs in diverse areas -- from 
computer vision to natural language processing. Meanwhile, computation used in 
AI training has been increasing exponentially. And even though significant 
improvements have been made in algorithms and hardware, using one machine to 
train a very large :term:`NN` is usually not optimal. The use of multiple nodes, 
then, becomes important for making deep learning training feasible with large 
datasets.   

Data parallelism is the most popular parallel architecture to accelerate deep 
learning with large datasets. The first algorithm we support is `based on the 
synchronous`_ :term:`SGD` method, and partitions the dataset among workers 
where each worker executes the same neural network model. For every iteration, 
nGraph backend computes the gradients in back-propagation, aggregates the gradients 
across all workers, and then update the weights. 

How? (Generic frameworks)
-------------------------

* :doc:`../howto/distribute-train`

To synchronize gradients across all workers, the essential operation for data 
parallel training, due to its simplicity and scalability over parameter servers, 
is “allreduce”. The AllReduce op is one of the nGraph Library’s core ops. To 
enable gradient synchronization for a network, we simply inject the AllReduce op 
into the computation graph, connecting the graph for the autodiff computation 
and optimizer update (which then becomes part of the nGraph graph). The 
nGraph Backend will handle the rest. 

Data scientists with locally-scalable rack or cloud-based resources will likely 
find it worthwhile to experiment with different modes or variations of  
distributed training. Deployments using nGraph Library with supported backends 
can be configured to train with data parallelism and will soon work with model 
parallelism. Distributing workloads is increasingly important, as more data and 
bigger models mean the ability to :doc:`../howto/distribute-train` work with 
larger and larger datasets, or to work with models having many layers that 
aren't designed to fit to a single device.  

Distributed training with data parallelism splits the data and each worker 
node has the same model; during each iteration, the gradients are aggregated 
across all workers with an op that performs "allreduce", and applied to update 
the weights.

Using multiple machines helps to scale and speed up deep learning. With large 
mini-batch training, `one could train ResNet-50 with Imagenet-1k data`_ to the 
*Top 5* classifier in minutes using thousands of CPU nodes. See also: 
`arxiv.org/pdf/1709.05011.pdf`_. 






MXNet
-----

We implemented a KVStore in MXNet\* (KVStore is unique to MXNet) to modify 
the SGD update op so the nGraph graph will contain the allreduce op and generate
corresponding collective communication kernels for different backends. We are 
using OpenMPI for CPU backends and plan to integrate `Intel MLSL`_ in future. 

The figure below shows a bar chart with preliminary results from a Resnet-50 
I1K training in MXNet 1, 2, 4, (and 8 if available) nodes, x-axis is the number 
of nodes while y-axis is the throughput (images/sec).



.. TODO add figure graphics/distributed-training-ngraph-backends.png
   



TensorFlow
----------

We plan to support the same in nGraph-TensorFlow. It is still work in progress.
Meanwhile, users could still use Horovod and the current nGraph TensorFlow, 
where allreduce op is placed on CPU instead of on nGraph device.
Figure: a bar chart shows preliminary results Resnet-50 I1K training in TF 1, 
2, 4, (and 8 if available) nodes, x-axis is the number of nodes while y-axis 
is the throughput (images/sec).

Future work
-----------

Model parallelism with more communication ops support is in the works. For 
more general parallelism, such as model parallel, we plan to add more 
communication collective ops such as allgather, scatter, gather, etc. in 
the future. 


.. _based on the synchronous: https://arxiv.org/pdf/1602.06709.pdf 
.. _one could train ResNet-50 with Imagenet-1k data: https://blog.surf.nl/en/imagenet-1k-training-on-intel-xeon-phi-in-less-than-40-minutes/
.. _arxiv.org/pdf/1709.05011.pdf: https://arxiv.org/pdf/1709.05011.pdf
.. _Intel MLSL: https://github.com/intel/MLSL/releases