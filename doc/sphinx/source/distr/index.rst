.. distr/index: 

Distributed Training in nGraph
==============================

Data scientists with locally-scalable or cloud-based resources at their disposal 
may find it worthwhile to experiment with several modes or variations for  
distributed training. With more data and bigger models, distributed deep 
learning accelerates training when working with large datasets or when working 
with models having many layers that usually cannot be fit to a single device.

The more data that is fed to a Deep Learning (DL) model, the “smarter” it gets; 
a natural result, then, is that the neural network model as a whole gets
“smarter”.  Likewise, one indicator of getting “smarter” is being able to make
better use of fewer resources, something that is not possible with the traditional, GPU-based approach.  

Using multiple machines helps to scale and speed up deep learning. With large 
mini-batch training, `one could train Resnet-50 with Imagenet-1k data`_ to the 
*Top 5* classifier in minutes using thousands of CPU nodes. See also: 
`arxiv.org/pdf/1709.05011.pdf`_.
  
Currently, we support data parallelism and plan to extend to model parallelism 
for nGraph. Data parallelism splits the data, and each worker node has the same
model; during each iteration, the gradients are aggregated across all workers with ``AllReduce | allreduce`` and applied to update the weights. 

MXNet
-----

We have implemented a KVStore in MXNet (KVStore is unique to MXNet) to modify 
the SGD update op to include allreduce op in it so that nGraph graph will 
contains the allreduce op and generates corresponding collective communication 
kernels for different backend. We are using OpenMPI for CPU backends and plan 
to integrate Intel MLSL in future. 

Figure: a bar chart shows preliminary results Resnet-50 I1K training in MXNet 
1, 2, 4, (and 8 if available) nodes, x-axis is the number of nodes while y-axis 
is the throughput (images/sec)

TensorFlow
----------

Plan to support the same in nGraph-TensorFlow. It is still work in progress.
Meanwhile, users could still use Horovod and the current nGraph TensorFlow, 
where allreduce op is placed on CPU instead of on nGraph device.
Figure: a bar chart shows preliminary results Resnet-50 I1K training in TF 1, 
2, 4, (and 8 if available) nodes, x-axis is the number of nodes while y-axis 
is the throughput (images/sec).

Future work
-----------

Model parallelism with more communication op support To support more general 
parallelism such as model parallel, we plan to add more communication collective 
ops such as allgather, scatter, gather, etc. in future. 



.. _one could train Resnet-50 with Imagenet-1k data: https://blog.surf.nl/en/imagenet-1k-training-on-intel-xeon-phi-in-less-than-40-minutes/
.. _arxiv.org/pdf/1709.05011.pdf: https://arxiv.org/pdf/1709.05011.pdf