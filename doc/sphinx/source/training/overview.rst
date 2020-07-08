.. training/overview.rst:

.. _overview:

.. contents::

Overview
========

Distributed training for :abbr:`Deep Learning (DL)` involves defining a per-node 
application workload (such as with a distributed TensorFlow script) and leveraging 
infrastructure (container, Docker, or Kubernetes-specific) to spawn processes on 
many nodes which communicate and work together.

About Workloads 
---------------

A workload is an :abbr:`Artificial Intelligence (AI)` training job that can be 
executed via one of the :ref:`distributed-training-methods` across one of the 
possible :ref:`network-topologies` with dedicated NNP-T devices.

There are a number of assumptions or constraints imposed by the 
:abbr:`DL (Deep Learning)` workload which have to be facilitated by the 
infrastructure.  For example, Synchronous :term:`SGD` workloads typically require 
all ``N`` processes to run -- and if one job fails, they all fail. Jobs using 
*asynchronous* techniques, however, (parameter server, for example) can tolerate 
varying numbers of worker processes dynamically. Jobs using only data center 
fabric for communication can be scheduled with more leniency on physical 
placement, while NNP-T jobs using :abbr:`Inter-Chip Links (ICL)` (ICL) in a ring 
or mesh topology have to be scheduled using adjacent accelerators.

In nGraph, we enable the :abbr:`High-Performance Compute (HPC)`, or HPC techniques 
that use MPI to launch distributed training, providing excellent scaling efficiency 
with very light overhead. See the :doc:`../core/constructing-graphs/distribute-train`
documentation for detail on how to deploy data-parallel training. Currently nGraph 
launches a series of duplicated graphs on each device; communication happens 
on the device without copying data to the host. nGraph currently supports data 
parallelism on two frameworks: TensorFlow* and PaddlePaddle.


.. _distributed-training-methods:

Distributed Training Methods
----------------------------

* Single-chip training
* Multi-chip training
* Multi-chassis training


.. _network-topologies:

Network Topologies
------------------

* Single-card configuration
* Multi-card
* POD

Compatibility with Docker and Kubernetes
----------------------------------------

The Intel® Nervana™ Neural Network Processor for Training (NNP-T) includes 
**kube-nnp** software that enables cluster-level observability and management 
(orchestration).

kube-nnp extends the functionality of a default installation of Kubernetes, a 
container orchestration system, to manage the life cycle of machine learning jobs 
in a cluster of machines that contains NNP T-1000 accelerators, in addition to 
other compute devices such as CPUs, GPUs, or FPGAs. The orchestration system 
provides fair sharing, fault tolerance, bin-packing, and hardware abstraction 
that makes the use of large compute clusters easy while providing a standardized 
experience for users.  For more detail on **kube-nnp**, see the documentation 
provided with the software.

