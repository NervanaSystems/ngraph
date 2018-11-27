.. release-notes:

Release Notes
#############

|release|


This release focuses on accelerating deep learning inference workloads on 
Intel® Xeon® (CPU processor) and has the following key features: 

* Out-of-box installation experience for TensorFlow*, MXNet*, and ONNX.
* Validated optimizations for 17 workloads each on both TensorFlow and MXNet, 
  as well as 14 for ONNX.
* Support for Ubuntu 16.04 (TensorFlow, MXNet and ONNX).
* Support for OSX 10.13.x (buildable for TensorFlow and MXNet).

This |version| release includes optimizations built for popular workloads 
already widely deployed in production environments. These workloads cover 
the following categories:

* ``image recognition & segmentation`` 
* ``object detection`` 
* ``language translation`` 
* ``speech generation & recognition``
* ``recommender systems`` 
* ``Generative Adversarial Networks (GAN)``
* ``reinforcement learning`` 

In our tests, the optimized workloads can perform up to 45X faster than native 
frameworks, and we expect performance gains for other workloads due to our 
powerful :doc:`../fusion/index` feature.


See also our recent `API changes`_



.. _API changes: https://github.com/NervanaSystems/ngraph/blob/master/changes.md
