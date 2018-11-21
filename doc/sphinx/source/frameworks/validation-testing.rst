.. frameworks/validation-testing: 


Validation and testing
######################

* **Validating** -- To provide optimizations with nGraph, we first 
  confirm that a given workload is "validated" as being functional; 
  that is, we can successfully load its serialized graph as an nGraph 
  :term:`function graph`. Following here is a list of 14 workloads 
  we've tested with success.


.. csv-table::
   :header: "Workload", "Validated"
   :widths: 27, 53
   :escape: ~

   DenseNet-121, Functional
   Inception-v1, Functional
   Inception-v2, Functional
   ResNet-50, Functional
   Shufflenet, Functional
   SqueezeNet, Functional
   VGG-19, Functional
   ZFNet-512, Functional
   MNIST, Functional
   Emotion-FERPlus, Functional
   BVLC AlexNet, Functional
   BVLC GoogleNet, Functional
   BVLC CaffeNet, Functional
   BVLC R-CNN ILSVRC13, Functional 



* **Testing & Performance Optimizations** for workloads that have been 
  "validated" with nGraph are also available via the nGraph 
  :abbr:`Intermediate Representation (IR)`). For example, a common use 
  case for data scientists is to train a new model with a large dataset, 
  and so nGraph already has several accelerations available "out of the 
  box" for the workloads noted below.


TensorFlow 
==========

.. csv-table::
   :header: "TensorFlow Workloads", "Performance"
   :widths: 27, 53
   :escape: ~

   Resnet50 v1 and v2, 50% of P40
   Inception V3 and V4, 50% of P40
   Inception-ResNetv2, 50% of P40
   MobileNet v1, 50% of P40
   SqueezeNet v1.1, 50% of P40
   SSD-VGG16, 50% of P40
   R-FCN, 50% of P40
   Faster RCNN, 50% of P40
   Yolo v2, 50% of P40
   GNMT, Greater than or equal to :abbr:`Direct Optimization (DO)`
   Transformer-LT, 50% of P40
   Wide & Deep, 50% of P40
   WaveNet, Functional
   U-Net, Greater than DO
   DRAW, 50% of P40
   A3C, 50% of P40


MXNet
=====


.. csv-table::
   :header: "MXNet Workloads", "Performance"
   :widths: 27, 53
   :escape: ~

   Resnet50 v1 and v2, 50% of P40
   DenseNet (121 161 169 201), 50% of P40
   InceptionV3, 50% of P40
   InceptionV4, 50% of P40
   Inception-ResNetv2, 50% of P40
   MobileNet v1, 50% of P40
   SqueezeNet v1 and v1.1, 50% of P40
   VGG16, Functional (No DO available)
   Faster RCNN, 50% of P40
   SSD-VGG16, 50% of P40
   GNMT, Greater than or equal to :abbr:`Direct Optimization (DO)`
   Transformer-LT, 50% of P40
   Wide & Deep, 50% of P40
   WaveNet, Functional
   DeepSpeech2, 50% of P40
   DCGAN, 50% of P40
   A3C, Greater than or equal to DO



  














