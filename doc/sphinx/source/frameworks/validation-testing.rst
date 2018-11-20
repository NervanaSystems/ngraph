.. frameworks/validation-testing: 


Validation and testing
######################

* **Validating** -- To provide optimizations with nGraph, we first 
  confirm that a given workload is :term:`validated` as being functional; 
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


Testing
#######

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

   Resnet50 v1 and v2, see nbench
   Inception V3 and V4, see nbench
   Inception-ResNetv2, see nbench
   MobileNet v1, see nbench
   SqueezeNet v1.1, see nbench
   SSD-VGG16, see nbench
   R-FCN, see nbench
   Faster RCNN, see nbench
   Yolo v2, see nbench
   GNMT, Greater than or equal to :abbr:`Direct Optimization (DO)`
   Transformer-LT, see nbench
   Wide & Deep, see nbench
   WaveNet, Functional
   U-Net, Greater than DO
   DRAW, see nbench
   A3C, see nbench


MXNet
=====


.. csv-table::
   :header: "MXNet Workloads", "Performance"
   :widths: 27, 53
   :escape: ~

   Resnet50 v1 and v2, see nbench
   DenseNet (121 161 169 201), see nbench
   InceptionV3, see nbench
   InceptionV4, see nbench
   Inception-ResNetv2, see nbench
   MobileNet v1, see nbench
   SqueezeNet v1 and v1.1, see nbench
   VGG16, Functional (No DO available)
   Faster RCNN, see nbench
   SSD-VGG16, see nbench
   GNMT, Greater than or equal to :abbr:`Direct Optimization (DO)`
   Transformer-LT, see nbench
   Wide & Deep, see nbench
   WaveNet, Functional
   DeepSpeech2, see nbench
   DCGAN, see nbench
   A3C, Greater than or equal to DO



.. important:: See Intel's `Optimization Notice`_ for details. 



.. _Optimization Notice: https://software.intel.com/en-us/articles/optimization-notice


.. Notice revision #20110804:: Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

















