.. frameworks/validated/list.rst: 

#################################
Validated workloads by framework
#################################

We validated performance [#f1]_ for the following TensorFlow\* and MXNet\* 
workloads:

* :ref:`tensorflow_valid`
* :ref:`mxnet_valid`
* :ref:`onnx_valid`
* :doc:`testing-latency`




.. _tensorflow_valid:

TensorFlow 
==========

.. csv-table::
   :header: "TensorFlow Workload", "Genre of Deep Learning"
   :widths: 27, 53
   :escape: ~

   Resnet50 v1, Image recognition
   Resnet50 v2, Image recognition
   Inception V3, Image recognition
   Inception V4, Image recognition
   Inception-ResNetv2, Image recognition
   MobileNet v1, Image recognition
   MobileNet v2, Image recognition
   VGG16, Image recognition
   SSD-VGG16, Object detection
   SSD-MobileNetv1, Object detection
   R-FCN, Object detection
   Faster RCNN, Object detection
   Yolo v2, Object detection
   Transformer-LT, Language translation
   Wide & Deep, Recommender system
   NCF, Recommender system
   U-Net, Image segmentation
   DCGAN, Generative adversarial network
   DRAW, Image generation
   A3C, Reinforcement learning


.. _mxnet_valid:

MXNet
=====

.. csv-table::
   :header: "MXNet Workload", "Genre of Deep Learning"
   :widths: 27, 53
   :escape: ~

   Resnet50 v1, Image recognition
   Resnet50 v2, Image recognition
   DenseNet-121, Image recognition
   InceptionV3, Image recognition
   InceptionV4, Image recognition
   Inception-ResNetv2, Image recognition
   MobileNet v1, Image recognition
   SqueezeNet v1 and v1.1, Image recognition
   VGG16, Image recognition
   Faster RCNN, Object detection
   SSD-VGG16, Object detection
   GNMT, Language translation
   Transformer-LT, Language translation
   Wide & Deep, Recommender system
   WaveNet, Speech generation
   DeepSpeech2, Speech recognition
   DCGAN, Generative adversarial network
   A3C, Reinforcement learning


.. _onnx_valid:

ONNX
====

Additionally, we validated the following workloads are functional through 
`nGraph ONNX importer`_: 


.. csv-table::
   :header: "ONNX Workload", "Genre of Deep Learning"
   :widths: 27, 53
   :escape: ~

   ResNet-50, Image recognition
   DenseNet-121, Image recognition
   Inception-v1, Image recognition
   Inception-v2, Image recognition
   Shufflenet, Image recognition
   SqueezeNet, Image recognition
   VGG-19, Image recognition
   ZFNet-512, Image recognition
   MNIST, Image recognition
   Emotion-FERPlus, Image recognition
   BVLC AlexNet, Image recognition
   BVLC GoogleNet, Image recognition
   BVLC CaffeNet, Image recognition
   BVLC R-CNN ILSVRC13, Object detection 


.. important:: Please see Intel's `Optimization Notice`_ for details on disclaimers. 

.. rubric:: Footnotes

.. [#f1] Benchmarking performance of DL systems is a young discipline; it is a
   good idea to be vigilant for results based on atypical distortions in the 
   configuration parameters. Every topology is different, and performance 
   changes can be attributed to multiple causes. Also watch out for the word 
   "theoretical" in comparisons; actual performance should not be compared to 
   theoretical performance.




.. _Optimization Notice: https://software.intel.com/en-us/articles/optimization-notice
.. _nGraph ONNX importer: https://github.com/NervanaSystems/ngraph-onnx/blob/master/README.md

.. Notice revision #20110804: Intel's compilers may or may not optimize to the same degree for 
   non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. 
   These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. 
   Intel does not guarantee the availability, functionality, or effectiveness of any optimization 
   on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this 
   product are intended for use with Intel microprocessors. Certain optimizations not specific 
   to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the 
   applicable product User and Reference Guides for more information regarding the specific 
   instruction sets covered by this notice.

