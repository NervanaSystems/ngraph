.. quant/index.rst: 


Quantization with nGraph 
########################

.. intro paragraph to be added later


Working with element types 
==========================

Graphs constructed with nGraph have a strong, static type system that applies 
both to element types and to shapes. For example, you can't accidentally plug 
something producing a ``float`` into something expecting an ``int``, or 
something producing a matrix into something expecting a vector.  

What this means is that models trained on one element type (FP32, for example, 
is the most common model defintion) cannot be simply quantized to a different 
element type after being trained. Rather, a "Quantization-Aware" training step 
must be implemented during construction of. This quantization-aware training 
step can take place outside of nGraph, or with the bridge; or, 
to take another approach, a graph that has been modified for quantization can be 
re-trained with different quantized weights to produce the desired or compatible 
type of output. Quantizing from FP32 into INT8 will produce results that are 
slightly different with respect to precision; this tradeoff is the gist of 
quantization.


.. +++++++++++++++++++++++++++++++++++ ..


..:ref:`about_int8`


.. :term:`qint8`


.. :ref:`about_fp32`






Methods of abstraction
======================

.. WIP


.. * :ref:`quantized_models`
.. * :ref:`quantized_weights`


.. +++++++++++++++++++++++++++++++++++ ..

Tutorial
========





.. _appendix:

Appendix 
========

* :ref:`appendix`


Further reading: 


* Lower numerical precision for deep learning inference and training: https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training

* Quantization and training of Neural Networks for efficient integer-arithmetic-only inference: https://arxiv.org/abs/1712.05877

* Quantizing deep convolutional networks for efficient inference: https://arxiv.org/abs/1806.08342

* https://software.intel.com/en-us/mkl-linux-developer-guide-language-specific-usage-options

* Introduction to Low-Precision 8-bit Integer Computations: https://intel.github.io/mkl-dnn/ex_int8_simplenet.html

* Model Quantization with Calibration in MXNet: https://github.com/apache/incubator-mxnet/tree/master/example/quantization

* PaddlePaddle design doc for fixed-point quantization: https://github.com/PaddlePaddle/Paddle/blob/79d797fde97aa9272bb4b9fe29e21dbd73ee837f/doc/fluid/design/quantization/fixed_point_quantization.md

* Pieces of Eight: 8-bit Neural Machine Translation: https://arxiv.org/pdf/1804.05038.pdf
