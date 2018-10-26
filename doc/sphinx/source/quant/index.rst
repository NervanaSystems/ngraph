.. quant/index.rst: 


Quantization with nGraph 
########################

Quantization is one form of low-precision computing, a technique used to reduce 
the time and energy needed to perform a computation by reducing the size of the 
data transfers and the number of steps needed to perform the computation. 

Most models are defined using 32-bit floating point arithmetic. This greatly
simplifies the model definition, but at a computational cost. A 32-bit floating
number is a packaging of an 8-bit signed exponent and a 25-bit signed integer.
A simple compression trick lets the 33 bits fit into 32 bits. Like manual
decimal arithmetic, floating point arithmetic is implemented in terms of basic
integer arithmetic and shifting on the components. Even though hardware performs
these operations quickly, and has been designed to make sequences of
floating-point operations able to skip some steps, each 32-bit floating-point
number requires four bytes of storage and data transfer.</p>
<p>Quantized arithmetic uses integers to represent floating-point values. For
example, let’s say we have a variable that is always somewhere in the range of
0.0 to 1.0. We could divide that range into 256 contiguous bins and use 0 for
the first bin that starts at 0.0, 1 for the next bin, and 255 for the bin that
ends as 1.0. 8-bit unsigned integer arithmetic is similar to bin arithmetic,
with some scaling and shifting, so we can replace each floating-point operation
with one or more small integer operations. Storage is only one byte instead of
four.


Working with element types 
==========================

Graphs constructed with nGraph have a strong, static type system that applies 
both to element types and to shapes. For example, you can't accidentally plug 
something producing a ``float`` into something expecting an ``int``, or 
something producing a matrix into something expecting a vector.  

What this means is that models defined in one element type (FP32) cannot be 
converted to a different element type after being trained. Rather, a 
"Quantization-Aware" step must be implemented during training. This step can 
take place outside of nGraph, or with the bridge (using code from 
``/src/ngraph/builder``); or, to take another approach, a graph that has been 
modified for quantization can be trained with different quantized weights to 
produce the desired or compatible type of output. Quantizing a model defined in 
FP32 to one defined in INT8 produces slightly different outputs with respect to 
precision, depending upon the quantization strategy. 

.. +++++++++++++++++++++++++++++++++++ ..


Methods of abstraction
======================

For a deeper dive into some of the strategies involved in model compression 
techniques, see the `Distiller`_ documentation. 

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





.. _Distiller: https://nervanasystems.github.io/distiller/quantization/index.html#integer-vs-fp32