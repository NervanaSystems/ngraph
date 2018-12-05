.. quant/index.rst: 


Quantization with nGraph 
########################

   
Intro to quantization
=====================

:term:`Quantization` refers to the conversion of numerical data into a 
lower-precision representation. Quantization is often used in deep learning 
to reduce the time and energy needed to perform computations by reducing 
the size of data transfers and the number of steps needed to perform a 
computation. 

This improvement in speed and energy usage comes at a cost in terms of 
numerical accuracy, but deep learning models are often able to function 
well in spite of this reduced accuracy due to their abstraction levels and
the kinds of values being quantized. 


.. _define_scope:

Defining "Quantifiable"
=======================

Before getting into the specifics of quantization, it is a good idea to 
think about what, exactly, we're trying to "abbreviate" in order to speed-up 
the computations. For example, we know that many of the values widely-used in 
the :term:`International System of Units` are abbreviated representations 
of longer numbers. One such example derived from the SI is Avogadro's Number, 
which can be written as follows: 

``6.022141793 * 10^23 mol^-1``

This abbreviation is often preferred for use by humans, due to the large 
integer resultant from an exponent-based calculation such as 10 ``^`` 23, 
something that would be  fairly difficult for a human to quickly calculate 
at extremely high-precision.  With other such examples of very large or 
very small numbers, it should be clear that rounding up or down becomes 
less important to the overall value of a number the further away from the 
decimal the rounding happens. Indeed, it can sometimes be true that 
calculating a number to dozens of digits after the decimal is neither 
necessary nor optimal.  

Rounding in nGraph
==================

The nGraph Core ``op`` for quantization contains several modes of rounding:  


+-------------------------------+----------------------------------------------------------------+
| ``round_mode``                | *ROUND_NEAREST_TOWARD_INFINITY:*                               |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers, round away from zero e.g. |
|                               | 2.5 -> 3                                                       |
|                               | -3.5 -> -4                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_TOWARD_ZERO:*                                   |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers, round toward zero e.g.    |
|                               | 2.5 -> 2                                                       |
|                               | -3.5 to -3                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_UPWARD:*                                        |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers, round up e.g.             |
|                               | 2.5 to 3                                                       |
|                               | -3.5 to -3                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_DOWNWARD:*                                      |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers, round down e.g.           |
|                               | 2.5 to 2                                                       |
|                               | -3.5 to -4                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_TOWARD_EVEN:*                                   |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers round to even e.g.         |
|                               | 2.5 to 2                                                       |
|                               | -3.5 to -4                                                     |
|                               |                                                                |
|                               | *ROUND_TOWARD_INFINITY:*                                       |
|                               | round to nearest integer away from zero                        |
|                               |                                                                |
|                               | *ROUND_TOWARD_ZERO:*                                           |
|                               | round to nearest integer toward zero                           |
|                               |                                                                |
|                               | *ROUND_UP:*                                                    |
|                               | round to nearest integer toward infinity (ceiling)             |
|                               |                                                                |
|                               | *ROUND_DOWN:*                                                  |
|                               | round to nearest integer toward negative infinity (floor)      |
+--------------------------------+---------------------------------------------------------------+



..

Working with element types 
==========================

Graphs constructed with nGraph have a strong, static type system that applies 
both to element types and to shapes. For example, you can't accidentally plug 
something producing a ``float`` into something expecting an ``int``, or 
something producing a matrix into something expecting a vector. That being said, 
be careful to not confuse element types in nGraph with generic C++ element 
types. Models defined in one element type (FP32) cannot be converted to a 
different element type after being trained. Rather, a "Quantization-Aware" step 
must be implemented during training.  

Quantizing a model defined in FP32 to one defined in INT8 produces slightly 
different outputs with respect to precision, depending upon the quantization 
strategy. 

.. +++++++++++++++++++++++++++++++++++ ..


Methods of abstraction
======================

For a deeper dive into some of the strategies involved in model compression 
techniques, including strategies for frugal -> aggressive quantization 
techniques, see the `Distiller`_ documentation. 

.. WIP


.. +++++++++++++++++++++++++++++++++++ ..

Most models are defined using 32-bit floating point arithmetic. This greatly
simplifies the model definition, but at a computational cost. A 32-bit floating
number is a packaging of an 8-bit signed exponent and a 25-bit signed integer,
and a simple compression trick fits everything into 32 bits. Like manual decimal 
arithmetic, floating-point arithmetic is implemented in terms of basic integer 
arithmetic and shifting on the components. Even though hardware performs these 
operations quickly, and has been designed to make sequences of floating-point 
operations able to skip some steps, each 32-bit floating-point number requires 
four bytes of storage and data transfer. 

Quantized arithmetic uses integers to represent floating-point values. For
example, let’s say we have a variable that is always somewhere in the range of
:math:`0.0` to :math:`1.0`. We could divide that range into 256 contiguous bins 
and use ``0`` for the first bin that starts at :math:`0.0`, ``1`` for the next 
bin, and so on until ``255`` for the bin that ends as :math:`1.0`. 8-bit unsigned 
integer arithmetic is similar to bin arithmetic, with some scaling and shifting, 
so we can replace each floating-point operation with one or more small integer 
operations. Storage is only one byte instead of four.



Tutorial
========





.. _appendix:

Appendix 
========

* :ref:`appendix`


Further reading: 


1. Lower numerical precision for deep learning inference and training: https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training

2. Quantization and training of Neural Networks for efficient integer-arithmetic-only inference: https://arxiv.org/abs/1712.05877

3. Quantizing deep convolutional networks for efficient inference: https://arxiv.org/abs/1806.08342

4.https://software.intel.com/en-us/mkl-linux-developer-guide-language-specific-usage-options

5. Introduction to Low-Precision 8-bit Integer Computations: https://intel.github.io/mkl-dnn/ex_int8_simplenet.html

6. Model Quantization with Calibration in MXNet: https://github.com/apache/incubator-mxnet/tree/master/example/quantization

7. PaddlePaddle design doc for fixed-point quantization: https://github.com/PaddlePaddle/Paddle/blob/79d797fde97aa9272bb4b9fe29e21dbd73ee837f/doc/fluid/design/quantization/fixed_point_quantization.md

8. Pieces of Eight: 8-bit Neural Machine Translation: https://arxiv.org/pdf/1804.05038.pdf

9. *Theory and Design for Mechanical Measurements*. ISBN-13: 978-0-471-44593-7                               (cloth : alk. paper) 

10. *Mechanics of Materials*, Sixth Edition. p. 868 ISBN 0-534-41793-0 Library of Congress Control Number: 2003113085 


.. _Distiller: https://nervanasystems.github.io/distiller/quantization/index.html#integer-vs-fp32