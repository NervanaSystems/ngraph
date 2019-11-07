.. core/quantization.rst:


.. _quantization:

Quantization
============

:term:`Quantization` refers the process of reducing the number of bits that 
represent a number. In a :abbr:`DL (Deep Learning)` context, weights and 
activations can be represented using 8-bit integers (INT8) to compress the 
model size of a trained neural network without any significant loss in model 
accuracy. INT8 is one kind of quantization. Compared with 32-bit floating point 
(FP32), using arithmetic with lower precision, such as INT8, to calculate 
weights and activation requires less memory.


Implementing a quantized model with nGraph
------------------------------------------

To implement a quantized model with nGraph, provide a partially (or fully)
quantized model (where the convolution layer in the model is replaced
with a quantized convolution, for example) to the nGraph Library along with 
quantized parameters: weights, activations, scale, and zero point. 

.. note:: As of version |version|, only quantization for inference is supported.

nGraph Quantized Operators (Ops)
--------------------------------

nGraph uses scale and zero point (also used by ONNX) to map real values to
quantized values. All quantized ops use scale and zero point
and can be used just like any other nGraph op. 

**Scale**: the quantization scale of the tensor 

**Zero point**: the zero point of the tensor 

**Round mode**: used in combination with scale and zero point to round real 
values to quantized values

.. table:: Quantization Ops


	+-----------------------------------------------------------------+------------------------------------------------+
	| Op                                                              | Description                                    |
	+=================================================================+================================================+
	| :doc:`Quantize <../ops/quantize>`                               | Maps real values (r) to quantized values (q)   |
	|                                                                 | using scale (s), zero point (z),               |
	|                                                                 | and round mode; produces a quantized tensor.   |
	+-----------------------------------------------------------------+------------------------------------------------+
	| :doc:`Dequantize <../ops/dequantize>`                           | Maps quantized values (q) to real values (r)   |
	|                                                                 | using scale (s) and zero point (z); converts   |
	|                                                                 | a quantized tensor to a floating-point tensor. |
	+-----------------------------------------------------------------+------------------------------------------------+
	| :mod:`FakeQuantize <ngraph.ops.fake_quantize>`                  | Performs element-wise linear quantization.     |
	+-----------------------------------------------------------------+------------------------------------------------+
	| :mod:`QuantizedConvolution <ngraph.ops.quantized_convolution>`  | Performs 8-bit convolution.                    |
	+-----------------------------------------------------------------+------------------------------------------------+
	| :mod:`QuantizedDot <ngraph.ops.quantized_dot>`                  | Performs 8-bit dot.                            |
	+-----------------------------------------------------------------+------------------------------------------------+

Some frameworks such as TensorFlow\* have fused ops. nGraph provides optional
operations to help users easily translate (map) any quantized model created from
frameworks with fused ops to nGraph. Unlike builders, experimental ops take
scale and zero point instead of min and max.

.. table:: Experimental Quantized Ops (optional)


	+-----------------------------------+-------------------------------------+
	| Operator                          | Description                         |
	+===================================+=====================================+
	| QuantizedConvolutionBias          | This experimental op can be         |
	|                                   | fused with a ReLU op.               |
	+-----------------------------------+-------------------------------------+
	| QuantizedConvolutionBiasAdd       | This experimental op constructs a   |
	|                                   | quantized convolution with bias and |
	|                                   | optional ReLU. And then takes input |
	|                                   | for the add operation.              |
	+-----------------------------------+-------------------------------------+
	| QuantizedConvolutionBiasSignedAdd | Same as QuantizedConvolutionBiasAdd |
	|                                   | but with signed add.                |
	+-----------------------------------+-------------------------------------+
	| QuantizedConvolutionRelu          | This experimental op is designed    |
	|                                   | for a particular use case that      |
	|                                   | would require convolution           |
	|                                   | and ReLU to be combined.            |
	+-----------------------------------+-------------------------------------+
	| QuantizedDotBias                  | This experimental op can be fused   |
	|                                   | with a ReLU op.                     |
	+-----------------------------------+-------------------------------------+

nGraph Quantization Design
--------------------------

The goal of nGraph quantization is to flexibly support a wide variety of
frameworks and users. The use of scale and zero point as well as quantized
builders in the nGraph design helps to achieve this goal.

Scale and Zero Point
~~~~~~~~~~~~~~~~~~~~

Using scale and zero point allows nGraph to be framework agnostic (i.e., it
can equally support all deep learning frameworks). nGraph Bridges will
automatically convert min and max (provided by a DL framework) to scale and zero
point as needed. Quantized builders are available to help the bridges perform
this calculation. However, if users are directly using nGraph (and not using a
bridge), they are required to provide scale and zero point for quantized ops.

Another advantage of using scale and zero point to express quantization
parameters is that users can flexibly implement quantized ops into various
nGraph backends. When implementing quantized ops, all current nGraph backends 
will directly use scale and zero point (and not min and max) to perform 
the quantized computation.

Quantized Builders
~~~~~~~~~~~~~~~~~~

Quantized builders are helper utilities to assist framework integrators to
enable quantized models with nGraph. They serve as an API (interface) between
framework bridges and nGraph, allowing framework bridges to directly construct
ops in the nGraph Abstraction Layer.

Quantized builders help nGraph framework bridges by:

* Breaking down a fused quantized operator in the framework to a subgraph (of
  quantized and non-quantized operators) in the nGraph core IR

* Converting from min and max to scale and zero point based on the quantization
  mode described by the DL framework

.. note::  Fused ops and quantized builders serve the same purpose. 
   In the future, fused ops will replace quantized builders.

.. table:: nGraph Quantized Builders

	+--------------------------+-----------------------------------+-----------------------------------------+
	| Category                 | Builder                           | Description                             |
	+==========================+===================================+=========================================+
	| Scaled Mode              | ScaledQuantize                    | Converts min and max to scale           |
	| Min / Max Builders       |                                   | and zero point using a scaled mode      |
	|                          |                                   | calculation and then constructs and     |
	|                          |                                   | returns an nGraph Quantize operator.    |
	|                          +-----------------------------------+-----------------------------------------+
	|                          | ScaledDequantize                  | Converts min and max to scale           |
	|                          |                                   | and zero point using a scaled mode      |
	|                          |                                   | calculation and then constructs and     |
	|                          |                                   | returns an nGraph Dequantize operator.  |
	+--------------------------+-----------------------------------+-----------------------------------------+
	| Quantized Convolution    | ScaledQuantizedConvolution        | Constructs a quantized convolution      |
	| and Variants             |                                   | with an optional ReLU.                  |
	|                          +-----------------------------------+-----------------------------------------+
	|                          | ScaledQuantizedConvolutionBias    | Constructs a quantized convolution      |
	|                          |                                   | with bias and an optional ReLU.         |
	|                          +-----------------------------------+-----------------------------------------+
	|                          | ScaledQuantizedConvolutionBiasAdd | Constructs a quantized convolution      |
	|                          |                                   | with bias and an optional ReLU, where   |
	|                          |                                   | the output is added to the output       |
	|                          |                                   | of another convolution (sum_input).     |
	+--------------------------+-----------------------------------+-----------------------------------------+
	| Quantized Dot (Matmul)   | ScaledQuantizedDot                | Constructs a quantized dot (Matmul)     |
	| and Variants             |                                   | with an optional ReLU.                  |
	|                          +-----------------------------------+-----------------------------------------+
	|                          | ScaledQuantizedDotBias            | Constructs a quantized dot (Matmul)     |
	|                          |                                   | with bias and an optional ReLU.         |
	+--------------------------+-----------------------------------+-----------------------------------------+
	| Quantized Concat         | ScaledQuantizedConcat             | Constructs a quantized concatenation.   |
	+--------------------------+-----------------------------------+-----------------------------------------+
