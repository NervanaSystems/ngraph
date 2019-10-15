.. _quantization:

Quantization
============

nGraph Quantization 
-------------------

Quantization refers the process of reducing the number of bits that represent a
number. In the context of deep learning, weights and activations can be
represented using 8-bit integers (or INT8) to compress the model size of a
trained neural network, without any significant loss in model accuracy. Compared
with 32-bit floating point (FP32), using arithmetic with lower precision such as
INT8 to calculate weights and activation requires less memory. However, memory
reduction varies based on the model (ResNet 50, Inception v3, BERT, etc.).

Implementing a quantized model with nGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a quantized model with
nGraph, provide a partially (or fully) quantized FP32 model (e.g., where the
convolution layer in the model is replaced with a quantized convolution) to the
nGraph library along with quantized parameters: weights, activations, scale, and
zero point. 

.. Note:: Currently, nGraph only supports quantization for inference.

nGraph Quantized Operators (Ops)
--------------------------------

nGraph uses scale and zero point (also used by ONNX) to map real values to
quantized values.  All quantized ops (APIs) in nGraph use scale and zero point
and they can be used just like any other nGraph op. 

**Scale**: the quantization scale for the tensor 

**Zero point**: the zero point of the tensor 

**Round mode**: used in combination with scale and zero point to round real 
values to quantized values

.. table:: nGraph Quantized Ops


	+----------------------+-----------------------------------------------+
	| Operator             | Description                                   |
	+======================+===============================================+
	| Quantize             | Maps real values (r) to quantized values (q)  |
	|                      | using scale (s), zero point (z),              |
	|                      | and round mode; produces a quantized tensor   |
	+----------------------+-----------------------------------------------+
	| Dequantize           | Maps quantized values (q) to real values (r)  |
	|                      | using scale (s) and zero point (z); converts  |
	|                      | a quantized tensor to a floating point tensor |
	+----------------------+-----------------------------------------------+
	| FakeQuantize         | Performs element-wise linear quantization     |
	+----------------------+-----------------------------------------------+
	| QuantizedConvolution | Performs quantized convolution; takes the     |
	|                      | takes the same parameters as the              |
	|                      | non-quantized convolution operator            |
	+----------------------+-----------------------------------------------+
	| QuantizedDot         | Performs quantized dot; takes the same        |
	|                      | parameters as the non-quantized dot operator  |
	+----------------------+-----------------------------------------------+

Some frameworks such as TensorFlow have fused (or layer) ops. nGraph provides
optional operations (listed in the following table) to help users easily
translate (map) any quantized model created from frameworks with fused ops to
nGraph.

.. table:: Experimental Quantized Ops (optional)


	+-----------------------------------+----------------------------------+
	| Operator                          | Description                      |
	+===================================+==================================+
	| QuantizedConvolutionBias          | Performs quantized convolution   |
	|                                   | with bias                        |
	+-----------------------------------+----------------------------------+
	| QuantizedConvolutionBiasAdd       | Performs quantized convolution   |
	|                                   | with bias and add                |
	+-----------------------------------+----------------------------------+
	| QuantizedConvolutionBiasSignedAdd | Performs quantized convolution   |
	|                                   | with bias and signed add         |
	+-----------------------------------+----------------------------------+
	| QuantizedConvolutionRelu          | Performs quantized convolution   |
	|                                   | and ReLu                         |
	+-----------------------------------+----------------------------------+
	| QuantizedDotBias                  | Performs quantized dot with bias |
	+-----------------------------------+----------------------------------+

nGraph Quantization Design
~~~~~~~~~~~~~~~~~~~~~~~~~~
The goal of nGraph quantization is to flexibly support a wide variety of
frameworks and users. The use of scale and zero point as well as quantized
builders in the nGraph design helps to achieve this goal.

Scale and Zero Point
********************
Using scale and zero point allows nGraph to be framework agnostic (that is, it
can equally support all deep learning frameworks). nGraph bridges will
automatically convert min and max (provided by a DL framework) to scale and zero
point as needed. Quantized builders are available to help the bridges perform
this calculation. However, if users are directly using nGraph (i.e., not using a
bridge), they are required to provide scale and zero point for quantized ops.

Another advantage of using scale and zero point to express quantization
parameters is that users can flexibly implement quantized ops into various
nGraph backends. When implementing quantized ops, nGraph backends can directly
use scale and zero point instead of min and max.

Quantized Builders
******************
Quantized builders are helper utilities to assist framework integrators to
enable quantized models with nGraph. They serve as an API (interface) between
framework bridges and nGraph, allowing framework bridges to directly construct
ops in nGraph core IR.

Quantized builders help nGraph framework bridges by:

* Breaking down a fused quantized operator in the framework to a subgraph (of
  quantized and non-quantized operators) in the nGraph core IR

* Converting from min and max to scale and zero point based on the quantization
  mode described by the DL framework

.. Note:: Fused ops and quantized builders serve the same purpose, in the future 
 fused ops will replace quantized builders.

 .. table:: nGraph Quantized Builders

	+-------------------------------------+-----------------------------------+----------------------------------------+
	| Category                            | Builder                           | Description                            |
	+=====================================+===================================+========================================+
	| Scaled Mode                         | ScaledQuantize                    | Converts min and max to scale          |
	| Min / Max Builders                  |                                   | and zero point using a scaled mode     |
	|                                     |                                   | calculation and then constructs and    |
	|                                     |                                   | returns an nGraph Quantize operator.   |
	|                                     +-----------------------------------+----------------------------------------+
	|                                     | ScaledDequantize                  | Converts min and max to scale          |
	|                                     |                                   | and zero point using a scaled mode     |
	|                                     |                                   | calculation and then constructs and    |
	|                                     |                                   | returns an nGraph Dequantize operator. |
	+-------------------------------------+-----------------------------------+----------------------------------------+
	| Quantized Convolution               | ScaledQuantizedConvolution        | Constructs a quantized convolution     |
	| and Variants                        |                                   | with an optional ReLu.                 |
	|                                     +-----------------------------------+----------------------------------------+
	|                                     | ScaledQuantizedConvolutionBias    | Constructs a quantized convolution     |
	|                                     |                                   | with bias and an optional ReLu.        |
	|                                     +-----------------------------------+----------------------------------------+
	|                                     | ScaledQuantizedConvolutionBiasAdd | Constructs a quantized convolution     |
	|                                     |                                   | with bias and an optional ReLu, where  |
	|                                     |                                   | the output is added to the output      |
	|                                     |                                   | of another convolution (sum_input)     |
	+-------------------------------------+-----------------------------------+----------------------------------------+
	| Quantized Dot (Matmul)              | ScaledQuantizedDot                | Constructs a quantized dot (Matmul)    |
	| and Variants                        |                                   | with an optional ReLu.                 |
	|                                     +-----------------------------------+----------------------------------------+
	|                                     | ScaledQuantizedDotBias            | Constructs a quantized dot (Matmul)    |
	|                                     |                                   | with bias and an optional ReLu.        |
	+-------------------------------------+-----------------------------------+----------------------------------------+
	| Quantized Concat                    | ScaledQuantizedConcat             | Constructs a quantized concayconcat.   |
	+-------------------------------------+-----------------------------------+----------------------------------------+
