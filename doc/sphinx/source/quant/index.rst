.. quant/index.rst: 


Quantization with nGraph 
########################

Both time and energy must be expended to train a :abbr:`Deep Learning (DL)` 
model with a large dataset. Eventually, anyone working with a :term:`NN` will
be faced with a decision to either expend further time and energy training for a 
higher-precision accuracy, or to test and benchmark the model as it has been 
trained "thus far". For either scenario, a basic understanding of how the nGraph 
Abstraction Layer respects operations involving quantization can be 
immensely helpful.  

.. * :ref:`about_int8`
.. * :ref:`about_fp32`
.. * :ref:`quantized_models`
.. * :ref:`quantized_weights`

* :ref:`appendix`




.. _appendix:

Appendix 
========

Further reading: 


* Lower numerical precision for deep learning inference and training: https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training

* Quantization and training of Neural Networks for efficient integer-arithmetic-only inference: https://arxiv.org/abs/1712.05877

* Quantizing deep convolutional networks for efficient inference: https://arxiv.org/abs/1806.08342

* Introduction to Low-Precision 8-bit Integer Computations: https://intel.github.io/mkl-dnn/ex_int8_simplenet.html

* Model Quantization with Calibration in MXNet: https://github.com/apache/incubator-mxnet/tree/master/example/quantization

* PaddlePaddle design doc for fixed-point quantization: https://github.com/PaddlePaddle/Paddle/blob/79d797fde97aa9272bb4b9fe29e21dbd73ee837f/doc/fluid/design/quantization/fixed_point_quantization.md

* Pieces of Eight: 8-bit Neural Machine Translation: https://arxiv.org/pdf/1804.05038.pdf
