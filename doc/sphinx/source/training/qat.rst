.. training/qat.rst:


Quantization-Aware Training
===========================

:abbr:`Quantization-Aware Training (QAT)` is a technique used to 
quantize models during the training process. The main idea is that 
the quantization is emulated in the forward path by inserting some 
"Quantization" and "De-Quantization" nodes (Q-DQ) several places in 
the network to emulate the inference quantization noise. The 
expectation is the backward propagation will alter the weights so 
that they will adapt to this noise, and the result loss will be much 
better than traditional Post-Training Quantization.

For the weights, it is also common to take different quantization 
functions that cut off outliers. Some examples are available in the  
`Distiller guide`_. Distiller is an open-source Python package for 
neural network compression research. Network compression can reduce 
the footprint of a neural network, increase its inference speed, and 
save energy. Additionally, a framework for pruning, regularization 
and quantization algorithms is provided. A set of tools for analyzing 
and evaluating compression performance on previously-known 
State-of-the-Art (SotA) algorithms 

When using :abbr:`QAT (Quantization-Aware Training)` techniques, the 
position in which the Q-DQ ops are placed needs to align with the 
fusions hardware does for inference.


.. _Distiller guide: https://nervanasystems.github.io/distiller/algo_quantization.html#quantization-aware-training

