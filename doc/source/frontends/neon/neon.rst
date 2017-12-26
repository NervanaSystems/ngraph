.. _neon:

.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel Corporation
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

neon™
*****

To start using the neon™ frontend with the Intel® nGraph™ library, simply run:

.. code-block:: console

   $ pip3 install neon

This command installs a Python3-based frontend over nGraph that provides common 
deep learning primitives, including activation functions, optimizers, layers, 
and more. 

- Layers: ``Linear``, ``Bias``, ``Conv2D``, ``Pool2D``, ``BatchNorm``, 
  ``Dropout``, ``Recurrent``
- Activations: ``Rectlin``, ``Rectlinclip``, ``Identity``, ``Explin``, 
  ``Normalizer``, ``Softmax``, ``Tanh``, ``Logistic``
- Initializers: ``GaussianInit``, ``UniformInit``, ``ConstantInit``
- Optimizers: ``GradientDescentMomentum``, ``RMSprop``
- Callbacks: ``TrainCostCallback``, ``RunTimerCallback``, ``ProgressCallback``, 
  ``TrainLoggerCallback``, ``LossCallback``

Additionally, installing neon™ equips your system with several example scripts 
that can be used to construct or customize models on a clean and simple frontend:

- ``examples/minst/mnist_mlp.py``: Multi-layer perceptron on the MNIST digits dataset.
- ``examples/cifar10/cifar10_mlp.py``: Multi-layer perceptron on the CIFAR10 dataset.
- ``examples/cifar10/cifar10_conv.py``: Convolutional neural networks applied to the CIFAR10 dataset.
- ``examples/ptb/char_rnn.py``: Character-level RNN language model on the Penn Treebank dataset.


For details on model-specific optimizations or integrations using the neon 
frontend, please see the `latest neon documentation`_ online.  

.. _latest neon documentation: http://neon.nervanasys.com/index.html/  
