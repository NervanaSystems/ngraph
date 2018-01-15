.. neon.rst:

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

- ``examples/minst/mnist_mlp.py``: Multi-layer perceptron on the MNIST digits 
  dataset.
- ``examples/cifar10/cifar10_mlp.py``: Multi-layer perceptron on the CIFAR10 
  dataset.
- ``examples/cifar10/cifar10_conv.py``: Convolutional neural networks applied 
   to the CIFAR10 dataset.
- ``examples/ptb/char_rnn.py``: Character-level RNN language model on the Penn 
   Treebank dataset.


For details on model-specific optimizations or integrations using the neon 
frontend, please see the `latest neon documentation`_ online.  


Neon Axes
---------


.. note:: The API for axis type specification is still heavily under development. 
   As such, this section represents both the current state of affairs and where 
   the API is heading.

Neon extends the implementation of ``Axes`` in Intel nGraph library by 
introducing the concept of axis ``types``. An axis ``type`` allows the frontend 
to make assumptions about tensor dimensions that are not quite as strict as those
requiring a specific axis-ordering.  This feature removes some of the 
specification burden from the user. 

Different layers require different sets of axis types from their inputs. For 
instance, a ``Convolution`` layer can operate only on inputs that contain

- Exactly 1 ``channel axis``, 
- From 1 to 3 ``spatial axes``, and 
- Exactly 1 ``batch axis``. 

Similarly, an unrolled RNN can operate only on inputs that contain exactly 1 
``recurrent axis``. 

Each axis type has a default name, or set of names, that can be used. These
names are listed below.  The default values of these names *can be overridden* 
during a layer's ``__call__`` method, making it easy to use axis names that best 
fit the type of data being processed by the network.

Axis Types
----------

- ``recurrent_axes``: The default name for all recurrent axes is ``REC``. 
  Recurrent layers can operate over a single recurrent axis, though this 
  restriction may be lifted in the future.
- ``spatial_axes``: Spatial axes currently support up to three dimensions.
    - ``height``: The default name for the height axis is "H".
    - ``width``: The default name for the width axis is "W".
    - ``depth``: The default name for the depth axis is "D".
- ``channel_axes``: The default name for the channel axis is "C". Convolutional 
  layers can currently only operate over a single channel axis, though this 
  restriction may be lifted in the future.
- ``batch_axes``: The default name for the batch axis is "N". Currently the 
  batch axis cannot be overridden.





.. _latest neon documentation: http://neon.nervanasys.com/index.html/  
