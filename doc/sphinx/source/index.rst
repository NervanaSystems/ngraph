.. ---------------------------------------------------------------------------
.. Copyright 2018 Intel Corporation
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

######################
Intel nGraph++ library
######################

Welcome to Intel® nGraph™, an open source C++ library and compiler. This 
project enables modern compute platforms to run and train :abbr:`Deep Neural Network (DNN)`
models. It is framework-neutral and supports a variety of backends used by 
:abbr:`Deep Learning (DL)` frameworks. 

.. figure:: graphics/ngraph-hub.png  

For this early release, we've provided :doc:`framework-integration-guides` to 
compile and run MXNet\* and TensorFlow\*-based projects.

.. note:: The library code is under active development as we're continually 
   adding support for more ops, more frameworks, and more backends. 

The nGraph++ library translates a framework’s representation of computations 
into an :abbr:`Intermediate Representation (IR)` that promotes computational 
efficiency on target hardware. Initially-supported backends include Intel 
Architecture CPUs (``CPU``), the Intel® Nervana Neural Network Processor™ (NNP), 
and NVIDIA\* GPUs. Currently-supported compiler optimizations include efficient 
memory management and data layout abstraction. 

Further project details can be found on our :doc:`project/about` page. 



=======

Sections
=========

.. toctree::
   :maxdepth: 1
   :name: tocmaster
   :caption: Table of Contents

   install.rst
   framework-integration-guides.rst
   graph-basics.rst
   howto/index.rst
   ops/index.rst
   project/index.rst


Indices and tables
==================

   * :ref:`search`   
   * :ref:`genindex`

     
