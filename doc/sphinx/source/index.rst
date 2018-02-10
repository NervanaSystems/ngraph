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

#############################
Intel nGraph library project
#############################

Welcome to the Intel nGraph project, an open source C++ library for developers
of :abbr:`Deep Learning (DL)` (DL) systems. Here you will find a suite of 
components, APIs, and documentation that can be used to compile and run  
:abbr:`Deep Neural Network (DNN)` (DNN) models defined in a variety of frameworks.  

For this early release, we provide :doc:`framework-integration-guides` to compile 
and run MXNet and TensorFlow-based projects.

The nGraph library translates a framework’s representation of computations into 
an :abbr:`Intermediate Representation (IR)` designed to promote computational 
efficiency on target hardware. Initially-supported backends include Intel 
Architecture CPUs (CPU), the Intel® Nervana Neural Network Processor™ (NNP), 
and NVIDIA\* GPUs. Currently-supported compiler optimizations include efficient 
memory management and data layout abstraction. 

Further overview details can be found on our :doc:`about` page. 

=======

.. toctree::
   :maxdepth: 1
   :caption: Table Of Contents
   :name: tocmaster

   installation.rst
   testing-libngraph.rst
   framework-integration-guides.rst
   graph-basics.rst

.. toctree::
   :maxdepth: 1
   :caption: Algorithms 
   :name: 

.. toctree::
   :maxdepth: 1
   :caption: Reference API

   api.rst
   autodiff.rst
   glossary.rst

.. toctree::
   :maxdepth: 1
   :caption: Ops

   ops/abs.rst
   ops/convolution.rst

.. toctree::
   :maxdepth: 1
   :caption: Project Docs

   about.rst
   release-notes.rst
   code-contributor-README.rst

.. toctree::
   :maxdepth: 0
   :hidden: 
   
   branding-notice.rst
   doc-contributor-README.rst


Indices and tables
==================

   * :ref:`search`   
   * :ref:`genindex`
     
