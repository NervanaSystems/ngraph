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


.. This documentation is available online at 
.. https://ngraph.nervanasys.com/docs/latest


Welcome
=======

nGraph is an open-source C++ library, compiler, and runtime accelerator for 
software engineering in the :abbr:`Deep Learning (DL)` ecosystem. nGraph 
simplifies the path from code to makes it possible to design, write, compile, and deploy 
:abbr:`Deep Neural Network (DNN)`-based solutions that can be easily scaled.
A more detailed explanation on the feature set of nGraph Compiler and runtime, 
as well as a high-level overview can be found on our project :doc:`project/about`. 

.. figure:: graphics/599px-Intel-ngraph-ecosystem.png
   :width: 599px


Quick Start
===========

We have various documents to help you get started.  

* **Framework users** of TensorFlow and MXNet can get started with 
  * :doc:`framework-integration-guides`.

* **Data scientists** interested in the `ONNX`_ format will find the 
  `nGraph ONNX companion tool`_ of interest and want to make use of the 
  :doc:`python_api/index`. 

* **Framework authors and architects** will likely want to :doc:`buildlb` 
  and read up on :doc:`howto/execute`. For examples of generic optimizations 
  available when designing your framework directly with nGraph, see 
  :doc:`frameworks/generic`.  

* **Optimization pass writers** will find :doc:`fusion/index` useful, as well
  as our :doc:`ops/index`. 


Currently-supported backends and future 
---------------------------------------

.. csv-table::
   :header: "Backend", "Current support", "Future nGraph support"
   :widths: 35, 10, 10

   Intel® Architecture Processors (CPUs), Yes, Yes
   Intel® Nervana™ Neural Network Processor (NNPs), Yes, Yes
   AMD\* GPUs, via PlaidML, Yes
   NVIDIA\* GPUs, via PlaidML, Some 
   Intel® Architecture GPUs, Yes, Yes 
   :abbr:`Field Programmable Gate Arrays (FPGA)` (FPGAs), Coming soon, Yes
   Intel Movidius™ Myriad™ 2 (VPU), Coming soon, Yes
   Other, Not yet, Ask

Supported frameworks
--------------------

.. csv-table::
   :header: "Framework", "Bridge Available?", "ONNX Support?"
   :widths: 27, 10, 10

   TensorFlow, Yes, Yes
   MXNet, Yes, Yes
   PaddlePaddle, Coming Soon, Yes
   PyTorch, No, Yes
   CNTK, No, Yes
   Other, Write your own, Custom


.. note:: The Library code is under active development as we're continually 
   adding support for more kinds of DL models and ops, framework compiler 
   optimizations, and backends.


=======

Contents
========

.. toctree::
   :maxdepth: 1
   :name: tocmaster
   :caption: Documentation

   buildlb.rst
   graph-basics.rst
   howto/index.rst
   ops/index.rst
   framework-integration-guides.rst
   frameworks/index.rst
   fusion/index.rst
   programmable/index.rst
   distr/index.rst
   python_api/index.rst
   project/index.rst


Indices and tables
==================

   * :ref:`search`   
   * :ref:`genindex`



.. _nGraph ONNX companion tool: https://github.com/NervanaSystems/ngraph-onnx
.. _ONNX: http://onnx.ai
.. _Movidius: https://www.movidius.com/
.. _contributions: https://github.com/NervanaSystems/ngraph#how-to-contribute
