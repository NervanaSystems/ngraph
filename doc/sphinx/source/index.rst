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


########
nGraph™ 
########

Welcome
=======

nGraph is an open-source C++ library and compiler suite for frameworks 
and developers in the :abbr:`Deep Learning (DL)` ecosystem. nGraph is 
framework-neutral and can be targeted to program and deploy solutions 
to the most modern compute and edge devices available today. For a more
detailed explanation on the many **features** and compatible companion 
tools available, read the project :doc:`project/about`. 

The value we're offering to the developer community is empowerment: we 
are confident that Intel® Architecture already provides the best 
computational resources available for the breadth of ML/DL tasks. 


Quick Start
===========

Depending on your level of programming familiarity, we have various 
documents to help you get started.  

* **Framework users** of TensorFlow and MXNet can get started with 
  * :doc:`framework-integration-guides` (Requires: Python3 and Command-line only)

* **Data scientists** interested in the `ONNX`_ format will find the 
  `nGraph ONNX compantion tool`_ of interest. (Requires Python3 and command-line)

* **Framework authors and architects** will likely want to :doc:`buildlb` 
  and read up on :doc:`howto/execute` for an example of how to integrate 
  your framework with nGraph via bridge code. (Requires knowledge of C++)

* **Optimization pass writers** will find :doc:`fusion/index` useful. 
  (Requires )

* **:doc:`howto/index` to learn how to write bridge code.


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


.. figure:: ../graphics/599px-Intel-ngraph-ecosystem.png
   :width: 599px


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
.. _Movidius: https://www.movidius.com/

     .. _contributions: https://github.com/NervanaSystems/ngraph#how-to-contribute
