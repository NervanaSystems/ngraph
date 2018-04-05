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
.. http://ngraph.nervanasys.com/docs/latest/


########
nGraph™ 
########

Welcome to the documentation site for nGraph™, an open-source C++ library and 
command-line suite for the :abbr:`Deep Learning (DL)` ecosystem. Our goal with 
this project is to empower algorithm designers, data scientists, framework 
architects, software engineers, and others with the means to make their work 
:ref:`portable`, :ref:`adaptable`, and :ref:`deployable` across the most modern 
:abbr:`Machine Learning (ML)` hardware available today: optimized Deep Learning
compute devices.

.. figure:: graphics/ngraph-ecosystem.png
   :width: 650px   
  

.. _portable:

Portable
========

One of nGraph's key features is **framework neutrality**. While we currently 
support :doc:`three popular <framework-integration-guides>` frameworks with 
pre-optimized deployment runtimes for training :abbr:`Deep Neural Network (DNN)`, 
models, you are not limited to these when choosing among frontends. Architects 
of any framework (even those not listed above) can use our How-to 
:doc:`compile and run <howto/execute>` a training model documentation to learn 
how to design or tweak a framework to bridge directly to the nGraph compiler. 
Note that when the framework is enabled for this direct optimization 
:term:`bridge`, the framework itself provides the developer-facing API and 
nGraph's optimizing compiler work happens automatically.    


.. _adaptable: 

Adaptable
=========

We've recently added initial support for the `ONNX`_ format. Developers who 
already have a "trained" :abbr: `DNN (Deep Neural Network)` model can use 
nGraph to bypass a lot of the framework-based complexity and :doc:`howto/import` 
to test or run it on targeted and efficient backends with our user-friendly 
Python-based API, ``ngraph_api``.  See the `ngraph onnx companion tool`_ to 
get started with it. 


.. csv-table::
   :header: "Framework", "Bridge Code Available?", "ONNX Support?"
   :widths: 27, 10, 10

   TensorFlow, Yes, Yes
   MXNet, Yes, Yes
   neon, none needed, Yes
   PyTorch, Not yet, Yes
   CNTK, Not yet, Yes
   Other, Not yet, Doable


.. _deployable:

Deployable
==========

It's no secret that the :abbr:`DL (Deep Learning)` ecosystem is evolving 
rapidly. Benchmarking comparisons can be blown steeply out of proportion by 
subtle tweaks to batch or latency numbers here and there. Where traditional 
GPU-based training excels, inference can lag and vice versa. Sometimes it's not 
so much about "speed at training a large dataset" as it is about latency in 
getting a little bit of data back to the edge network, where it can be analyzed
by an already-trained model. 

Indeed, when choosing among "deployable" options, important to not lose sight of 
the ultimate deployability and machine-runtime demands. Sometimes you don't need 
a backhoe to plant a flower bulb. If, for example, you are developing a new 
*genre* for ML/DL modeling, it may be especially beneficial to map out or  
forecast the compute demands you need, and see where investment makes sense.  


Supported platforms
--------------------

Initially-supported backends include:

* Intel® Architecture Processors (CPUs), 
* Intel® Nervana™ Neural Network Processor™ (NNPs), and 
* NVIDIA\* CUDA (GPUs). 

Tentatively in the pipeline, we plan to add support for more backends,
including:

* :abbr:`Field Programmable Gate Arrays (FPGA)` (FPGAs)
* Movidius 

.. note:: The library code is under active development as we're continually 
   adding support for more kinds of DL models and ops, framework compiler 
   optimizations, and backends. 


Further project details can be found on our :doc:`project/about` page, or see 
our :doc:`install` guide for how to get started.   



=======

Contents
========

.. toctree::
   :maxdepth: 1
   :name: tocmaster
   :caption: Documentation

   install.rst
   graph-basics.rst
   howto/index.rst
   ops/index.rst
   framework-integration-guides.rst
   project/index.rst


Indices and tables
==================

   * :ref:`search`   
   * :ref:`genindex`

     
.. _ONNX:  http://onnx.ai
.. _ngraph onnx companion tool: https://github.com/NervanaSystems/ngraph-onnx