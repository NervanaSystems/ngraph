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

###############
nGraph library
###############


Welcome to nGraph™, an open-source C++ compiler library for running and 
training :abbr:`Deep Neural Network (DNN)` models. This project is 
framework-neutral and can target a variety of modern devices or platforms. 

.. figure:: graphics/ngraph-ecosystem.png
   :width: 585px   
  
nGraph currently supports :doc:`three popular <framework-integration-guides>` 
frameworks for :abbr:`Deep Learning (DL)` models through what we call 
a :term:`bridge` that can be integrated during the framework's build time. 
For developers working with other frameworks (even those not listed above), 
we've created a :doc:`How to Guide <howto/index>` guide so you can learn how to create 
custom bridge code that can be used to :doc:`compile and run <howto/execute>` 
a training model.

We've recently added initial support for the ONNX format. Developers who 
already have a "trained" model can use nGraph to bypass a lot of the 
framework-based complexity and :doc:`howto/import` to test or run it 
on targeted and efficient backends with our user-friendly ``ngraph_api``. 
With nGraph, data scientists can focus on data science rather than worrying 
about how to adapt models to train and run efficiently on different devices.


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

     
