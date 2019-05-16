.. ---------------------------------------------------------------------------
.. Copyright 2018-2019 Intel Corporation
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

######################
nGraph Compiler stack 
######################


nGraph is an open-source graph compiler for :abbr:`Artificial Neural Networks (ANNs)`. 
The nGraph Compiler stack provides an inherently efficient graph-based compilation 
infrastructure designed to be compatible with many upcoming 
:abbr:`Application-Specific Integrated Circuits (ASICs)`, like the Intel® Nervana™ 
Neural Network Processor (Intel® Nervana™ NNP), while also unlocking a massive 
performance boost on any existing hardware targets for your neural network: both 
GPUs and CPUs. Using its flexible infrastructure, you will find it becomes much 
easier to create Deep Learning (DL) models that can adhere to the "write once, 
run anywhere" mantra that enables your AI solutions to easily go from concept to 
production to scale.

Frameworks using nGraph to execute workloads have shown `up to 45X`_ performance 
boost compared to native implementations. For a high-level overview, see the 
:doc:`project/introduction` and our latest :doc:`project/release-notes`.

.. toctree::
   :maxdepth: 1
   :caption: Connecting Frameworks
   
   frameworks/index.rst
   frameworks/validated/list.rst
   frameworks/generic-configs.rst


.. toctree::
   :maxdepth: 1
   :caption: nGraph Core

   buildlb.rst
   core/overview.rst
   core/fusion/index.rst
   nGraph Core Ops <ops/index.rst>
   core/constructing-graphs/index.rst
   core/passes/passes.rst
   
.. toctree::
   :maxdepth: 1
   :caption: nGraph Python API

   python_api/index.rst

   
.. toctree::
   :maxdepth: 1
   :caption: Backend Support

   backend-support/index.rst
   backend-support/cpp-api.rst


.. toctree::
   :maxdepth: 1
   :caption: Distributed Training

   distr/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Inspecting Graphs

   inspection/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Project Metadata

   project/introduction.rst
   project/release-notes.rst
   project/contribution-guide.rst
   project/governance.rst
   project/doc-contributor-README.rst
   project/index.rst
   project/extras.rst 
   glossary.rst

Indices and tables
==================

   * :ref:`search`
   * :ref:`genindex`




.. nGraph: https://www.ngraph.ai
.. _up to 45X: https://ai.intel.com/ngraph-compiler-stack-beta-release/