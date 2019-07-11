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

nGraph Compiler stack
#####################


nGraph is an open-source graph compiler for :abbr:`Artificial Neural Networks (
ANNs)`. The nGraph Compiler stack provides an inherently efficient graph-based
compilation infrastructure designed to be compatible with many upcoming
:abbr:`Application-Specific Integrated Circuits (ASICs)`, like the Intel®
Nervana™ Neural Network Processor (Intel® Nervana™ NNP), while also unlocking
a massive performance boost on any existing hardware targets for your neural
network: both GPUs and CPUs. Using its flexible infrastructure, you will find
it becomes much easier to create Deep Learning (DL) models that can adhere to
the "write once, run anywhere" mantra that enables your AI solutions to easily
go from concept to production to scale.

Frameworks using nGraph to execute workloads have shown `up to 45X`_
performance boost compared to native implementations. 
For a high-level overview, see the :ref:`introduction` and our latest :doc:`project/release-notes`.

.. _intro:

Introduction
============

.. toctree::
   :maxdepth: 1
   :glob:

   introduction/*


.. _framework_support:

Framework Support
=================

.. toctree::
   :maxdepth: 1

   frameworks/overview.rst
   frameworks/tensorflow.rst
   frameworks/onnx.rst
   frameworks/paddlepaddle.rst


.. _ngraph_core:

nGraph Core
===========

.. toctree::
   :maxdepth: 1

   core/overview.rst
   core/pattern-matcher.rst
   core/graph_construction.rst
   core/passes/passes.rst
   nGraph Core Ops <ops/index.rst>
.. core/quantization.rst
.. core/dynamic_shape.rst
.. core/control_flow.rst


.. _backend_support:

Backend Support
===============

.. toctree::
   :maxdepth: 1

   backend-support/overview.rst
.. backend-support/cpu.rst
.. backend-support/kernel_library.rst
.. backend-support/plaidml.rst


.. _distributed:

Distributed Training
====================

.. toctree::
   :maxdepth: 1

   distributed/overview.rst
.. distributed/tensorflow.rst
.. distributed/paddlepaddle.rst


.. _validated_workloads:

Validated Workloads
===================

.. toctree::
   :maxdepth: 1

   validated_workloads/list.rst


.. toctree::
   :maxdepth: 1

..  diagnostics/nbench.rst
..  diagnostics/provenance.rst
..  diagnostics/netron.rst


.. _contribution_guide:

Contribution
============

.. toctree::
   :maxdepth: 1

   contribution/guide.rst
.. contribution/governance.rst


.. nGraph: https://www.ngraph.ai
.. _up to 45X: https://ai.intel.com/ngraph-compiler-stack-beta-release/