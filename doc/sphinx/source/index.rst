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
.. https://ngraph.ai/documentation

nGraph Compiler stack
#####################

.. _ngraph_home:

.. only:: release

  nGraph Compiler stack documentation for version |version|.


   Documentation for the latest (master) development branch can be found 
   at https://ngraph.nervanasys.com/docs/latest 
   .. https://docs.ngraph.ai/

.. only:: (development or daily)

   nGraph Compiler stack documentation for the master tree under development 
   (version |version|).

For information about the releases, see the :doc:`../project/release-notes`. 

The nGraph Library and Compiler stack are provided under the `Apache 2.0 license`_ 
(found in the LICENSE file in the project's `repo`_). It may also import or reference 
packages, scripts, and other files that use licensing.

.. _Apache 2.0 license: https://github.com/NervanaSystems/ngraph/blob/master/LICENSE
.. _repo: https://github.com/NervanaSystems/ngraph


Getting Started
===============

.. toctree::
   :maxdepth: 1

   introduction.rst
   tutorials/index.rst


Framework Support
=================

.. toctree::
   :maxdepth: 1

   frameworks/overview.rst
   frameworks/tensorflow.rst
   frameworks/onnx.rst
   frameworks/paddlepaddle.rst
   frameworks/new-frameworks.rst


nGraph Core
===========

.. toctree::
   :maxdepth: 1

   core/overview.rst
   buildlb.rst
   core/fusion/index.rst
   core/constructing-graphs/index.rst
   core/passes/passes.rst
   nGraph Core Ops <ops/index.rst>



Backend Support
===============

.. toctree::
   :maxdepth: 1

   backends/overview.rst
   backends/plaidml-ng-api/index.rst


Distributed Training
====================

.. toctree::
   :maxdepth: 1

   distributed/overview.rst


Validated Workloads
===================

.. toctree::
   :maxdepth: 1

   frameworks/validated/list.rst


Visualization Tools
===================

.. toctree::
   :maxdepth: 1

   inspection/index.rst


Contribution
============

.. toctree::
   :maxdepth: 1

   contributing/guide.rst