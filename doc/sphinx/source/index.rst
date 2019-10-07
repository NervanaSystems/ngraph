.. Copyright 2017-2019 Intel Corporation
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
nGraph Compiler stack 
######################


.. _ngraph_home:

.. only:: release

   nGraph Compiler stack documentation for version |version|.

   Documentation for the latest (master) development branch can be found 
   at https://ngraph.nervanasys.com/docs/latest 
   .. https://docs.ngraph.ai/

.. only:: (development or daily)

   nGraph Compiler stack documentation for the master tree under development 
   (version |version|).


The nGraph Library and Compiler stack are provided under the `Apache 2.0 license`_ 
(found in the LICENSE file in the project's `repo`_). It may also import or reference 
packages, scripts, and other files that use licensing.

.. _Apache 2.0 license: https://github.com/NervanaSystems/ngraph/blob/master/LICENSE
.. _repo: https://github.com/NervanaSystems/ngraph

.. toctree::
   :name: mastertoctree
   :titlesonly: 

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction.rst
   features.rst

.. toctree::
   :maxdepth: 1
   :caption: Framework Support

   frameworks/index.rst
   frameworks/validated/list.rst


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
   :caption: APIs 

   python_api/index.rst
   backends/index.rst
   backends/cpp-api.rst


.. toctree::
   :maxdepth: 1
   :caption: Inspecting Graphs

   inspection/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Project Metadata

   project/release-notes.rst
   project/contribution-guide.rst
   project/index.rst
   project/extras/index.rst 
   glossary.rst



.. only:: html

Indices and tables
==================

   * :ref:`search`
   * :ref:`genindex`
