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


nGraph Compiler Stack Documentation 
###################################


.. _ngraph_home:

.. only:: release

   nGraph Compiler stack documentation for version |version|.

.. only:: (development or daily)

   nGraph Compiler stack documentation for the master tree under development 
   (version |version|).


.. toctree::
   :name: mastertoctree
   :titlesonly: 

.. toctree::
   :maxdepth: 1

   introduction.rst

.. toctree::
   :maxdepth: 1
   :caption: Framework Support

   frameworks/overview.rst
   frameworks/tensorflow_connect.rst
   frameworks/onnx_integ.rst
   frameworks/paddle_integ.rst
   frameworks/other/index.rst

.. toctree::
   :maxdepth: 1
   :caption: nGraph Core

   core/overview.rst
   buildlb.rst
   core/constructing-graphs/index.rst
   core/passes/passes.rst
   core/fusion/index.rst
   nGraph Core Ops <ops/index.rst>
   provenance/index.rst
   core/quantization.rst
   dynamic/index.rst

   
.. toctree::
   :maxdepth: 1
   :caption: Backend Support

   Basic Concepts <backends/index.rst>
   backends/plaidml-ng-api/index.rst
   Integrating Other Backends <backends/cpp-api.rst>


.. toctree::
   :maxdepth: 1
   :caption: Training

   training/index.rst
   training/qat.rst


.. toctree::
   :maxdepth: 1
   :caption: Validated Workloads

   frameworks/validated/list.rst


.. toctree::
   :maxdepth: 1
   :caption: Debugging Graphs

   inspection/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Contributing

   project/contribution-guide.rst
   glossary.rst


.. toctree::
   :maxdepth: 1
   :hidden:

   project/release-notes.rst
   project/index.rst
   project/extras/index.rst 


.. only:: html

Indices and tables
==================

   * :ref:`search`
   * :ref:`genindex`
