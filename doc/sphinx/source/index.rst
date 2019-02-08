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


.. toctree::
   :maxdepth: 1 

   project/introduction.rst


.. toctree::
   :maxdepth: 1
   :caption: Framework Support
   
   frameworks/index.rst
   frameworks/validated/list.rst
   frameworks/generic-configs.rst


.. toctree::
   :maxdepth: 1
   :caption: nGraph Core

   core/overview.rst
   Pattern matcher <fusion/index.rst>
   nGraph ops <ops/about.rst>
   Graph construction <howto/index.rst>
   Using the Python API <python_api/index.rst>
   Compiler passes  <fusion/graph-rewrite.rst>
   buildlb.rst
   Using the C++ API <ops/index.rst>
   
.. toctree::
   :maxdepth: 1
   :caption: Backend support

   backend-support/index.rst
   backend-support/cpp-api.rst


.. toctree::
   :maxdepth: 1
   :caption: Distributed training

   distr/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Diagnostics and visualization

   diagnostics/nbench.rst
   diagnostics/performance-profile.rst
   diagnostics/visualize.rst
   diagnostics/debug.rst 


.. toctree::
   :maxdepth: 1
   :caption: Project Metadata

   project/release-notes.rst
   project/contribution-guide.rst
   project/index.rst 
   glossary.rst



Indices and tables
==================

   * :ref:`search`
   * :ref:`genindex`



.. _nGraph ONNX companion tool: https://github.com/NervanaSystems/ngraph-onnx
.. _ONNX: http://onnx.ai
.. _Movidius: https://www.movidius.com/
.. _contributions: https://github.com/NervanaSystems/ngraph#how-to-contribute
.. _TensorFlow bridge to nGraph: https://github.com/NervanaSystems/ngraph-tf/blob/master/README.md
.. _Compiling MXNet with nGraph: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/README.md
.. _ecosystem: https://github.com/NervanaSystems/ngraph/blob/master/ecosystem-overview.md
