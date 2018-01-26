.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel Corporation
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

.. Intel nGraph library core documentation master file, created on Mon Dec 25 13:04:12 2017.

Intel nGraph library
====================
Intel nGraph is a suite of components that serve as backends for deep learning
frameworks, allowing the framework to be used on a variety of Intel and 
non-Intel execution platforms. There is a *framework bridge* for each
supported framework which acts as an intermediary between the *ngraph core*
and the framework. A *transformer* plays a similar role between the ngraph
core and the various execution platforms.

The *nGraph core* uses a strongly typed platform-neutral statelss graph 
representation for computations. Each node, or *op*, in the graph corresponds
to one step in a computation, where each step produces zero or more tensor
outputs from zero or more tensor inputs.

Transformers compile the graph using a combination of generic and 
platform-specific graph transformation. The result is a function that
can be executed from the framework bridge. Transformers also allocate
and deallocate, as well as read and write, tensors under direction of the
bridge.

.. toctree::
   :maxdepth: 1
   :caption: Table Of Contents
   :name: tocmaster

   installation.rst
   testing-libngraph.rst
   framework-integration-guides.rst
   build-a-functiongraph.rst

.. toctree::
   :maxdepth: 1
   :caption: Models 
   :name: Models

.. toctree::
   :maxdepth: 2
   :caption: Backend Components

.. toctree::
   :maxdepth: 1
   :caption: Reference API

   api.rst
   autodiff.rst
   glossary.rst

.. toctree::
   :maxdepth: 1
   :caption: Ops

   op/convolution.rst


.. toctree::
   :maxdepth: 1
   :caption: Project Docs

   release-notes.rst
   code-contributor-README.rst

.. toctree::
   :maxdepth: 0
   :hidden: 
   
   branding-notice.rst
   doc-contributor-README.rst



Indices and tables
==================

   * :ref:`search`   
   * :ref:`genindex`