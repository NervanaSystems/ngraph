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

#############################
Intel nGraph library project
#############################

Welcome to the Intel nGraph project, an open source C++ library for developers
of :abbr:`Deep Learning (DL)` (DL) systems and frameworks. Here you will find 
a suite of components, documentation, and APIs that can be used with 
:abbr:`Deep Neural Network (DNN)` models defined in a variety of frameworks.  

The nGraph library translates a framework’s representation of computations into 
a neutral-:abbr:`Intermediate Representation (IR)` designed to promote 
computational efficiency on target hardware; it works on Intel and non-Intel 
platforms.

For further overview details, see the :doc:`about` page.

=======

.. toctree::
   :maxdepth: 1
   :caption: Table Of Contents
   :name: tocmaster

   installation.rst
   testing-libngraph.rst
   framework-integration-guides.rst
   graph-basics.rst

.. toctree::
   :maxdepth: 1
   :caption: Algorithms 
   :name: 

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

   ops/convolution.rst

.. toctree::
   :maxdepth: 1
   :caption: Project Docs

   about.rst
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