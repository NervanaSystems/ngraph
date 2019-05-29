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


.. _ngraph_home:

.. only:: release

  nGraph Compiler stack documentation for version |version|.

   Documentation for the latest (master) development branch can be found 
   at https://docs.ngraph.ai/

.. only:: (development or daily)

   nGraph Compiler stack documentation for the master tree under development 
   (version |version|).

For information about the changes and additions for releases, please
consult the published :doc:`project/release-notes`.

The nGraph Library and Compiler stack are provided under the `Apache 2.0 license`_ 
(as found in the LICENSE file in the project's `GitHub repo`_). The nGraph 
Library and Compiler stack may also import or reuse packages, scripts, and 
other files that use other licensing.

.. _Apache 2.0 license:
   https://github.com/NervanaSystems/ngraph/blob/master/LICENSE

.. _GitHub repo: https://github.com/NervanaSystems/ngraph


   * :doc:`project/index` -- Introduction to the nGraph Compiler stack: Overview, Architecture, Features, and Licensing
   * :doc:`getting-started/index` -- Getting Started Guides for framework users: find out how to ``pip install`` compatible bridges for various frameworks 
   * :doc:`python_api/index` -- Python API 
   * :doc:`backend-developers/index` -- Developer Guides: Documentation and APIs for backend developers building custom hardware or frameworks on nGraph Core. 



.. raw:: html

   <ul class="grid">
       <li class="grid-item">
      <a href="project/introduction.html">
          <img alt="" src="logo.png"/>
          <h2>Introduction</h2>
      </a>
      <p></p>
       </li>
       <li class="grid-item">
   

      <a href="getting_started/index.html">
               
          <h2>Getting Started Guides</h2>
      </a>
      <p>Follow this guide to build and test the nGraph Library's core ops.</p>
       </li>
       <li class="grid-item">
   
      <a href="contribute/index.html">
               
          <h2>Contribution Guidelines</h2>
      </a>
      <p>As an open-source project, we welcome and encourage the community
           to submit patches directly to the project.</p>
       </li>
       <li class="grid-item">
      <a href="samples/index.html">
               
          <h2>Samples and Demos</h2>
      </a>
      <p>A list of deep learning models tested and validated on nGraph</p>
       </li>
       <li class="grid-item">
      <a href="frameworks/validated/index.html">
               
          <h2>API Reference</h2>
      </a>
      <p>API Documentation and Backend Developer Reference</p>
       </li>
       <li class="grid-item">
      <a href="security/index.html">
          <h2>APIs</h2>
      </a>
      <p>Requirements, processes, and developer guidelines for ensuring security is addressed within the Zephyr project.</p>
       </li>
       <li class="grid-item">
      <a href="boards/index.html">
               <span class="grid-icon fa fa-object-group"></span>
          <h2>Supported Boards</h2>
      </a>
      <p>List of supported frameworks.</p>
       </li>
       <li class="grid-item">
      <a href="guides/index.html">
               <span class="grid-icon fa fa-puzzle-piece"></span>
          <h2>User and Developer Guides</h2>
      </a>
      <p>User and Developer Guides, including tools and tips</p>
       </li>
   </ul>



.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
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
   :caption: Backend Developers

   backend-developers/index.rst
   backend-developers/cpp-api.rst


.. toctree::
   :maxdepth: 1
   :caption: Inspecting Graphs

   inspection/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Project Metadata

   project/release-notes.rst
   project/introduction.rst
   project/contribution-guide.rst
   project/doc-contributor-README.rst
   project/index.rst
   project/extras.rst 
   glossary.rst

.. only:: html

Indices and tables
==================

   * :ref:`search`
   * :ref:`genindex`
