.. doc-contributor-README:

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

Documentation Contributor README
================================

How to start contributing?
--------------------------

For Intel® nGraph™ library core project components only, please submit a PR with 
any changes or ideas you'd like integrated. This helps us maintain trackability 
with respect to additions or feature requests.

If you prefer to use a containerized application, like Jupyter\* notebooks, 
Google Docs\*, or MS Word\* to write and share documentation contributions, 
you can convert the ``doc/sphinx/source/*.rst`` files to another format with a tool 
like ``pypandoc`` and share a link to your docs on our `wiki`_.    

Another option is to fork the `ngraph repo`_, essentially snapshotting it at 
that point in time, and to build a Jupyter\* notebook or other set of docs around 
it for a specific use case, and to share that contribution with us directly on
our wiki.  

.. note:: Please do not submit Jupyter* notebook code to the Intel nGraph library 
   repos; best practice is to maintain any project-specific examples, tests, or 
   walk-throughs separately. Alternatively, you may wish to upstream documentation 
   contributions directly to whatever frontend framework supports your example.



Documenting source code examples 
--------------------------------

When **verbosely** documenting functionality of specific sections of code -- whether 
they're entire code blocks within a file, or code strings that are **outside** the 
Intel nGraph `documentation repo`_, here is an example of best practice: 

Say the file named `` `` has some interesting functionality that could
benefit from more explanation about one or more of the pieces in context. To keep 
the "in context" format, write something like the following in your documentation
source file (``.rst``):

::

  .. literalinclude:: ../../../src/ngraph/descriptor/primary_tensor_view.cpp
     :language: cpp
     :lines: 20-31

And the raw code will render as follows


.. literalinclude:: ../../../src/ngraph/descriptor/primary_tensor_view.cpp
   :language: cpp
   :lines: 20-31

You can now verbosely explain the code block without worrying about breaking
the code.

The trick here is to add the file you want to reference relative to the folder
where the ``Makefile`` is that generates the documentation you're writing. See the 
**note** at the bottom of this page for more detail about how this works in Intel
Nervana Graph project documentation. 


Adding captions to code blocks 
------------------------------

One more trick to helping users understand exactly what you mean with a section
of code is to add a caption with content that describes your parsing logic. To 
build on the previous example, let's take a bigger chunk of code, add some 
line numbers, and add a caption "One way to define neon axes within the dqn_atari.py file":

::

  .. literalinclude:: ../../../src/ngraph/descriptor/primary_tensor_view.cpp
    :language: cpp
    :lines: 20-31
    :caption: 


and the generated output will show readers of your helpful documentation

.. literalinclude:: ../../../src/ngraph/descriptor/primary_tensor_view.cpp
   :language: cpp
   :lines: 20-31
   :caption: 


Take note that the ``linenos`` line will add a new context for line numbers
within your file; it will not bring the original line numbering with it. This
usually is not a problem because users will not see the back-end code rendering
the raw source code file, just the output defined by your reference.  

Our documentation practices are designed around "write once, reuse" that we can 
use to prevent code bloat.  A ``literalinclude`` with the ``caption`` option 
also generates a permalink (see above) that makes finding "verbose" documentation 
easier.       


.. build-docs:

Build the Documentation
========================


.. note:: Stuck on how to generate the html?  Run these commands; they assume 
   you start at a command line running within a clone (or a cloned fork) of the 
   ``ngraph`` repo.  You do **not** need to run a virtual environment to create 
   documentation if you don't want; running ``$ make clean`` in the ``doc/`` folder
   removes any generated files.


Right now the minimal version of Sphinx needed to build the documentation is 
Sphinx v. 1.6.5.  This can be installed with `pip3`, either to a virtual 
environment, or to your base system if you plan to contribute much to docs.
`Breathe`_ can also be installed to build C++ API documentation (currently WIP).      

To build documentation locally, run: 

   .. code-block:: console

      $ pip3 install [-I] Sphinx==1.6.5 [--user] 
      $ pip3 install [-I] breathe [--user]
      $ cd doc/sphinx/
      $ make html


For tips similar to this, see the `sphinx`_ stable reST documentation.   

.. _ngraph repo: https://github.com/NervanaSystems/ngraph-cpp/
.. _documentation repo: https://github.com/NervanaSystems/ngraph/tree/master/doc
.. _sphinx: http://www.sphinx-doc.org/en/stable/rest.html
.. _wiki: https://github.com/NervanaSystems/ngraph/wiki/
.. _Breathe: https://breathe.readthedocs.io/en/latest/


