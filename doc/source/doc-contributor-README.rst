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

################################
Documentation Contributor README
################################

How to start contributing?
==========================

For Intel® Nervana™ Graph project components only, please submit a PR with any 
changes or ideas you'd like integrated to help us maintain trackability with 
respect to additions or feature requests.

If you prefer to use a containerized application, like Jupyter* notebooks, 
Google Docs*, or MS Word* to write and share your documentation contributions, 
you can convert the ``doc/source/.rst`` files in this folder to another file 
extension-friendly format with a tool like ``pypandoc``.  Another option is to 
fork the `ngraph repo`_, essentially snapshotting it at that point in time, 
and to build a Jupyter* notebook or other set of docs around it for a specific 
use case.  

Keep in mind though, that if you do that, we request the following: 

.. note:: Please do not submit Jupyter* notebook code to Intel Nervana Graph
   project repos; best practice is to maintain any project-specific examples, 
   tests, or walk-throughs separately. Alternatively you may wish to upstream 
   documentation contributions directly to whatever frontend framework supports 
   the rendering and reproducibility of your example.

To share your non-component contributions with us, please add a link to your project on 
our new `Showcase`_ page on the wiki.    



Documenting source code examples 
--------------------------------

When **verbosely** documenting functionality of specific sections of code -- whether 
they're entire code blocks within a file, or code strings that are **outside** the 
Intel Nervana Graph `documentation repo`_, here is an example of best practice: 

Say the file named ``dqn_atari.py`` has some interesting functionality that could
benefit from more explanation about one or more of the pieces in context. To keep 
the "in context" format, write something like the following in your documentation
source file (``.rst``):

::

  .. literalinclude:: ../../examples/dqn/dqn_atari.py
     :language: python
     :lines: 12-30

And the raw code will render as follows

.. literalinclude:: ../../examples/dqn/dqn_atari.py
   :language: python
   :lines: 12-30 

You can now verbosely explain the code block without worrying about breaking
the code!

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

  .. literalinclude:: ../../examples/dqn/dqn_atari.py
     :language: python
     :lines: 12-49
     :linenos:
     :caption: Defining action_axes within the dqn_atari.py file for neon frontend


and the generated output will show readers of your helpful documentation

.. literalinclude:: ../../examples/dqn/dqn_atari.py
   :language: python
   :lines: 12-49
   :linenos:
   :caption: Defining action_axes within the dqn_atari.py file for neon frontend

Take note that the ``linenos`` line will add a new context for line numbers
within your file; it will not bring the original line numbering with it. This
usually is not a problem because users will not see the back-end code rendering
the raw source code file, just the output defined by your reference.  

Our documentation practices are designed around "write once, reuse" that we can 
use to prevent code bloat.  A ``literalinclude`` with the ``caption`` option 
also generates a permalink (see above) that makes finding "verbose" documentation 
easier.       

.. note:: Stuck on how to generate the html?  Run these commands; they assume 
   you start at a command line running within a clone (or a cloned fork) of the 
   ``ngraph`` repo.  You do **not** need to run a virtual environment to create 
   documentation if you don't want; running ``$ make clean`` in the ``doc/`` folder
   removes any generated files.

   .. code-block:: console

      $ pip install -r doc_requirements.txt
      $ cd /doc/source/
      /ngraph/doc/source$ make html


For tips similar to this, see the `sphinx`_ stable reST documentation.   

.. _ngraph repo: https://github.com/NervanaSystems/ngraph/
.. _documentation repo: https://github.com/NervanaSystems/ngraph/tree/master/doc
.. _sphinx: http://www.sphinx-doc.org/en/stable/rest.html
.. _showcase: https://github.com/NervanaSystems/ngraph/wiki/Showcase


