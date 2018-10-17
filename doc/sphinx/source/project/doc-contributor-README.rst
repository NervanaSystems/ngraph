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

nGraph Library docs  
===================

Read this for changes affecting anything in ``ngraph/doc``
----------------------------------------------------------

For updates to the Intel® nGraph Library ``/doc`` repo, please submit a PR with 
any changes or ideas you'd like integrated. This helps us maintain trackability 
with respect to additions or feature requests.

If you prefer to use a containerized application, like Jupyter\* notebooks, 
Google Docs\*, or MS Word\* to explain, write, or share documentation contributions, 
you can convert the ``doc/sphinx/source/*.rst`` files to another format with a tool 
like ``pypandoc`` and share a link to your docs on our `wiki`_.

Another option is to fork the `ngraph repo`_, essentially snapshotting it at 
that point in time, and to build a Jupyter\* notebook or other set of docs around 
it for a specific use case; then share a link with the community on our wiki.   

.. note:: Please do not submit Jupyter* notebook code to the Intel nGraph library 
   or core repos; best practice is to maintain any project-specific examples, 
   tests, or walk-throughs separately. 


Documenting source code examples 
--------------------------------

When **verbosely** documenting functionality of specific sections of code -- whether 
they are entire code blocks within a file, or code strings that are **outside** 
the Intel nGraph `documentation repo`_, here is an example of best practice: 

Say a file has some interesting functionality that could benefit from more 
explanation about one or more of the pieces in context. To keep the "in context" 
navigable, write something like the following in your ``.rst`` documentation 
source file:

::

  .. literalinclude:: ../../../examples/abc/abc.cpp
     :language: cpp
     :lines: 20-31

And the raw code will render as follows


.. literalinclude:: ../../../examples/abc/abc.cpp 
   :language: cpp
   :lines: 20-31

You can now verbosely explain the code block without worrying about breaking
the code. The trick here is to add the file you want to reference relative to 
the folder where the ``Makefile`` is that generates the documentation you're 
writing. 


See the **note** at the bottom of this page for more detail about how 
this works in the current |version| version of Intel nGraph library 
documentation. 


Adding captions to code blocks 
------------------------------

One more trick to helping users understand exactly what you mean with a section
of code is to add a caption with content that describes your parsing logic. To 
build on the previous example, let's take a bigger chunk of code, add some 
line numbers, and add a caption:

::

  .. literalinclude:: ../../../examples/abc/abc.cpp
     :language: cpp
     :lines: 48-56
     :caption: "caption for a block of code that initializes tensors"


and the generated output will show readers of your helpful documentation

.. literalinclude:: ../../../examples/abc/abc.cpp
   :language: cpp
   :lines: 48-56
   :caption: "caption for a block of code that initializes tensors"

Our documentation practices are designed around "write once, reuse" that we can 
use to prevent code bloat.  See the :doc:`code-contributor-README` for our code 
style guide.       


.. build-docs:

Build the documentation
=======================


.. note:: Stuck on how to generate the html? Run these commands; they assume 
   you start at a command line running within a clone (or a cloned fork) of the 
   ``ngraph`` repo.  You do **not** need to run a virtual environment to create 
   documentation if you don't want; running ``$ make clean`` in the 
   ``doc/sphinx`` folder removes any generated files.

Right now the minimal version of Sphinx needed to build the documentation is 
Sphinx v. 1.7.5.  This can be installed with :command:`pip3`, either to a virtual 
environment, or to your base system if you plan to contribute much core code or
documentation. For C++ API docs that contain inheritance diagrams and collaboration
diagrams which are helpful for framework integratons, building bridge code, or 
creating a backend UI for your own custom framework, be sure you have a system 
capable of running `doxygen`_.   

To build documentation locally, run: 

   .. code-block:: console

      $ sudo apt-get install python3-sphinx
      $ pip3 install [-I] Sphinx==1.7.5 [--user] 
      $ pip3 install [-I] breathe numpy [--user]
      $ cd doc/sphinx/
      $ make html
      $ cd build/html
      $ python3 -m http.server 8000

Then point your browser at ``localhost:8000``.

To build documentation in a python3 virtualenv, run: 

   .. code-block:: console

      $ python3 -m venv py3doc
      $ . py3doc/bin/activate
      (py3doc)$ pip install python3-sphinx breathe numpy
      (py3doc)$ cd doc/sphinx
      (py3doc)$ make html
      (py3doc)$ cd build/html
      (py3doc)$ python -m http.server 8000

Then point your browser at ``localhost:8000``.

.. note:: For docs built in a virtual env, Sphinx latest changes may break 
   documentation; try building with a specific version of Sphinx.  



For tips on writing reStructuredText-formatted documentation, see the `sphinx`_ 
stable reST documentation.

.. _ngraph repo: https://github.com/NervanaSystems/ngraph/
.. _documentation repo: https://github.com/NervanaSystems/ngraph/tree/master/doc
.. _sphinx: http://www.sphinx-doc.org/en/stable/rest.html
.. _wiki: https://github.com/NervanaSystems/ngraph/wiki/
.. _breathe: https://breathe.readthedocs.io/en/latest/
.. _doxygen: http://www.doxygen.org/index.html
