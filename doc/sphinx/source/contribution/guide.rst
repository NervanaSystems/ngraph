.. _contribution:

Contribution Guide
##################

.. contents::

.. _license:

License
=======

All contributed code must be compatible with the `Apache 2`_ license,
preferably by being contributed under the Apache 2 license. Code
contributed with another license will need the license reviewed by
Intel before it can be accepted.

.. _formatting:

Code formatting
===============

All C/C++ source code in the repository, including the test code, must
adhere to the source-code formatting and style guidelines described
here.  The coding style described here applies to the nGraph
repository. Related repositories may make adjustments to better match
the coding styles of libraries they are using.


Adding ops to nGraph Core
-------------------------

Our design philosophy is that the graph is not a script for running
optimized kernels; rather, the graph is a specification for a
computation composed of basic building blocks which we call
``ops``. Compilation should match groups of ``ops`` to appropriate
optimal semantically equivalent groups of kernels for the backend(s)
in use. Thus, we expect that adding of new Core ops should be
infrequent and that most functionality instead gets added with new
functions that build sub-graphs from existing core ops. 


Coding style
-------------

We have a coding standard to help us to get development done. If part of the
standard is impeding progress, we either adjust that part or remove it. To this
end, we employ coding standards that facilitate understanding of *what nGraph
components are doing*. Programs are easiest to understand when they can be
understood locally; if most local changes have local impact, you do not need to
dig through multiple files to understand what something does and if it
is safe to modify.

Names
~~~~~

Names should *briefly* describe the thing being named and follow these casing
standards:

- Define C++ class or type names with ``CamelCase``.
- Assign template parameters with ``UPPER_SNAKE_CASE``.
- Case variable and function names with ``lower_snake_case``.

Method names for basic accessors are prefixed by ``get_``, ``is_``, or ``set_`` and
should have simple :math:`\mathcal{O}(1)` implementations:

- A ``get_`` method should be externally idempotent. It may perform some simple
  initialization and cache the result for later use.  Trivial ``get_``
  methods can be defined in a header file. If a method is
  non-trivial, that is often a sign that it is not a basic accessor.

- An ``is_`` may be used instead of ``get_`` for boolean accessors.

- A ``set_`` method should change the value returned by the corresponding ``get_``
  method.

  * Use ``set_is_`` if using ``is_`` to get a value.
  * Trivial ``set_`` methods may be defined in a header file.

- Names of variables should indicate the use of the variable.

  * Member variables should be prefixed with ``m_``.
  * Static member variables should be rare and be prefixed with ``s_``.

- Do not use ``using`` to define a type alias at top-level in header file.
  If the abstraction is useful, give it a class.

  * C++ does not enforce the abstraction. For example if ``X`` and ``Y`` are
    aliases for the same type, you can pass an ``X`` to something expecting a ``Y``.
  * If one of the aliases were later changed, or turned into a real type, many
    callers could require changes.


Namespaces
~~~~~~~~~~

- ``ngraph`` is for the public API, although this is not currently enforced.

  * Use a nested namespace for implementation classes.
  * Use an unnamed namespace or ``static`` for file-local names. This helps
    prevent unintended name collisions during linking and when using shared
    and dynamically-loaded libraries.
  * Never use ``using`` at top-level in a header file.

    - Doing so leaks the alias into users of the header, including headers that
      follow.
    - It is okay to use ``using`` with local scope, such as inside a class
      definiton.
  * Be careful of C++'s implicit namespace inclusions. For example, if a
    parameter's type is from another namespace, that namespace can be visible
    in the body.
  * Only use ``using std`` and/or ``using ngraph`` in ``.cpp`` files. ``using`` a
    nested namespace has can result in unexpected behavior.


File Names
~~~~~~~~~~

- Do not use the same file name in multiple directories. At least one
  IDE/debugger ignores the directory name when setting breakpoints.

- Use ``.hpp`` for headers and ``.cpp`` for implementation.

- Reflect the namespace nesting in the directory hierarchy.

- Unit test files are in the ``tests`` directory.

  * Transformer-dependent tests are tests running on the default transformer or
    specifying a transformer. For these, use the form

    .. code-block:: cpp

       TEST(file_name, test_name)

  * Transformer-independent tests:

    - File name is ``file_name.in.cpp``
    - Add ``#include "test_control.hpp"`` to the file's includes
    - Add the line ``static std::string s_manifest = "${MANIFEST}";`` to the top of the file.
    - Use

      .. code-block:: sh

         NGRAPH_TEST(${BACKEND_NAME}, test_name)

      for each test. Files are
      generated for each transformer and the ``${BACKEND_NAME}`` is replaced
      with the transformer name.

      Individual unit tests may be disabled by adding the name of the test to the
      ``unit_test.manifest`` file found in
      the transformer's source file directory.


Formatting
~~~~~~~~~~

Things that look different should look different because they are different. We
use **clang format** to enforce certain formatting. Although not always ideal,
it is automatically enforced and reduces merge conflicts.

- The :file:`.clang-format` file located in the root of the project specifies
  our format.  Simply run:  

  .. code-block:: console

     $ make style-check
     $ make style-apply

- Formatting with ``#include`` files:

  * Put headers in groups separated by a blank line. Logically order the groups
    downward from system-level to 3rd-party to ``ngraph``.
  * Formatting will keep the files in each group in alphabetic order.
  * Use this syntax for files that **do not change during nGraph development**; they
    will not be checked for changes during builds. Normally this will be
    everything but the ngraph files:

    .. code-block:: cpp

       #include <file>

  * Use this syntax for files that **are changing during nGraph development**; they will
    be checked for changes during builds. Normally this will be ngraph headers:

    .. code-block:: cpp

       #include "file"

  * Use this syntax for system C headers with C++ wrappers:

    .. code-block:: cpp

       #include <c...>

- To guard against multiple inclusion, use:

  .. code-block:: cpp

     #pragma once

  * The syntax is a compiler extension that has been adopted by all
    supported compilers.

- The initialization

  .. code-block:: cpp

     Foo x{4, 5};

  is preferred over

  .. code-block:: cpp

     Foo x(4, 5);

- Indentation should be accompanied by braces; this includes single-line bodies
  for conditionals and loops.

- Exception checking:

  * Throw an exception to report a problem.
  * Nothing that calls ``abort``, ``exit`` or ``terminate`` should be used. Remember
    that ngraph is a guest of the framework.
  * Do not use exclamation points in messages!
  * Be as specific as practical. Keep in mind that the person who sees the error
    is likely to be on the other side of the framework and the message might be
    the only information they see about the problem.

- If you use ``auto``, know what you are doing. ``auto`` uses the same
  type-stripping rules as template parameters. If something returns a reference,
  ``auto`` will strip the reference unless you use ``auto&``:

  * Don't do things like

    .. code-block:: cpp

       auto s = Shape{2,3};

    Instead, use

    .. code-block:: cpp

       Shape s{2, 3};

  * Indicate the type in the variable name.

- One variable declaration/definition per line

  - Don't use the C-style

    .. code-block:: cpp

       int x, y, *z;

    Instead, use:

    .. code-block:: cpp

       int x;
       int y;
       int* z;

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

.. _contribute:

Contributing to documentation
=============================

.. important:: Read this for changes affecting **anything** in ``ngraph/doc``

For updates to the nGraph Library ``/doc`` repo, please submit a PR with 
any changes or ideas you'd like integrated. This helps us maintain trackability 
with respect to changes made, additions, deletions, and feature requests.

If you prefer to use a containerized application, like Jupyter\* notebooks, 
Google Docs\*, the GitHub* GUI, or MS Word\* to explain, write, or share 
documentation contributions, you can convert the ``doc/sphinx/source/*.rst`` 
files to another format with a tool like ``pypandoc`` and share a link   
to your efforts on our `wiki`_. 

Another option is to fork the `ngraph repo`_, essentially snapshotting it at
that point in time, and to build a Jupyter\* notebook or other set of docs
around it for a specific use case. Add a note on our wiki to show us what you
did; new and novel applications may have their projects highlighted on an
upcoming `ngraph.ai`_ release.


.. note:: Please do not submit Jupyter* notebook code to the nGraph Library 
   or core repos; best practice is to maintain any project-specific examples, 
   tests, or walk-throughs in a separate repository.


Documenting source code examples 
--------------------------------

When **verbosely** documenting functionality of specific sections of code --
whether they are entire code blocks within a file, or code strings that are
**outside**  the nGraph Library's `documentation repo`_, here is an example of
best practice:

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
build on the previous example, let's take a bigger chunk of code, add some line
numbers, and add a caption:

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
use to prevent code bloat.  See the :doc:`contribution-guide` for our code
style guide.


.. build-docs:

How to build the documentation
-------------------------------


.. note:: Stuck on how to generate the html? Run these commands; they assume
   you start at a command line running within a clone (or a cloned fork) of the
   ``ngraph`` repo.  You do **not** need to run a virtual environment to create
   documentation if you don't want; running ``$ make clean`` in the
   ``doc/sphinx`` folder removes any generated files.

Right now the minimal version of Sphinx needed to build the documentation is
Sphinx v. 1.7.5.  This can be installed with :command:`pip3`, either to a
virtual  environment, or to your base system if you plan to contribute much core
code or documentation. For C++ API docs that contain inheritance diagrams and
collaboration diagrams which are helpful for framework integratons, building
bridge code, or  creating a backend UI for your own custom framework, be sure
you have a system  capable of running `doxygen`_.

To build documentation locally, run: 

   .. code-block:: console

      $ sudo apt-get install python3-sphinx
      $ pip3 install Sphinx==1.7.5
      $ pip3 install breathe numpy
      $ cd doc/sphinx/
      $ make html
      $ cd build/html
      $ python3 -m http.server 8000

Then point your browser at ``localhost:8000``.

To build documentation in a python3 virtualenv, try: 

   .. code-block:: console

      $ python3 -m venv py3doc
      $ . py3doc/bin/activate
      (py3doc)$ pip install Sphinx breathe numpy
      (py3doc)$ cd doc/sphinx
      (py3doc)$ make html
      (py3doc)$ cd build/html
      (py3doc)$ python -m http.server 8000

Then point your browser at ``localhost:8000``.

.. note:: For docs built in a virtual env, Sphinx latest changes may break 
   documentation; try building with a specific version of Sphinx. 


For tips on writing reStructuredText-formatted documentation, see the `sphinx`_ 
stable reST documentation.

.. _Apache 2: https://www.apache.org/licenses/LICENSE-2.0
.. _ngraph repo: https://github.com/NervanaSystems/ngraph/
.. _ngraph.ai: https://www.ngraph.ai
.. _documentation repo: https://github.com/NervanaSystems/ngraph/tree/master/doc
.. _sphinx: http://www.sphinx-doc.org/en/stable/rest.html
.. _wiki: https://github.com/NervanaSystems/ngraph/wiki/
.. _breathe: https://breathe.readthedocs.io/en/latest/
.. _doxygen: http://www.doxygen.org/index.html


.. 45555555555555555555555555555
