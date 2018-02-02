.. code-contributor-README:


Core Contributor Guidelines
###########################

Code formatting
================

All C/C++ source code in the ``libngraph`` repository, including the test code 
when practical, should adhere to the project's source-code formatting guidelines.

The script ``maint/apply-code-format.sh`` enforces that formatting at the C/C++ 
syntactic level. 

The script at ``maint/check-code-format.sh`` verifies that the formatting rules 
are met by all C/C++ code (again, at the syntax level.)  The script has an exit 
code of ``0`` when this all code meets the standard; and non-zero otherwise.  
This script does *not* modify the source code.


Core Ops
--------

Our design philosophy is that the graph is not a script for running kernels, but, rather,
that the graph should describe the computation in terms of ops that are building blocks,
and compilation should match these ops to appropriate kernels for the backend(s) in use.
Thus, we expect that adding core ops should be infrequent. Instead, functionality should
be added by adding functions that build sub-graphs from existing core ops.


Coding style  
-------------

.. TODO:  add the core coding style Google Doc collab here when final


GitHub  
------

- How to submit a PR 
- Best practices
- Etc.






