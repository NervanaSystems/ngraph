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

We have some core ops. Other ops may be added to core when they
have sufficient documentation and examples of those ops in practice
or potentially-practical use cases.  



Coding style  
-------------

.. TODO:  add the core coding style Google Doc collab here when final


GitHub  
------

- How to submit a PR 
- Best practices
- Etc.






