.. howto/index: 

How to 
======

.. note:: This section is aimed at intermediate users of Intel nGraph library. 
   It assumes a developer has understanding of the concepts in the previous 
   sections. It does not assume knowledge of any particular frontend framework. 

The "How to" articles in this section explain how to do specific tasks with 
Intel nGraph. The recipes are all framework agnostic; in other words, any 
frontend framework that wishes to access the optimizations inherent in nGraph 
will either need to do these things programatically through the framework, or 
to provide documentation for the user who has already has decided they want to 
performance optimizations available through nGraph's management of custom 
backends. 

To get started, we've provided a basic example for how to execute a computation 
that can runs on an nGraph backend; this is analogous to a framework bridge.  

This section is under development; it will eventually contain articles targeted
toward data scientists, algorithm designers, framework developers, and backend
engineers -- anyone who wants to pivot on our examples and experiment with the 
variety of hybridization and performance extractions available through the 
nGraph library.    

.. toctree::
   :maxdepth: 1
   :caption: How-to 

   execute.rst
    