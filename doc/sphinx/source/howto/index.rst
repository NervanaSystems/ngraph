.. howto/index: 

How to 
======

.. toctree::
   :maxdepth: 1
   :caption: How to 

   execute.rst
    

The "How to" articles in this section explain how to do specific tasks with the 
Intel nGraph++ library. The recipes are all framework agnostic; in other words, 
if an entity (framework or user) wishes to make use of target-based computational 
resources, it can either:

* Do the tasks programatically through the framework, or 
* Provide a clear model definition with documentation for the computational 
  resources needed. 

.. note:: This section is aimed at intermediate-level developers working with
   the nGraph++ library. It assumes a developer has understanding of the concepts 
   in the previous sections. It does not assume knowledge of any particular 
   frontend framework. 
  
Since our primary audience is developers who are pushing the boundaries of deep 
learning systems, we go beyond the use of deep learning primitives, and include 
APIs and documentation for developers who want the ability to write programs 
that use custom backends. For example, we know that GPU resources can be useful 
backends for *some* kinds of algorithmic operations while they impose inherent 
limitations and slow down others. We are barely scraping the surface of what is 
possible for a hybridized approach to many kinds of training and inference-based 
computational tasks. 

One of our goals with the nGraph project is to enable developers with tools to 
build programs that quickly access and process data with or from a breadth of 
edge and network devices.  Furthermore, we want them to be able to make use of 
the best kind of computational resources for the kind of data they are processing,
after it has been gathered.

To get started, we've provided a basic example for how to execute a computation 
that can run on an nGraph backend; this is analogous to a framework bridge.  

This section is under development; it will eventually be populated with more 
articles geared toward data scientists, algorithm designers, framework developers, 
backend engineers, and others.  We welcome contributions from the community and
invite you to experiment with the variety of hybridization and performance 
extractions available through the nGraph library.    

