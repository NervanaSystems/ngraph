.. fusion/index.rst: 


Optimize Graphs 
===============

with nGraph Compiler fusions
----------------------------

The nGraph Compiler is an optimizing compiler. As such, it provides a way to 
capture a given :term:`function graph` and perform a series of optimization 
passes over that graph. The result is a semantically-equivalent graph that, when 
executed using any |InG| :doc:`backend <../programmable/index>`, has optimizations 
inherent at the hardware level: superior runtime characteristics to increase 
training performance or reduce inference latency.   

There are several ways to describe what happens when we capture and translate 
the framework's output of ops into an nGraph graph. :term:`Fusion` is the term 
we shall use in our documentation; the action also can be described as: 
*combining*, *folding*, *squashing*, *collapsing*, or *merging* of graph 
functions. 

Optimization passes may include algebraic simplifications, domain-specific 
simplifications, and fusion. Most passes share the same mode of operation (or 
the same operational structure) and consist of various stages (each one a 
:term:`step`) where a developer can experiment with the intercepted or dynamic 
graph. These steps may be cycled or recycled as needed: 

#. Locate a list of potentially-transformable subgraphs in the given graph.
#. Transform the selected candidates into semantically-equivalent subgraphs 
   that execute faster, or with less memory (or both). 
#. Verify that the optimization pass performs correctly, with any or all expected 
   transformations, with the ``NGRAPH_SERIALIZE_TRACING`` option, which 
   serializes a graph in the `json` format after a pass.
#. Measure and evaluate your performance improvements with ``NGRAPH_CPU_TRACING``, 
   which produces timelines compatible with ``chrome://tracing``.

Optimizations can be experimented upon without using any backend by registering 
a pass with pass manager (``Manager``), calling ``run_passes`` on a function, and 
then inspecting the transformed graph. 

Optimization passes can be programmed ahead of time if you know or can predict 
what your graph will look like when it's ready to be executed (in other words: 
which `ops` can be automatically translated into :doc:`nGraph Core ops <../ops/index>`). 

The ``Interpreter`` is simply a backend providing reference implementations of 
ngraph ops in C++, with the focus on simplicity over performance.
 

Example
-------

Let us first consider a simple example. A user would like to execute a graph 
that describes the following arithmetic expression:

:math:`a + b * 1` or :math:`Add(a, Mul(b, 1))` 

In the above expressions, `1` is an identity element; any element multiplied by 
the identity element is equal to itself. This is the same as saying:

:math:`b * 1 = b` 

The writer of an optimization pass which uses algebraic simplification would 
probably want to first ``locate`` all multiplication expressions where 
multiplicands are multiplied by `1` (for stage 1) and to then ``transform``, 
``simplify``, or ``replace`` those expressions with just their multiplicands 
(for stage 2).  

To make the work of an optimization pass writer easier, the nGraph Library 
includes facilities that enable the *finding* of relevant candidates using 
pattern matching (via ``pattern/matcher.hpp``), and the *transforming* of the 
original graph into a condensed version (via ``pass/graph_rewrite.hpp``).

Let's consider each in more detail and many ways they can help the graph 
optimizer. 


.. toctree::
   :maxdepth: 1 

   graph-rewrite.rst
   passes-that-use-matcher.rst
   
