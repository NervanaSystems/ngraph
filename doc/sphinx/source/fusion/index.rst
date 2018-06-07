.. fusion/index.rst: 


Optimize graphs 
===============

with nGraph Compiler fusions
-----------------------------

The nGraph |trade| Compiler is an *optimizing* compiler. As such, it performs a 
series of optimization passes over a given function graph to translate it into a 
semantically-equivalent but inherently-optimized graph with superior runtime characteristics for any of nGraph's current or future backends.

Indeed, the ability to increase training performance or reduce inference latency 
by simply adding another device of *any* form factor -- compute, ASIC, or VPU 
-- is one of the key benefits available to any framework that integrates nGraph
Library.  

In handling a :term:`function graph`, there are many ways to describe what 
happens when we translate the framework's output of ops into an nGraph 
graph. *Fusion* is the term we use in our documentation, but the the action also 
can be described as: *combining*, *folding*, *collapsing*, or *merging* graph 
functions. The most common use case is to *fuse* a subgraph from the function 
graph into :doc:`one of the nGraph Core ops <../ops/index>`. In other words, 
the nGraph compiler can find and take a sub-set (subgraph) of computations 
from the function graph and make it more efficient via :term:`fusion`.

The optimization passes may include algebraic simplifications, domain-specific 
simplifications, and fusion. Most passes share the same mode of operation (or 
the same operational structure) and consist of two stages:

#. Locating a list of potential transformation candidates (usually, subgraphs) 
   in the given graph.
#. Transforming the selected candidates into semantically-equivalent subgraphs 
   that run faster and/or with less memory..

Optimization passes can be programmed ahead of time if you know what your graph 
will look like when it's ready to be executed, or the optimization passes can 
be figured out manually with *Interpreter* mode on a stateless graph. 

Let's consider an example. A user would like to execute a simple graph that 
describes the following arithmetic expression:

:math:`a + b * 1` or :math:`Add(a, Mul(b, 1))` 

In the above expressions, `1` is an identity element: any element 
multiplied by the identity element is equal to itself. This is the same as saying

:math:`b * 1 = b` 

The writer of an optimization pass that uses algebraic-simplification would 
probably want to first ``locate`` all multiplication expressions where 
multiplicands are multiplied by `1` (for stage 1) and the ``transform``, 
``simplify``, or ``replace`` those expressions with just their multiplicands 
(for stage 2).  

To make the work of an optimization pass writer easier, the nGraph library 
includes facilities that enable the *finding* of relevant candidates using pattern 
matching (via ``pattern/matcher.hpp``), and the *transforming* of the original 
graph into a condensed version (via ``pass/graph_rewrite.hpp``).

Let's consider each of the two in more detail and many ways they can help the 
work of the optimization pass writer.



.. toctree::
   :maxdepth: 1 

   pattern-matching.rst
   graph-rewrite.rst




