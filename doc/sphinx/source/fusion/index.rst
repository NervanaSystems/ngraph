.. fusion/index.rst: 


Optimize Graphs 
===============

with nGraph Compiler fusions
-----------------------------

The nGraph Compiler is an optimizing compiler. As such, it performs a series 
of optimization passes over a given function graph to translate it into a 
semantically-equivalent and inherently-optimized graph with superior runtime 
characteristics for any of nGraph's current or future backends. Indeed, a  
framework's capability to increase training performance or to reduce inference 
latency by simply adding another device of *any* specialized form factor (CPU, 
GPU, VPU, or FPGA) is one of the :doc:`key benefits <../project/about>` of 
developing upon a framework that uses the nGraph Compiler.   

In handling a :term:`function graph`, there are many ways to describe what 
happens when we translate the framework's output of ops into an nGraph 
graph. :term:`Fusion` is the term we shall use in our documentation, but the the 
action also can be described as: *combining*, *folding*, *collapsing*, or 
*merging* of graph functions. The most common use case is to *fuse* a subgraph 
from the function graph into :doc:`one of the nGraph Core ops <../ops/index>`. 

Optimization passes may include algebraic simplifications, domain-specific 
simplifications, and fusion. Most passes share the same mode of operation (or 
the same operational structure) and consist of two stages:

#. Locating a list of potential transformation candidates (usually, subgraphs) 
   in the given graph.
#. Transforming the selected candidates into semantically-equivalent subgraphs 
   that run faster and/or with less memory.

Optimization passes can be programmed ahead of time if you know what your graph 
will look like when it's ready to be executed, or the optimization passes can 
be figured out manually with *Interpreter* mode on a stateless graph. 

Let us first consider an example. A user would like to execute a simple graph 
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

To make the work of an optimization pass writer easier, the nGraph library 
includes facilities that enable the *finding* of relevant candidates using 
pattern matching (via ``pattern/matcher.hpp``), and the *transforming* of the 
original graph into a condensed version (via ``pass/graph_rewrite.hpp``).

Let's consider each of the two in more detail and many ways they can help the 
work of the optimization pass writer.



.. toctree::
   :maxdepth: 1 

   graph-rewrite.rst
   passes-that-use-matcher.rst




