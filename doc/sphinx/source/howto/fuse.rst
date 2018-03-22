.. fuse.rst  


#########
Fuse ops
#########

* :ref:`pattern_matching`
* :ref:`graph_rewrite`

In working with a :term:`function graph`, there are many ways to describe what 
happens when we need to do something with the ops (AKA "operational functions") 
from that graph. *Fusing* is the term we'll use here, but it's also described 
as: *combining*, *folding*, *collapsing*, or *merging*. One common use case for 
nGraph is to fuse a subgraph from the graph into 
:doc:`one of our core ops <../ops/index>`. In other words, nGraph can take a 
sub-set of computations from the graph and turn it into something it can use 
via :term:`fusing`.   

ngraph is an *optimizing* compiler and as such it performs a series 
of optimization passes over a given function graph to transform it into 
a different graph that is both semantically equivalent to the original 
and also possesses better runtime characteristics.
The optimization passes may include algebraic simplifications, domain-specific simplifications (ReshapeElimination TODO: add more), and
fusion. Most passes share the same M.O. (or operational structure) and consist of two stages.
1. The first stage is to locate a list of candidates (usually, subgraphs) in the given graph
2. The second stage is to transform these candidates into semantically equivalent subgraphs that would hopefully :-) run faster.

Let's consider an example: a user would like to execute a simple graph that describes the following arithmetic expression `a + b * 1` or 
`Add(a, Mul(b, 1))`. `1` is an identity element; any element multiplied by the identity element is equal to itself. In other words,
`b * 1 = b`. The writer of an algebraic-simplification pass would probably want to locate all multiplication expressions where multiplicands are multiplied by `1` (stage 1)
and `transform/simplify/replace` these expressions with just their multiplicands (stage 2).

To make the job of an optimization pass writer easier ngraph provides facilities for *locating/finding* relevant candidates using pattern matching (`pattern/matcher.hpp`)
and transforming the original graph into its simpler version (`pass/graph_rewrite.hpp`)
Let's consider the two in more details and many ways they could help the pass writer.


.. _pattern_matching: 

Before delving into the details of pattern matching, it's worthwhile to point out that
the sole purpose of pattern matching is to **find** patterns in the given graph.
What is a *pattern*? The pattern (in the context of ngraph) is simply a subgraph that could contain any operation
ngraph's IR defines (e.g. addition, subtraction, etc) and some special wildcard nodes which will be discussed in just a moment.

A good analogy would be regular expressions. One writes a regex (pattern) and runs it through some text (our input graph) to find and/or replace
the occurences of the pattern in the given text. Similarly, the pass writer constructs patterns which are just regular ngraph graphs and then
runs those patterns through given graphs.


Apply pattern matching to fuse ops
------------------------------------


.. TODO





.. _graph_rewrite:

Using GraphRewrite to fuse ops
--------------------------------

.. TODO 