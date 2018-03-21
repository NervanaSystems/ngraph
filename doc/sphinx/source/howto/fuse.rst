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

Often the first challenge of an API user, then, is to figure out how to fuse 
those ops.  There are two main ways to to this:  


.. pattern_matching: 

Apply pattern matching to fuse ops
------------------------------------


.. TODO





.. graph_rewrite:

Using GraphRewrite to fuse ops
--------------------------------

.. TODO 