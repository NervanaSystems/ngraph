.. provenance/overview.rst

Basic concepts
==============

The term :term:`provenance` refers to the matching of device code to 
framework subgraphs; it is analogous to source code locators in 
conventional compilers, which associate regions of object code with 
source files and line numbers. Provenance is *extensible* in that it 
may also include the chain of passes that lead from the framework graph 
to the executing code. 

It can associate device code with specific tags added by a framework bridge which 
correspond to the framework ops that create the nGraph nodes. This works only for 
those transformations that take place in nGraph: the information stored 
in the nodes can include additional details about how the device code was 
chosen. For example, whenever a graph transformation is performed with one 
of the nGraph core :doc:`Ops <../ops/index>`, a lower level of abstraction 
can record information about the transformation that may be useful to 
anyone wondering why a kernel was "chosen"; a complete description of the 
steps leading to the device kernels being used, as well as all of the 
framework nodes that led to the kernel, can be obtained. 


Existing use cases
------------------

Currently, every node nGraph touches can optionally have a set of provenance 
tags, which are strings set by a framework bridge. When a set of nodes is 
replaced by a new set of nodes, a combination of heuristics and special casing 
is used to set the tags on the new nodes based on the tags from the old nodes. 

A :term:`builder` is a function that creates a sub-graph and returns a root 
node to the bridge. The bridge is not necessarily aware of the subgraph, only 
of the returned node, where it sets tags. The remaining nodes' tags are set 
by associating a set of nodes, called a *provenance group*, with the node. Any 
tags added to the node are also added to the nodes in the provenance group.

An updated implementation of the functionality of builders is the *fused op*, 
a node that can replace itself with a subgraph. When the node is expanded 
into a subgraph, a vector of values is returned, corresponding to outputs 
of the original fused op; the tags of the fused op are added to all nodes 
in the values in reverse dataflow direction, up to (though not including) the 
input values of the fused op.

