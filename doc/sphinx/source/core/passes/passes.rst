.. core/passes:

Compiler passes
===============

.. toctree::
   :maxdepth: 1
   :caption: Compiler passes 

   list-of-passes.rst 
   passes-that-use-matcher.rst



Overview: Generic graph optimization passes
-------------------------------------------

The pass manager infrastructure in nGraph makes it easy to reuse and mix the 
generic optimization passes. It also permits you to roll your own device-specific 
optimizations; that is, the same unified interface and APIs may be used to 
cover both things.

Invoking these passes is fairly straightforward:  

#. Create a "pass manager" object. 
#. Populate it with the desired pass(es). 
#. Invoke the pass manager with a pointer to your unoptimized graph, and it’ll return a pointer 
   to an optimized graph.

nGraph Core includes a large library of hardware-agnostic passes useful 
for almost any kind of hardware backend. Some of these passes are likely familiar 
to people who are comfortable with classical compiler designs. Others, like the 
reshape/transpose elimination and sinking passes, are quite specific to deep 
learning.

Example of Passes
-----------------

The effectiveness of graph-level optimization with nGraph is more striking to look 
at in terms of an actual input graph, such as one from the framework bridge.

*Figure A* shows an excerpt from ``MobileNet v1``, a topology which makes heavy 
use of group convolution.

.. _figure-mobilenet-gc:

.. figure:: ../../graphics/mobilenet-group-conv.png
   :width: 700px
   :alt: 

   Figure A: Each of these grouped convolution complexes -- the 
   operations within the rectangles on the left -- is very wide; each is too 
   wide to fit legibly on the illustration.

The group convolution fusion is able to replace each of those giant subgraphs 
with a single CPU group convolution node. This ends up being a win in several 
ways: 

* sheer node count, 
* mappability to MKL-DNN (which has an accelerated group convolution implementation), 
* elimination of unnecessary temporaries, and so on.