.. inspection/index: 

Visualization Tools
###################

nGraph provides serialization and deserialization facilities along with the 
ability to create image formats. When visualization is enabled, a ``dot`` file 
is generated, along with a ``png``. The default can be adjusted by setting the 
``NGRAPH_VISUALIZE_TREE_OUTPUT_FORMAT`` flag to another format, like PDF. 

.. note:: Large graphs are usually not legible with formats like PDF.  

Large graphs may require additional work to get into a human-readable format.

Manual adjustments with Graphviz
================================

As we are visualizing the graph, we can make some tweaks to the generated ``dot`` 
file to make routing more tractable for Graphviz. Note that there are many possible 
ways to go about this; what follows here is one way. 

First trim edges that, intuitively speaking, have long "skip distance", which can 
be useful for training graphs; those tend to have very long feed-forward edges where 
intermediate values from ``fprop`` are stored for later reuse in the ``bprop`` phase:

.. code-block:: raw
   
   Actual Graph Structure      Visualization
    n0                             n0
    | \                            |  \
    n1 \                           n1  [to n50]
    |   |                          |
    n2  |                          n2
    |   |                          |
    n3  |                          n3
    |   |                          |
   ...  |                         ...  [from n0]
    |  /                           |  /
   n50                            n50


Efficiently detecting a "long skip" is a bit tricky. We want to come up with a metric 
that is reasonably fast to compute, but that does not result in cuts that will split 
the graph itself into multiple components. The heuristic we are using for the jump 
distance between ``n`` and ``m`` is the maximum difference in maximum path 
length from ``n`` and ``m`` to any result node that is reachable from both 
``n`` and ``m`` (or ``0``, if no such result node exists). 

Formally:

**Compute-Heights-Above-Each-Parameter:** ``N``

.. literalinclude:: ../../../../../ngraph/src/ngraph/pass/visualize_tree.cpp
   :language: cpp
   :lines: 71-82


**Jump-Distance:** (``n``, ``m``, ``height_maps``)

.. literalinclude:: ../../../../../ngraph/src/ngraph/pass/visualize_tree.cpp
   :language: cpp
   :lines: 85-92


Later on, if ``E`` is an edge from ``n`` to ``m``, and Jump-Distance(n,m,height_map) > K (where K is kind
of arbitrary but currently set to 20), we will "cut" the edge as illustrated above.

Another tweak aims to eliminate routing pressure from nodes that have large outdegree and
are connected to many otherwise-distant places in the graph. For this, the only thing we are
doing at the moment is to "float" Parameter and Constant nodes. This means that rather than
visualizing them as a single node (which might have very large outdegree as in, e.g., a
learning rate parameter being fed to many different places), we make a "copy" of the node at
each occurrence site (with a dashed outline).

See the full code at `visualize_tree.cpp`_ 

Additional scripts
==================

We have provided a script to convert the `most common default output`_, nGraph JSON,
to an output that is better able to handle detailed graphs; however, we do not 
offer user support for this script. After running the script, you should have a 
``.graphml`` that can be imported and inspected with third-party tools like: 

#. `Gephi`_

#. `Cytoscape`_

.. #. `Netron`_ support tentatively planned to come soon


.. _CMakeLists.txt: https:github.com/NervanaSystems/ngraph/blob/master/CMakeLists.txt
.. _most common default output: https:github.com/NervanaSystems/ngraph/contrib/tools/graphml/ngraph_json_to_graphml.py
.. _visualize_tree.cpp: https://github.com/NervanaSystems/ngraph/blob/master/src/ngraph/pass/visualize_tree.cpp
.. _Netron: https:github.com/lutzroeder/netron/blob/master/README.md
.. _Gephi: https:gephi.org
.. _Cytoscape: https:cytoscape.org

