.. inspection/index: 

Visualization Tools
###################

One option for visualizing the graph or node structure of a graph is to compile 
with :envvar:`-DNGRAPH_JSON_ENABLE` flags (see also ``CMakeLists.txt``) to 
conditionally enable some kinds of formats that can be converted to formats like 
PDF. Note that large graphs cannot be viewed with formats like PDF.

Large graphs may require additional work to get into a human-readable format. We 
have included an `ngraph_converter`_ script (not officially supported), to test 
working with other formats that can handle large graphs.

What follows here is a non-comprehensive list of third-party visualization tools or 
aides for loading and analyzing graphs.

#. `Netron`_

#. `Gephi`_

#. `Cytoscape`_

#. `TFBoard`_


.. _ngraph_converter: see /ngraph/contrib/tools/ngraph_converter/ngc_util.py
.. _Netron: https://github.com/lutzroeder/netron/blob/master/README.md
.. _Gephi: https://gephi.org
.. _Cytoscape: https://cytoscape.org
.. _TFBoard: https://www.tensorflow.org/guide/summaries_and_tensorboard

