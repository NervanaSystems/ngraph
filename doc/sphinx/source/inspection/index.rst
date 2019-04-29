.. inspection/index: 

Visualization Tools
###################

One option for visualizing the graph or node structure of a graph is to compile 
with :envvar:`-DNGRAPH_JSON_ENABLE` flags (see also `CMakeLists.txt`_) to 
conditionally enable some kinds of serialization that can be converted to image 
files, or to formats like PDF. Note that large graphs are usually not legible 
with formats like PDF.

Large graphs may require additional work to get into a human-readable format. We 
have provided a script to convert the `most common default output`_  to an 
output that is better able to handle detailed graphs; however, we do not offer 
user support for this script.

What follows here is a non-comprehensive list of third-party visualization tools 
or aides for loading and analyzing large or highly-detailed graphs:

#. `Netron`_

#. `Gephi`_

#. `Cytoscape`_

#. `TFBoard`_


.. _CMakeLists.txt: https://github.com/NervanaSystems/ngraph/blob/master/CMakeLists.txt
.. _most common default output: https://github.com/NervanaSystems/ngraph/contrib/tools/graphml/ngraph_json_to_graphml.py
.. _Netron: https://github.com/lutzroeder/netron/blob/master/README.md
.. _Gephi: https://gephi.org
.. _Cytoscape: https://cytoscape.org
.. _TFBoard: https://www.tensorflow.org/guide/summaries_and_tensorboard



