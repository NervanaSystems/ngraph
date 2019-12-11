.. inspection/index: 

.. _inspection: 

Visualization Tools
###################

nGraph provides serialization and deserialization facilities, along with the 
ability to create image formats or a PDF. 

When visualization is enabled, ``svg`` files for your graph get generated. The 
default can be adjusted by setting the ``NGRAPH_VISUALIZE_TRACING_FORMAT`` 
flag to another format, like PNG or PDF. 

.. note:: Large graphs are usually not legible with formats like PDF.

Large graphs may require additional work to get into a human-readable format. 
On the back end, very long edges will need to be cut to make (for example) a 
hard-to-render training graph tractable. This can be a tedious process, so 
incorporating the help of a rendering engine or third-party tool like those 
listed below may be useful.  


.. Additional scripts
.. ==================

.. We have provided a script to convert the `most common default output`_, nGraph 
.. ``JSON``, to an output that is better able to handle detailed graphs; however, 
.. we do not offer user support for this script. The script will produce a 
.. ``.graphml`` file that can be imported and inspected with third-party tools 
.. like: 

#. `Gephi`_

#. `Cytoscape`_

.. #. `Netron`_ support tentatively planned to come soon


.. _CMakeLists.txt: https:github.com/NervanaSystems/ngraph/blob/master/CMakeLists.txt
.. _most common default output: https:github.com/NervanaSystems/ngraph/contrib/tools/graphml/ngraph_json_to_graphml.py
.. _visualize_tree.cpp: https://github.com/NervanaSystems/ngraph/blob/master/src/ngraph/pass/visualize_tree.cpp
.. _Netron: https:github.com/lutzroeder/netron/blob/master/README.md
.. _Gephi: https:gephi.org
.. _Cytoscape: https:cytoscape.org
