.. inspection/index: 

Visualization Tools
###################

nGraph provides serialization and deserialization facilities along with the 
ability to create image formats. When visualization is enabled, a ``dot`` file 
is generated, along with a ``png``. The default can be adjusted by setting the 
``NGRAPH_VISUALIZE_TREE_OUTPUT_FORMAT`` flag to another format, like PDF. 

Note: Large graphs are usually not legible with formats like PDF.  

Large graphs may require additional work to get into a human-readable format. We 
have provided a script to convert the `most common default output`_, nGraph JSON,
to an output that is better able to handle detailed graphs; however, we do not 
offer user support for this script. After running the script, you should have a 
``.graphml`` that can be imported and inspected with third-party tools like: 

#. `Gephi`_

#. `Cytoscape`_

.. #. `Netron`_ support tentatively planned to come soon


.. _CMakeLists.txt: https://github.com/NervanaSystems/ngraph/blob/master/CMakeLists.txt
.. _most common default output: https://github.com/NervanaSystems/ngraph/contrib/tools/graphml/ngraph_json_to_graphml.py
.. _Netron: https://github.com/lutzroeder/netron/blob/master/README.md
.. _Gephi: https://gephi.org
.. _Cytoscape: https://cytoscape.org

