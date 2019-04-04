.. python_api/index.rst

###########
Python API 
###########

This section contains the Python API component of the nGraph Compiler stack. The 
Python API exposes nGraph™ C++ operations to Python users. For quick-start you 
can find an example of the API usage below. 

Note that the output at ``print(model)`` may vary; it varies according to the 
number of nodes or variety of :term:`step` used to compute the printed solution. 
Various NNs configured in different ways should produce the same result for 
simple calculations or accountings. More complex computations may have minor 
variations with respect to how precise they ought to be. For example, a more 
efficient graph ``<Multiply: 'Multiply_12' ([2, 2])>`` can also be achieved 
with some configurations.


.. literalinclude:: ../../../../python/examples/basic.py
   :language: python
   :lines: 18-47	
   :caption:  "Basic example"

=======

.. toctree::
   :maxdepth: 1
   :titlesonly:

   structure
   List of operations <_autosummary/ngraph.ops.rst>
