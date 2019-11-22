.. dynamic/index.rst:


Dynamic Shapes
==============

For an example on how to use dynamic shapes, see *Scenario Two* on 
the :ref:`scenario_two` documentation.

Runtime Error Checking
----------------------

Static type-checking in the presence of dynamic shapes will make optimistic 
assumptions about things like shape mismatches. For example, if an elementwise 
op is provided inputs of shapes ``(2,?)`` and ``(?,5)``, the type checker will 
proceed under the assumption that the user is not going to pass tensors with 
inconsistent shape at runtime, and therefore infer an output shape of ``(2,5)``. 
That means that shape mismatches can now occur at runtime. 


Dimension, Rank, and PartialShape Classes
-----------------------------------------

Partial shape information is expressed via the PartialShape, Dimension, and Rank, and 
classes.

.. doxygenclass:: ngraph::PartialShape
   :project: ngraph
   :members: 

=======
.. toctree::
   :name: 
   :maxdepth: 1
