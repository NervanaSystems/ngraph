.. dynamic/index.rst:


Dynamic Shapes
==============

For an example on how to use dynamic shapes, see the :ref:`scenario_two` 
documentation.

Runtime Error Checking
----------------------

Static type-checking in the presence of dynamic shapes will make optimistic 
assumptions about things like shape mismatches. For example, if an elementwise 
op is provided inputs of shapes ``(2,?)`` and ``(?,5)``, the type checker will 
proceed under the assumption that the user is not going to pass tensors with 
inconsistent shape at runtime, and therefore infer an output shape of ``(2,5)``. 
That means that shape mismatches can now occur at runtime. 


.. _partial_shapes:

PartialShape, Dimension, and Rank Classes
-----------------------------------------

Partial shape information is expressed via the ``PartialShape``, ``Dimension``, 
and ``Rank`` classes.

.. note:: ``Rank``  is an alias for ``Dimension``, used when the value represents 
   the number of axes in a shape, rather than the size of one dimension in a shape. 


.. doxygenclass:: ngraph::PartialShape
   :project: ngraph
   :members: 


.. doxygenclass:: ngraph::Dimension
   :project: ngraph
   :members: 

