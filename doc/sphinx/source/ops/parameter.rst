.. parameter.rst

#########
Parameter
#########

.. code-block: cpp

   Paramerer // A function parameter.

Description
===========

Parameters are nodes that represent the arguments that will be passed to user-defined functions.  Function creation requires a sequence of parameters.  Basic graph operations do not need parameters attached to a function.

Attributes
----------

+------------------+------------------------------------------+
| Name             | Description                              |
+==================+==========================================+
| ``element_type`` | The ``element::Type`` of the parameter.  |
+------------------+------------------------------------------+
| ``shape``        | The ``Shape`` of the parameter.          |
+------------------+------------------------------------------+

Outputs
-------

+------------+------------------+------------+
| Name       | Element type     | Shape      |
+============+==================+============+
| ``output`` | ``element_type`` | ``shape``  |
+------------+------------------+------------+

A ``Parameter`` produces the value of the tensor passed to the function in the position of the parameter in the function's arguments. The passed tensor must have the element type and shape specified by the parameter.

Backprop
========

.. math::

   \leftarrow \Delta


C++ Interface
=============

.. doxygenclass:: ngraph::op::Parameter
   :project: ngraph
   :members:
