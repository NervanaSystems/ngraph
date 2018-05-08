.. parameter.rst

#########
Parameter
#########

.. code-block: cpp

   Parameter // A function parameter.

Description
===========

Parameters are nodes that represent the arguments that will be passed
to user-defined functions.  Function creation requires a sequence of
parameters.

Attributes
----------

+------------------+------------------------------------------+
| Name             | Description                              |
+==================+==========================================+
| ``element_type`` | The ``element::Type`` of the parameter.  |
+------------------+------------------------------------------+
| ``shape``        | The ``Shape`` of the parameter.          |
+------------------+------------------------------------------+
| ``cacheable``    | True if the parameter is not expected to |
|                  | be frequently updated.                   |
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
