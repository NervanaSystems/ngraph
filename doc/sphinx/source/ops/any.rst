.. any.rst:

###
Any
###

.. code-block:: cpp

   Any // Boolean "any" reduction operation.

Description
===========

Reduces a tensor of booleans, eliminating the specified reduction axes by taking the logical disjunction (i.e., "OR-reduce").

Inputs
------

+-----------------+------------------------------+--------------------------------+
| Name            | Element Type                 | Shape                          |
+=================+==============================+================================+
| ``arg``         | ``ngraph::element::boolean`` | Any                            |
+-----------------+------------------------------+--------------------------------+

Attributes
----------
+--------------------+--------------------------------------------------------------------+
| Name               | Description                                                        |
+====================+====================================================================+
| ``reduction_axes`` | The axis positions (0-based) on which to calculate the disjunction |
+--------------------+--------------------------------------------------------------------+

Outputs
-------

+-----------------+-------------------------+---------------------------------------------------+
| Name            | Element Type            | Shape                                             |
+=================+=========================+===================================================+
| ``output``      | Same as ``arg``         | Same as ``arg``, with ``reduction_axes`` removed. |
+-----------------+-------------------------+---------------------------------------------------+

C++ Interface
=============

.. doxygenclass:: ngraph::op::Any
   :project: ngraph
   :members:
