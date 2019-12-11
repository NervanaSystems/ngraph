.. all.rst:

###
All
###

.. code-block:: cpp

   All // Boolean "all" reduction operation.

Description
===========

Reduces a tensor of booleans, eliminating the specified reduction axes by taking the logical conjunction (i.e., "AND-reduce").

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
| ``reduction_axes`` | The axis positions (0-based) on which to calculate the conjunction |
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

.. doxygenclass:: ngraph::op::v0::All
   :project: ngraph
   :members:
