.. convert.rst:

#######
Convert
#######

.. code-block:: cpp
   
   Convert // Convert a tensor from one element type to another


Description
===========

.. TODO 

Long description

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Attributes
----------

+------------------+---------------------------+---------------------------------+
| Name             | Type                      | Notes                           |
+==================+===========================+=================================+
| ``element_type`` | ``ngraph::element::type`` | The element type of the result  |
+------------------+---------------------------+---------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | ``element_type``        | Same as ``arg``                |
+-----------------+-------------------------+--------------------------------+


Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \mathtt{Convert}(\Delta,\mathtt{arg->get_element_type()})


C++ Interface
=============

.. doxygenclass:: ngraph::op::Convert
   :project: ngraph
   :members:
