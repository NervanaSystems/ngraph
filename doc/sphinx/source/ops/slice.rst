.. slice.rst: 

#####
Slice
#####

.. code-block:: cpp

   Slice  // Produces a sub-tensor of its input.


Description
===========

Takes a slice of an input tensor, i.e., the sub-tensor that
resides within a bounding box, optionally with a stride.


Inputs
------


+-----------------+-------------------------+----------------------------------+
| Name            | Element Type            | Shape                            |
+=================+=========================+==================================+
| `arg`           | Any                     | :math:`D=D_1, D_2, \ldots, D_n`. |
+-----------------+-------------------------+----------------------------------+

Attributes
----------

+-------------------------------+-----------------------------------------------+
| Name                          | Description                                   |
+===============================+===============================================+
| `lower_bounds`                | The (inclusive) lower-bound coordinates       |
|                               | :math:`L=L_1, L_2, \ldots, L_n.`              |
+-------------------------------+-----------------------------------------------+
| `upper_bounds`                | The (exclusive) upper-bound coordinates       |
|                               | :math:`U=U_1, U_2, \ldots, U_n.`              |
+-------------------------------+-----------------------------------------------+
| `strides`                     | The strides :math:`S=S_1, S_2, \ldots, S_n`   |
|                               | for the slices. Defaults to 1s.               |
+-------------------------------+-----------------------------------------------+


Outputs
-------

+-----------------+-------------------------+-----------------------------------------------+
| Name            | Element Type            | Shape                                         |
+=================+=========================+===============================================+
| ``output``      | Same as `arg`           | :math:`D'_i=\lceil\frac{U_i-L_i}{S_i}\rceil`. |
+-----------------+-------------------------+-----------------------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_I = \mathtt{arg}_{L+I*S}

where :math:`I=I_1, I_2, \ldots, I_n` is a coordinate of the output.

C++ Interface
=============

.. doxygenclass:: ngraph::op::Slice
   :project: ngraph
   :members:

