.. pad.rst:

###
Pad
###

.. code-block:: cpp

   Pad // General padding operation

Description
===========

 Adds edge padding.

Inputs
------

+-------------------+-------------------------+--------------------------------+
| Name              | Element Type            | Shape                          |
+===================+=========================+================================+
| ``arg``           | Any                     | :math:`(d_1, \ldots, d_n)`     |
+-------------------+-------------------------+--------------------------------+
| ``arg_pad_value`` | Same as ``arg``         | Scalar                         |
+-------------------+-------------------------+--------------------------------+


Attributes
----------

+-----------------------+---------------------------------------------------------------------+
| Name                  | Description                                                         |
+=======================+=====================================================================+
| ``padding_below``     | Padding added before ``arg``. May be negative.                      |
+-----------------------+---------------------------------------------------------------------+
| ``padding_above``     | Padding added after ``arg``. May be negative.                       |
+-----------------------+---------------------------------------------------------------------+
| ``pad_mode``          | Padding mode: ``CONSTANT(default)``, ``EDGE`` or ``REFLECT``.       |
+-----------------------+---------------------------------------------------------------------+

Outputs
-------

+-------------------+-------------------------+--------------------------------+
| Name              | Element Type            | Shape                          |
+===================+=========================+================================+
| ``output``        | Same as ``arg``         | :math:`(d'_1, \ldots, d'_n)`   |
+-------------------+-------------------------+--------------------------------+

.. math::

   d'_i =
   \mathtt{padding\_below}_i+d_i\cdot(\mathtt{padding\_interior}_i)+\mathtt{padding\_above}_i


Takes an input tensor of shape :math:`(d_1,\dots,d_n)` and pads by
inserting a scalar :math:`x` supplied as input, in three possible
ways:

1. *exterior padding* inserts copies of :math:`x` *below or above* the
   bounds of existing rows, columns, etc.,
2. *interior padding* inserts copies of :math:`x` *between* rows, columns, etc., or
3. both of the above.

The number and position of elements to be inserted along a given axis
is determined by three attributes:

1. *the padding-below* ``CoordinateDiff`` :math:`(p_1,\ldots,p_n)`,
2. *the padding-above* ``CoordinateDiff`` :math:`(q_1,\ldots,q_n)`, and
3. *the interior padding* ``Shape`` :math:`(r_1,\ldots,r_n)`.

The output tensor will have the shape :math:`(d'_1,\dots,d'_n)` where
:math:`d'_i = p_i + (d_i - 1)(r_i + 1) + 1 + q_i` if :math:`d_i > 0`,
and :math:`d'_i = p_i + q_i` if :math:`d_i = 0`.

Example: given a :math:`3\times 3` tensor, with interior-padding sizes
of :math:`(1,2)`, padding-below of :math:`(1,2)`, padding-above of
:math:`(1,0)`, and a pad-value of :math:`42`, we obtain: ::

              42 42 42 42 42 42 42 42 42
              42 42  1 42 42  2 42 42  3
    1 2 3     42 42 42 42 42 42 42 42 42
    4 5 6 --> 42 42  4 42 42  5 42 42  6
    7 8 9     42 42 42 42 42 42 42 42 42
              42 42  7 42 42  8 42 42  9
              42 42 42 42 42 42 42 42 42

In other words we have inserted one new row between each pair of
adjacent rows, two new columns between each pair of adjacent columns,
one new row at the top and two new columns on the left, and one new
row at the bottom and zero new columns on the right; then filled the
new rows and columns with 42.

.. note::

   The terms `below` and `above` here refer respectively to lower- or
   higher-numbered coordinate indices, and numbering starts at the
   upper-left corner; thus inserting a row "below" actually inserts it
   at the "top" of the matrix.

C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Pad
   :project: ngraph
   :members:
