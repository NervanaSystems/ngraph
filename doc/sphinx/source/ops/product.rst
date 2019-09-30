.. product.rst:

#######
Product
#######

.. code-block:: cpp

   Product // Product reduction operation.

Description
===========

Reduces the tensor, eliminating the specified reduction axes by taking the product.

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Attributes
----------
+--------------------+----------------------------------------------------------------+
| Name               | Description                                                    |
+====================+================================================================+
| ``reduction_axes`` | The axis positions (0-based) on which to calculate the product |
+--------------------+----------------------------------------------------------------+

Outputs
-------

+-----------------+-------------------------+---------------------------------------------------+
| Name            | Element Type            | Shape                                             |
+=================+=========================+===================================================+
| ``output``      | Same as ``arg``         | Same as ``arg``, with ``reduction_axes`` removed. |
+-----------------+-------------------------+---------------------------------------------------+

Mathematical Definition
=======================

.. math::

   \mathit{product}\left(\{0\},
   \left[ \begin{array}{ccc}
   1 & 2 \\
   3 & 4 \\
   5 & 6 \end{array} \right]\right) &=
   \left[ (1 * 3 * 5), (2 * 4 * 6) \right] =
   \left[ 15, 48 \right]&\text{ dimension 0 (rows) is eliminated} \\
   \mathit{product}\left(\{1\},
   \left[ \begin{array}{ccc}
   1 & 2 \\
   3 & 4 \\
   5 & 6 \end{array} \right]\right) &=
   \left[ (1 * 2), (3 * 4), (5 * 6) \right] =
   \left[ 2, 12, 30 \right]&\text{ dimension 1 (columns) is eliminated}\\
   \mathit{product}\left(\{0,1\},
   \left[ \begin{array}{ccc}
   1 & 2 \\
   3 & 4 \\
   5 & 6 \end{array} \right]\right) &=
   (1 * 2) * (3 * 4) * (5 * 6) =
   720&\text{ both dimensions (rows and columns) are eliminated}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Product
   :project: ngraph
   :members:
