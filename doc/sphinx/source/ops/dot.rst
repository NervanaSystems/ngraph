.. dot.rst:

###
Dot
###

Description
===========

Generalized dot product operation, including scalar-tensor product,
matrix-vector product, and matrix multiplication.

A few common cases are as follows:

* If :math:`m = 0` and :math:`n = 1` or :math:`p = 1`, the operation
  is a scalar-tensor product.
* If :math:`m = 1`, :math:`n = 2`, and :math:`p = 1`, the operation is
  a matrix-vector product.
* If :math:`m = 1` and :math:`n = p = 2`, the operation is a matrix
  multiplication.


Inputs
------

+-----------------+-------------------------+-----------------------------------------+
| Name            | Element Type            | Shape                                   |
+=================+=========================+=========================================+
| ``arg0``        | any                     | :math:`(i_1,\dots,i_n,j_1,\dots,j_m)`   |
+-----------------+-------------------------+-----------------------------------------+
| ``arg1``        | same as ``arg0``        | :math:`(j_1,\ldots,j_m,k_1,\dots,k_p)`  |
+-----------------+-------------------------+-----------------------------------------+

Attributes
----------

+------------------------+---------------+--------------------------------------------------+
| Name                   |               |                                                  |
+========================+===============+==================================================+
| reduction_axes_count   | ``size_t``    | The number of axes to reduce through dot-product |
|                        |               | (corresponds to :math:`m` in the formulas above) |
+------------------------+---------------+--------------------------------------------------+

Outputs
-------

+-----------------+-------------------------+----------------------------------------+
| Name            | Element Type            | Shape                                  |
+=================+=========================+========================================+
| ``output``      | same as ``arg0``        | :math:`(i_1,\ldots,i_n,k_1,\dots,k_p)` |
+-----------------+-------------------------+----------------------------------------+


Mathematical Definition
=======================

.. math::

   \texttt{output}_{i_1,\dots,i_n,k_1,\ldots,k_p} =
   \begin{cases}
   \texttt{arg0}_{i_1,\dots,i_n} \cdot
   \texttt{arg1}_{k_1,\dots,k_p}&\text{if }m=0,\\
   \sum_{j_1, \ldots, j_m}
   \texttt{arg0}_{i_1,\dots,i_n,j_1,\dots,j_m}
   \cdot
   \texttt{arg1}_{j_1,\ldots,j_m,k_1,\ldots,k_p}
   &\text{otherwise}.
   \end{cases}


Backprop
========

To be documented.


C++ Interface
=============

.. doxygenclass:: ngraph::op::Dot
   :members:
