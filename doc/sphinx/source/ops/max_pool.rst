.. max_pool.rst:

#######
MaxPool
#######

.. code-block:: cpp

   MaxPool  // MaxPool operations


Description
===========

Batched max pooling operation, with optional padding and window
stride.

Inputs
------

+-----------------+-------------------------+----------------------------------+
| Name            | Element Type            | Shape                            |
+=================+=========================+==================================+
| ``arg``         | any                     | :math:`(N, C, d_1, \ldots, d_n)` |
+-----------------+-------------------------+----------------------------------+

Attributes
----------

+-------------------------------+-----------------------------------------------+
| Name                          | Description                                   |
+===============================+===============================================+
| ``window_shape``              | The window shape.                             |
+-------------------------------+-----------------------------------------------+
| ``window_movement_strides``   | The window movement strides. (defaults to 1s) |
+-------------------------------+-----------------------------------------------+
| ``padding_below``             | The below-padding shape. (defaults to 0s)     |
+-------------------------------+-----------------------------------------------+
| ``padding_above``             | The above-padding shape. (defaults to 0s)     |
+-------------------------------+-----------------------------------------------+


Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | same as ``arg``         | :math:`(N,C,d'_1,\ldots,d'_n)` |
+-----------------+-------------------------+--------------------------------+

The input for max pooling is a data batch tensor of shape
:math:`(N,C,d_1,\dots,d_n)` where :math:`n > 0`, every :math:`d_i >
0`, and where :math:`N` is the batch size, and :math:`C > 0` is the
number of channels (sometimes called features).  The dimensions
:math:`(d_1,\dots,d_n)` correspond to the shape of an
:math:`n`-dimensional data item in a batch. For example, where
:math:`n=2`, the data may represent a two-dimensional image.  It also
has two attributes:

1. *the window shape* a size vector :math:`(w_1,\ldots,w_n)` where every :math:`w_i \le d_i`; and
2. *the window movement strides, optional* a vector of positive integers :math:`(s_1,\dots,s_n)`.

The output has the shape :math:`(N,C,d'_1,\ldots,d'_n)`, where :math:`d'_n = \lceil \frac{d_i - w_i + 1}{s_i} \rceil`.


Mathematical Definition
=======================

Given an input data batch tensor :math:`T_{in}`, the output tensor is defined by the equation

.. math::

        T_{out}[a,c,i_1,\dots,i_n] =
	\max_{j_1 = s_1 i_1, \dots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \dots, j_n = s_n i_n + w_n - 1} (T_{in}[a,c,j_1,\dots,j_n])


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::MaxPool
   :project: ngraph
   :members:
