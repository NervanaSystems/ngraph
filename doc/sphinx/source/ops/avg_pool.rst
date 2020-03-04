.. avg_pool.rst:

#######
AvgPool
#######

.. code-block:: cpp

   AvgPool  // Average Pooling operation


Description
===========

Average pooling windows its input and produces an average for each window.

Inputs
------

+-----------------+----------------+--------------------------------+--------------------+
| Name            | Element Type   | Shape                          | Notes              |
+=================+================+================================+====================+
| ``data``        | Any            | :math:`(N,C,d_1,\ldots,d_n)`   | :math:`n>0, d_i>0` |
+-----------------+----------------+--------------------------------+--------------------+


Attributes
----------

+----------------------+-----------------+----------------------------------+
| Name                 | Type            | Notes                            |
+======================+=================+==================================+
| ``w``                | ``Shape[n]``    | Window shape. :math:`w_i\le d_i` |
+----------------------+-----------------+----------------------------------+
| ``s``                | ``Strides[n]``  | Window strides.                  |
+----------------------+-----------------+----------------------------------+
| ``p``                | ``Shape[n]``    | Padding below.                   |
+----------------------+-----------------+----------------------------------+
| ``q``                | ``Shape[n]``    | Padding above.                   |
+----------------------+-----------------+----------------------------------+
| ``i``                | ``Boolean``     | Include padding in average.      |
+----------------------+-----------------+----------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Any                     | :math:`(N,C,d'_1,\ldots,d'_n)` |
+-----------------+-------------------------+--------------------------------+


Average pooling takes as its input, a batch tensor `data` of shape
:math:`(N,C,d_1,\ldots,d_n)`, where  where :math:`N` is the batch
size, and :math:`C > 0` is the
number of channels (sometimes called features). The dimensions
:math:`(d_1,\ldots,d_n)` correspond to the shape of an
:math:`n`-dimensional data item in a batch. For example, where
:math:`n=2`, the data may represent a two-dimensional image. It also
takes four attributes:

1. *window shape*,
2. *window movement strides*, (optional)
3. *padding below*, (optional)
4. *padding above*, (optional)
5. *include padding in average*

The shape of `output` is :math:`(N,C,d'_1,\ldots,d'_n)`, where
:math:`d'_n = \lceil \frac{p_i + d_i + q_i - w_i + 1}{s_i} \rceil`.

**Informal definition:**
If :math:`\textit{i}` is :math:`\textit{true}`, then averages are computed as though the
padding region contained regular elements of value zero.
If :math:`\textit{i}` is :math:`\textit{false}`, then averages are computed using only the non-padding
tensor elements that are present in each window.

*Example:* Consider two instances of this operator with the following attributes:
:math:`\textit{w} = (2,2)`,
:math:`\textit{s} = (1,1)`,
:math:`\textit{p} = (1,1)`,
and (in one instance) :math:`\textit{i} = false` or (in the other instance) :math:`\textit{i} = true`.

Consider how those two operator instances would handle this input tensor:

.. math::

  T_\textit{in} = \begin{bmatrix}
     1     &  3     &  5     & \ldots \\
     7     & 11     & 13     & \ldots \\
    17     & 19     & 23     & \ldots \\
    \vdots & \vdots & \vdots & \ddots
  \end{bmatrix}


Applying the padding indicated by the value of :math:`\textit{p}`, we have the padded image of :math:`T_\textit{in}`
as follows:

.. math::

  T_\textit{in,padded} = \begin{bmatrix}
   (0) & (0)     & (0)    & (0)      & \ldots \\
   (0) &   1     &   3    &   5      & \ldots \\
   (0) &   7     &  11    &  13      & \ldots \\
   (0) &  17     &  19    &  23      & \ldots \\
   (0) &  \vdots & \vdots &  \vdots  & \ddots
  \end{bmatrix}

Now consider how the two variations of this example's *AvgPool* operator will compute the "average" value
of the top-left window, which contains exactly the elements:

.. math::

  \begin{bmatrix}
   (0) & (0)   \\
   (0) &   1
  \end{bmatrix}

If :math:`\textit{i} = false`, then the operator simply ignores the padding elements.  It therefore computes the
average of the single-element set :math:`\{ 1 \}`, yielding :math:`1.0`.

If :math:`\textit{i} = true`, then the operator computes the average of the set :math:`\{ 0, 0, 0, 1\}`,
yielding `0.25`.

*Note:* This operator is ill-defined when *both* of the following conditions hold:
(1) :math:`\textit{i} = false`, and (2) the operator's other attribute values indicate
that at least one window will contain only padding elements.

**Formal definition:**
*In the absence of padding*, given an input data batch tensor
:math:`T_\textit{in}`, the output tensor is defined by the equation

.. math::

   T_\textit{out}[a,c,i_1,\ldots,i_n] =
   \frac{\sum_{j_1 = s_1 i_1, \ldots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \ldots, j_n = s_n i_n + w_n - 1}
   T_\textit{in}[a,c,j_1,\ldots,j_n]}{\prod_{i=1}^n{w_n}}

*In the presence of padding*, we do not always want to divide by a
reciprocal equal to the number of elements in the window, since some
of the output points are determined by a window that is partly hanging
beyond the edge of the tensor. In this case we can define the output


In this case we can define the output
via a few intermediate steps.

First define the *sum tensor* :math:`T_\textit{sum}`, with shape
:math:`(N,C,d'_1,\ldots,d'_n)`, as follows.

.. math::

   T_\textit{sum}[a,c,i_1,\ldots,i_n] =
   \frac{\sum_{j_1 = s_1 i_1, \ldots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \ldots, j_n = s_n i_n + w_n - 1}
   \textit{val}[a,c,j_1,\ldots,j_n]}{\prod_{i=1}^n{w_n}}

where

.. math::

   \textit{val}[a,c,j_1,\ldots,j_n] =
   \begin{cases}
   T_\textit{in}[a,c,j_1,\ldots,j_n]&\text{if for all } k, p_k \le j_k < p_k + d_k\\
   0&\text{otherwise}.
   \end{cases}

Second, define the *divisor tensor* :math:`T_\textit{div}`, with shape :math:`(N,C,d'_1,\ldots,d'_n)`, as follows.

.. math::

   T_\textit{div}[a,c,i_1,\ldots,i_n] =
   \frac{\sum_{j_1 = s_1 i_1, \ldots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \ldots, j_n = s_n i_n + w_n - 1}
   \textit{val}[a,c,j_1,\ldots,j_n]}{\prod_{i=1}^n{w_n}}

where

.. math::

   \textit{val}[a,c,j_1,\ldots,j_n] =
   \begin{cases}
   1&\text{if for all }k, p_k \le j_k < p_k + d_k\\
   0&\text{otherwise}.
   \end{cases}

Finally, define :math:`T_\textit{out}` as the result of elementwise
dividing :math:`T_\textit{sum}` by :math:`T_\textit{div}`.  Note that
at positions where :math:`T_\textit{div}` is zero, values may be
infinity or nan.  (This corresponds to a condition where the pooling
window is completely out of bounds, encompassing no valid values.)

Backprop
========


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::AvgPool
   :project: ngraph
   :members:

