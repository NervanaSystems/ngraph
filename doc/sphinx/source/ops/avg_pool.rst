.. avg_pool.rst:

#######
AvgPool
#######

Average Pooling operation.

Average pooling takes as its input an data batch tensor of shape
:math:`(N,C,d_1,\ldots,d_n)` where :math:`n > 0`, every :math:`d_i >
0`, and where :math:`N` is the batch size, and :math:`C > 0` is the
number of channels (sometimes called features).  The dimensions
:math:`(d_1,\ldots,d_n)` correspond to the shape of an
:math:`n`-dimensional data item in a batch. For example, where
:math:`n=2`, the data may represent a two-dimensional image. It also
takes four parameters:
        
1. *(the window shape)* a size vector :math:`(w_1,\ldots,w_n)` where every :math:`w_i \le d_i`; and
2. *(the window movement strides, optional)* a vector of positive integers :math:`(s_1,\ldots,s_n)`.
3. *(the padding below, optional)* a vector of positive integers :math:`(p_1,\ldots,p_n)`.
4. *(the padding above, optional)* a vector of positive integers :math:`(q_1,\ldots,q_n)`.
        
The output has the shape :math:`(N,C,d'_1,\ldots,d'_n)`, where
:math:`d'_n = \lceil \frac{p_i + d_i + q_i - w_i + 1}{s_i} \rceil`.
        
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

.. doxygenclass:: ngraph::op::AvgPool
   :members:

Python Interface
================

is not merged yet, but could go here!
