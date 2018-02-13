.. acos.rst:

####
Acos
####

Description
===========

Elementwise acos operation.

Produces a tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the inverse cosine of the
value at the corresponding coordinate of ``arg`` .

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``arg``         | Same as ``arg``.               |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \texttt{output}_{i_0, \ldots, i_{n-1}} = \cos^{-1}(\texttt{arg}_{i_0, \ldots, i_{n-1}})

Backprop
========

.. math::

   \overline{\texttt{arg}} \leftarrow -\frac{\Delta}{\sqrt{1-\texttt{arg}^2}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Acos
   :members:
