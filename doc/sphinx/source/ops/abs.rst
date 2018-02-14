.. abs.rst:

###
Abs
###

Description
===========

Elementwise absolute value operation.

Produces a single output tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the absoloute value of the
value at each ``arg`` coordinate.

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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \left|\mathtt{arg}_{i_0,
   \ldots, i_{n-1}}\right|

Backprop
========

.. math::

   \overline{\texttt{arg}} \leftarrow \Delta\ \mathrm{sgn}(\texttt{arg})


C++ Interface
=============

.. doxygenclass:: ngraph::op::Abs
   :members:
