.. ceiling.rst:

#######
Ceiling
#######

Description
===========

Elementwise ceiling operation.

Produces a single output tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the ceiling of the
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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \lceil \mathtt{arg}_{i_0,
   \ldots, i_{n-1}}\rceil

Backprop
========

Not defined.

C++ Interface
=============

.. doxygenclass:: ngraph::op::Ceiling
   :members:
