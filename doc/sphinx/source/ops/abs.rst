.. abs.rst:

###
Abs
###

.. code-block:: cpp

   Abs  // Elementwise absolute value operation


Description
===========

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
| ``output``      | Same as ``arg``         | Same as ``arg``                |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \left|\mathtt{arg}_{i_0,
   \ldots, i_{n-1}}\right|

Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \Delta\ \mathrm{sgn}(\mathtt{arg})


C++ Interface
=============

.. doxygenclass:: ngraph::op::Abs
   :project: ngraph
   :members:
