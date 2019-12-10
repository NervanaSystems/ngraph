.. tanh.rst:

#####
Tanh
#####

.. code-block:: cpp

   Tanh  // Elementwise hyperbolic tangent operation.

Description
===========

Produce a tensor with the same shape and element typye as ``arg,``
where the value at each coordinate of ``output`` is the hyperbolic
tangent of the value of ``arg`` at the same coordinate.

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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \tanh(\mathtt{arg}_{i_0,
   \ldots, i_{n-1}})

Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow \Delta\ (1 - \mathtt{output}^2)


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Tanh
   :project: ngraph
   :members:
