.. negative.rst:

########
Negative
########

.. code-block:: cpp

   Negative  //  Elementwise negative operation


Description
===========

Produces a single output tensor of the same element type and shape as ``arg``,
where the value at each coordinate of ``output`` is the negative of the
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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = -\mathtt{arg}_{i_0,
   \ldots, i_{n-1}}

Backprop
========

.. math::

   \overline{\mathtt{arg}} \leftarrow -\Delta


C++ Interface
=============

.. doxygenclass:: ngraph::op::Negative
   :project: ngraph
   :members:
