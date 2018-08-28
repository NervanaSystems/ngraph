.. add.rst:

###
Add
###

.. code-block:: cpp

   Add  // Elementwise add operation


Description
===========

Elementwise add operation.

Produces tensor of the same element type and shape as the two inputs,
where the value at each coordinate of ``output`` is the sum of the
values at the corresponding input coordinates.

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg0``        | any                     | any                            |
+-----------------+-------------------------+--------------------------------+
| ``arg1``        | same as ``arg0``        | same as ``arg0``               |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | same as ``arg0``        | same as ``arg0``               |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \mathtt{arg0}_{i_0, \ldots, i_{n-1}} + \mathtt{arg1}_{i_0, \ldots, i_{n-1}}

Backprop
========

.. math::

   \overline{\mathtt{arg0}} &\leftarrow \Delta \\
   \overline{\mathtt{arg1}} &\leftarrow \Delta


C++ Interface
=============

.. doxygenclass:: ngraph::op::Add
   :project: ngraph
   :members:
