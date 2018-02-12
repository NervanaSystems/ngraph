.. add.rst:

###
Add
###

Elementwise add operation.

Produces a single output tensor of the same element type and shape as the input,
where the value at each coordinate of the output is the acos of the
value at each input coordinate.

+-----------------+-------------------------+--------------------------------+
| Input Name      | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg0``        | any                     | any                            |
+-----------------+-------------------------+--------------------------------+
| ``arg1``        | same as ``arg0``        | same as ``arg0``               |
+-----------------+-------------------------+--------------------------------+

+-----------------+-------------------------+--------------------------------+
| Output Name     | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | same as ``arg0``        | same as ``arg0``               |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   output_{i_0, \ldots, i_{n-1}} = arg0_{i_0, \ldots, i_{n-1}} + arg1_{i_0, \ldots, i_{n-1}}

Backprop
========

.. math::

   \overline{arg0} &\leftarrow \Delta \\
   \overline{arg1} &\leftarrow \Delta


C++ Interface
=============

.. doxygenclass:: ngraph::op::Add
   :members:

Python Interface
================

is not merged yet, but could go here!
