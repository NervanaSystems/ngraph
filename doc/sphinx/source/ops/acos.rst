.. acos.rst:

####
Acos
####

Elementwise acos operation.

Produces a single output tensor of the same element type and shape as the input,
where the value at each coordinate of the output is the acos of the
value at each input coordinate.

+-----------------+-------------------------+--------------------------------+
| Input Name      | Element Type            | Shape                          |
+=================+=========================+================================+
| ``input``       | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

+-----------------+-------------------------+--------------------------------+
| Output Name     | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``input``       | Same as input.                 |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   output_{i_0, \ldots, i_{n-1}} = \mathrm{cos}^{-1}(input_{i_0, \ldots, i_{n-1}})

Backprop
========

.. math::

   \overline{input} \leftarrow -\frac{\Delta}{\sin{output}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Acos
   :members:

Python Interface
================

is not merged yet, but could go here!
