.. divide.rst:

######
Divide
######

.. code-block:: cpp

   Divide  //  Elementwise divide operation


Description
===========

Produces tensor of the same element type and shape as the two inputs,
where the value at each coordinate of ``output`` is the quotient of the
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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \frac{\mathtt{arg0}_{i_0, \ldots, i_{n-1}}}{\mathtt{arg1}_{i_0, \ldots, i_{n-1}}}

Backprop
========

.. math::

   \overline{\mathtt{arg0}} &\leftarrow \frac{\Delta}{\mathtt{arg1}}\\
   \overline{\mathtt{arg1}} &\leftarrow -\Delta \frac{\mathtt{Output}}{\mathtt{arg1}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Divide
   :project: ngraph
   :members:
