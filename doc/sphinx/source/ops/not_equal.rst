.. not_equal.rst:

#####
NotEqual
#####

Description
===========

Elementwise not equal operation.

Produces tensor of the same element type and shape as the two inputs,
where the value at each coordinate of ``output`` is true (1) if
``arg0`` is not equal to ``arg1``, 0 otherwise.


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

+-----------------+------------------------------+--------------------------------+
| Name            | Element Type                 | Shape                          |
+=================+==============================+================================+
| ``output``      | ``ngraph::element::boolean`` | same as ``arg0``               |
+-----------------+------------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \texttt{output}_{i_0, \ldots, i_{n-1}} = \texttt{arg0}_{i_0, \ldots, i_{n-1}} \neq \texttt{arg1}_{i_0, \ldots, i_{n-1}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::NotEqual
   :members:
