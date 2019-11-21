.. xor.rst:

###
Xor
###

.. code-block:: cpp

   Xor  // Elementwise logical-xor operation


Description
===========

Produces tensor with boolean element type and shape as the two inputs,
which must themselves have boolean element type, where the value at each
coordinate of ``output`` is ``0`` (true) if ``arg0`` or ``arg1`` both
zero or both nonzero, or ``1`` otherwise.


Inputs
------

+-----------------+------------------------------+--------------------------------+
| Name            | Element Type                 | Shape                          |
+=================+==============================+================================+
| ``arg0``        | ``ngraph::element::boolean`` | any                            |
+-----------------+------------------------------+--------------------------------+
| ``arg1``        | ``ngraph::element::boolean`` | same as ``arg0``               |
+-----------------+------------------------------+--------------------------------+

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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \mathtt{arg0}_{i_0, \ldots, i_{n-1}}\, \mathtt{XOR}\, \mathtt{arg1}_{i_0, \ldots, i_{n-1}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Xor
   :project: ngraph
   :members:
