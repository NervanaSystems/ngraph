.. or.rst:

##
Or
##

.. code-block:: cpp

   Or  // Elementwise logical-or operation


Description
===========

Produces tensor with boolean element type and shape as the two inputs,
which must themselves have boolean element type, where the value at each
coordinate of ``output`` is ``1`` (true) if ``arg0`` or ``arg1`` is
nonzero, ``0`` otherwise.


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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \mathtt{arg0}_{i_0, \ldots, i_{n-1}}\, \mathtt{||}\, \mathtt{arg1}_{i_0, \ldots, i_{n-1}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Or
   :project: ngraph
   :members:
