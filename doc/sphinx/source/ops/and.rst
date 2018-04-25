.. and.rst:

###
And
###

.. code-block:: cpp

   And  // Elementwise logical-and operation


Description
===========

Produces tensor with boolean element type and shape as the two inputs,
which must themselves have boolean element type, where the value at each
coordinate of ``output`` is ``1`` (true) if ``arg0`` and ``arg1`` are
both nonzero, ``0`` otherwise.


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

   \texttt{output}_{i_0, \ldots, i_{n-1}} = \texttt{arg0}_{i_0, \ldots, i_{n-1}}\, \texttt{&&}\, \texttt{arg1}_{i_0, \ldots, i_{n-1}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::And
   :project: ngraph
   :members:
