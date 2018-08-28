.. equal.rst:

#####
Equal
#####

.. code-block:: cpp

   Equal  // Elementwise equal operation


Description
===========

Produces tensor of the same element type and shape as the two inputs,
where the value at each coordinate of ``output`` is ``1`` (true) if
``arg0`` is equal to ``arg1``, ``0`` otherwise.


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

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \mathtt{arg0}_{i_0, \ldots, i_{n-1}} == \mathtt{arg1}_{i_0, \ldots, i_{n-1}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Equal
   :project: ngraph
   :members:
