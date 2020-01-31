.. not.rst:

###
Not
###

.. code-block:: cpp

   Not // Elementwise negation operation


Description
===========

Produces a single output tensor of boolean type and the same shape as ``arg,``
where the value at each coordinate of ``output`` is the negation of the
value at each ``arg`` coordinate.

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | ``element::boolean``    | Any                            |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | ``element::boolean``    | Same as ``arg``                |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0, \ldots, i_{n-1}} = \neg\mathtt{arg}_{i_0, \ldots, i_{n-1}}


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Not
   :project: ngraph
   :members:
