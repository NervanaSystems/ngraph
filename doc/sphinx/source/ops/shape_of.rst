.. shape_of.rst:

#######
ShapeOf
#######

.. code-block:: cpp

   ShapeOf  // Operation that returns the shape of its input tensor


Description
===========

.. warning:: This op is experimental and subject to change without notice.

Returns the shape of its input argument as a tensor of element type ``u64``.

Inputs
------

+-----------------+-------------------------+---------------------------------+
| Name            | Element Type            | Shape                           |
+=================+=========================+=================================+
| ``arg``         | Any                     | Any                             |
+-----------------+-------------------------+---------------------------------+

Outputs
-------

+-----------------+-------------------------+-----------------------------------------------------+
| Name            | Element Type            | Shape                                               |
+=================+=========================+=====================================================+
| ``output``      | ``element::u64``        | ``{r}`` where ``r`` is the rank of ``arg``'s shape. |
+-----------------+-------------------------+-----------------------------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output} = \mathtt{shapeof}(\mathtt{arg})


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::ShapeOf
   :project: ngraph
   :members:
