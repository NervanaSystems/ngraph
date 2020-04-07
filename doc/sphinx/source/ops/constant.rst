.. constant.rst:

########
Constant
########

.. code-block:: cpp

   Constant // Literal constant tensor


Description
===========

The output is a tensor initialized from the ``values`` attribute.

Attributes
----------

+-----------------+------------------------------+---------------------------------------+
| Name            | Type                         | Notes                                 |
+=================+==============================+=======================================+
| ``type``        | ``ngraph::element::type``    | The element type of the value         |
|                 |                              | in the computation                    |
+-----------------+------------------------------+---------------------------------------+
| ``shape``       | ``ngraph::Shape``            | The shape of the constant             |
+-----------------+------------------------------+---------------------------------------+
| ``values``      | ``const std::vector<T>&``    | Constant elements in row-major order. |
|                 |                              | T must be compatible with the element |
|                 |                              | type                                  |
+-----------------+------------------------------+---------------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | ``type``                | ``shape``                      |
+-----------------+-------------------------+--------------------------------+


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Constant
   :project: ngraph
   :members:
