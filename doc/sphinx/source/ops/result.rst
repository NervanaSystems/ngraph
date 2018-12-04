.. result.rst:

######
Result
######

.. code-block:: cpp

   Result  // Allow a value to be a result


Description
===========

Captures a value for use as a function result. The output of the
op is the same as the input.

Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``arg``         | Same as ``arg``                |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output} = \mathtt{arg}


C++ Interface
=============

.. doxygenclass:: ngraph::op::Result
   :project: ngraph
   :members:
