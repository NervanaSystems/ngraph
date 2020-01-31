.. allreduce.rst:

#########
AllReduce
#########

.. code-block:: cpp

   AllReduce // Collective operation


Description
===========

Combines values from all processes or devices and distributes the result back
to all processes or devices.


Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | ``element::f32``        | Any                            |
|                 | ``element::f64``        |                                |
+-----------------+-------------------------+--------------------------------+


Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | ``element::f32``        | Same as ``arg``                |
|                 | ``element::f64``        |                                |
+-----------------+-------------------------+--------------------------------+


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::AllReduce
   :project: ngraph
   :members:
