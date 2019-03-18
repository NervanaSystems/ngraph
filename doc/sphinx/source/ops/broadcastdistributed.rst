.. broadcastdistributed.rst:

#########
BroadcastDistributed
#########

.. code-block:: cpp

   BroadcastDistributed // Collective operation


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


Outputs (in place)
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``      | ``element::f32``        | Same as ``arg``                |
|                 | ``element::f64``        |                                |
+-----------------+-------------------------+--------------------------------+


C++ Interface
=============

.. doxygenclass:: ngraph::op::BroadcastDistributed
   :project: ngraph
   :members:
