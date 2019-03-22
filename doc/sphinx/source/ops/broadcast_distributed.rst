.. broadcastdistributed.rst:

#####################
BroadcastDistributed
#####################

.. code-block:: cpp

   BroadcastDistributed // Collective operation


Description
===========

Broadcast values from one process or device (root) to the rest processes or 
devices of the communicator.


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
| ``arg``         | ``element::f32``        | Same as ``arg``                |
|                 | ``element::f64``        |                                |
+-----------------+-------------------------+--------------------------------+


C++ Interface
=============

.. doxygenclass:: ngraph::op::BroadcastDistributed
   :project: ngraph
   :members:
