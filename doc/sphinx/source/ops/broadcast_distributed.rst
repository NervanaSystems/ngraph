.. broadcastdistributed.rst:

#####################
BroadcastDistributed
#####################

.. code-block:: cpp

   BroadcastDistributed // Collective operation


Description
===========

Broadcast values from a primary root process or device to other processes or 
devices within the op communicator.


Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | ``element::f32``        | Any                            |
|                 | ``element::f64``        |                                |
+-----------------+-------------------------+--------------------------------+


Outputs (in place)
------------------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | ``element::f32``        | Same as ``arg``                |
|                 | ``element::f64``        |                                |
+-----------------+-------------------------+--------------------------------+


C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::BroadcastDistributed
   :project: ngraph
   :members:
