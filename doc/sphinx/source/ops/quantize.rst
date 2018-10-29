.. quantize.rst: 

########
Quantize
########

.. code-block:: cpp

   Quantize // Maps real input to quantized output

Description
===========

Produces a tensor of element type ``type`` and the same shape as ``input`` 
where the value of each coordinate :math:`i` of ``output`` is the corresponding coordinate of 
``input`` divided by ``scale`` rounded as specified by ``round_mode`` plus ``offset``.
The coordinate :math:`j` of ``scale`` and ``offset`` is the coordinate of ``output`` 
projected onto ``axes``.

Inputs
------

+-----------------+-------------------------+------------------------------------------+
| Name            | Element Type            | Shape                                    |
+=================+=========================+==========================================+
| ``input``       | Any real type           | Any                                      |
+-----------------+-------------------------+------------------------------------------+
| ``scale``       | Same as ``input``       | ``input`` shape projected onto ``axes``  |
+-----------------+-------------------------+------------------------------------------+
| ``offset``      | Same as ``output``      | ``input`` shape projected onto ``axes``  |
+-----------------+-------------------------+------------------------------------------+

Attributes
----------

+-------------------------------+----------------------------------------------------------------+
| Name                          | Description                                                    |
+===============================+================================================================+
| ``type``                      | ``output`` element type; any quantized type                    |
+-------------------------------+----------------------------------------------------------------+
| ``axes``                      | Axis positions on which ``scale`` and ``offset`` are specified |
+-------------------------------+----------------------------------------------------------------+
| ``round_mode``                | *ROUND_NEAREST_TOWARD_INFINITY:*                               |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers round away from zero e.g.  |
|                               | 2.5 -> 3                                                       |
|                               | -3.5 -> -4                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_TOWARD_ZERO:*                                   |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers round toward zero e.g.     |
|                               | 2.5 -> 2                                                       |
|                               | -3.5 to -3                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_UPWARD:*                                        |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers round up e.g.              |
|                               | 2.5 to 3                                                       |
|                               | -3.5 to -3                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_DOWNWARD:*                                      |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers round down e.g.            |
|                               | 2.5 to 2                                                       |
|                               | -3.5 to -4                                                     |
|                               |                                                                |
|                               | *ROUND_NEAREST_TOWARD_EVEN:*                                   |
|                               | round to nearest integer                                       |
|                               | in case of two equidistant integers round to even e.g.         |
|                               | 2.5 to 2                                                       |
|                               | -3.5 to -4                                                     |
|                               |                                                                |
|                               | *ROUND_TOWARD_INFINITY:*                                       |
|                               | round to nearest integer away from zero                        |
|                               |                                                                |
|                               | *ROUND_TOWARD_ZERO:*                                           |
|                               | round to nearest integer toward zero                           |
|                               |                                                                |
|                               | *ROUND_UP:*                                                    |
|                               | round to nearest integer toward infinity (ceiling)             |
|                               |                                                                |
|                               | *ROUND_DOWN:*                                                  |
|                               | round to nearest integer toward negative infinity (floor)      |
+-------------------------------+----------------------------------------------------------------+

Outputs
-------

+-----------------+-------------------------+---------------------------------------+
| Name            | Element Type            | Shape                                 |
+=================+=========================+=======================================+
| ``output``      | ``type``                | Same as ``input``                     |
+-----------------+-------------------------+---------------------------------------+

Mathematical Definition
=======================

.. math::
  
   \mathtt{output}_{i,j} = \mathtt{round}\left(\frac{\mathtt{input}_{i,j}}{\mathtt{scale}_{j}}\right) + \mathtt{offset}_{j}    

C++ Interface
=============

.. doxygenclass:: ngraph::op::Quantize
   :project: ngraph
   :members: 
