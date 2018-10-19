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
|                               | x.5 to x+1                                                     |
|                               | -x.5 to -(x+1)                                                 |
|                               | everything else to nearest integer                             |
|                               |                                                                |
|                               | *ROUND_NEAREST_TOWARD_ZERO:*                                   |
|                               | x.5 to x-1                                                     |
|                               | -x.5 to -(x-1)                                                 |
|                               | everything else to nearest integer                             |
|                               |                                                                |
|                               | *ROUND_NEAREST_UPWARD:*                                        |
|                               | x.5 to x+1                                                     |
|                               | -x.5 to -x                                                     |
|                               | everything else to nearest integer                             |
|                               |                                                                |
|                               | *ROUND_NEAREST_DOWNWARD:*                                      |
|                               | x.5 to x                                                       |
|                               | -x.5 to -(x+1)                                                 |
|                               | everything else to nearest integer                             |
|                               |                                                                |
|                               | *ROUND_NEAREST_TOWARD_EVEN:*                                   |
|                               | x.5 and -x.5 to nearest even ineger                            |
|                               | everything else to nearest integer                             |
|                               |                                                                |
|                               | *ROUND_TOWARD_INFINITY:*                                       |
|                               | everything to next integer away from zero                      |
|                               |                                                                |
|                               | *ROUND_TOWARD_ZERO:*                                           |
|                               | everything to next integer towards zero                        |
|                               |                                                                |
|                               | *ROUND_UP:*                                                    |
|                               | everything to next integer towards infinity                    |
|                               |                                                                |
|                               | *ROUND_DOWN:*                                                  |
|                               | everything to next integer towards negative infinity           |
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
