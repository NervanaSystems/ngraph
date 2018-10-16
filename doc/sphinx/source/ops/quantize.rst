.. quantize.rst: 

########
Quantize
########

.. code-block:: cpp

   Quantize // Maps real input to quantized output using scale, offset and round mode

Description
===========

Produces a tensor of element type ``type`` and the same shape as ``input``
where the value of each coordinate (i) of ``output`` is the corresponding coordinate of 
``input`` divided by ``scale`` rounded as specified by ``round_mode`` minus ``offset``.  
The coordinate (j) of ``scale`` and ``offset`` is the coordinate of ``output``
projected along ``axes``.

Inputs
------

+-----------------+-------------------------+---------------------------------------+
| Name            | Element Type            | Shape                                 |
+=================+=========================+=======================================+
| ``input``       | is_real()               | Any                                   |
+-----------------+-------------------------+---------------------------------------+
| ``scale``       | Same as ``input``       | ``input`` shape projected on ``axes`` |
+-----------------+-------------------------+---------------------------------------+
| ``offset``      | Same as ``output``      | ``input`` shape projected on ``axes`` |
+-----------------+-------------------------+---------------------------------------+

Attributes
----------

+-------------------------------+----------------------------------------------------------------+
| Name                          | Description                                                    |
+===============================+================================================================+
| ``type``                      | ``output`` element type                                        |
+-------------------------------+----------------------------------------------------------------+
| ``axes``                      | Axis positions on which ``scale`` and ``offset`` are specified |
+-------------------------------+----------------------------------------------------------------+
| ``round_mode``                | See src/ngraph/op/quantize.hpp                                 |
+-------------------------------+----------------------------------------------------------------+

Outputs
-------

+-----------------+-------------------------+---------------------------------------+
| Name            | Element Type            | Shape                                 |
+=================+=========================+=======================================+
| ``output``      | is_quantized()          | Same as ``input``                     |
+-----------------+-------------------------+---------------------------------------+

Mathematical Definition
=======================

.. math::
    $\mathtt{output}_{i} = round(\frac{\mathtt{input}_{i}}{\mathtt_{scale}_{j}}}) - \mathtt_{offset}_{j}$

C++ Interface
=============

.. doxygenclass:: ngraph::op::Quantize
   :project: ngraph
   :members: m_type, m_axes, m_round_mode