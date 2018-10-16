.. dequantize.rst: 

##########
Dequantize
##########

.. code-block:: cpp

   Dequantize // Maps quantized input to real output using scale and offset

Description
===========

Produces a tensor of element type ``type`` and the same shape as ``input``
where the value of each coordinate (i) of ``output`` is the corresponding coordinate of 
``input`` plus ``offset`` quantity multiplied by ``scale``.  
The coordinate (j) of ``scale`` and ``offset`` is the coordinate of ``output``
projected along ``axes``.

Inputs
------

+-----------------+-------------------------+---------------------------------------+
| Name            | Element Type            | Shape                                 |
+=================+=========================+=======================================+
| ``input``       | is_quantized()          | Any                                   |
+-----------------+-------------------------+---------------------------------------+
| ``scale``       | Same as ``output``      | ``input`` shape projected on ``axes`` |
+-----------------+-------------------------+---------------------------------------+
| ``offset``      | Same as ``input``       | ``input`` shape projected on ``axes`` |
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



Outputs
-------

+-----------------+-------------------------+---------------------------------------+
| Name            | Element Type            | Shape                                 |
+=================+=========================+=======================================+
| ``output``      | is_real()               | Same as ``input``                     |
+-----------------+-------------------------+---------------------------------------+

Mathematical Definition
=======================

.. math::
    $\mathtt{output}_{i} = (\mathtt{input}_{i}} + \mathtt_{offset}_{j}) \mathtt_{scale}_{j}$

C++ Interface
=============

.. doxygenclass:: ngraph::op::Dequantize
   :project: ngraph
   :members: m_type, m_axes