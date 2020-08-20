.. dequantize.rst: 

##########
Dequantize
##########

.. code-block:: cpp

   Dequantize // Maps quantized input to real output

Description
===========

Produces a tensor of element type ``type`` and the same shape as ``input`` 
where the value of each coordinate :math:`i` of ``output`` is the corresponding coordinate of 
``input`` minus ``zero_point`` quantity multiplied by ``scale``.
The coordinate :math:`j` of ``scale`` and ``zero_point`` is the coordinate of ``output`` 
projected onto ``axes``.

Inputs
------

+-----------------+-------------------------+----------------------------------------------+
| Name            | Element Type            | Shape                                        |
+=================+=========================+==============================================+
| ``input``       | Any quantized type      | Any                                          |
+-----------------+-------------------------+----------------------------------------------+
| ``scale``       | Same as ``output``      | ``input`` shape projected onto ``axes``      |
+-----------------+-------------------------+----------------------------------------------+
| ``zero_point``      | Same as ``input``       | ``input`` shape projected onto ``axes``  |
+-----------------+-------------------------+----------------------------------------------+

Attributes
----------

+-------------------------------+--------------------------------------------------------------------+
| Name                          | Description                                                        |
+===============================+====================================================================+
| ``type``                      | ``output`` element type; any real type                             |
+-------------------------------+--------------------------------------------------------------------+
| ``axes``                      | Axis positions on which ``scale`` and ``zero_point`` are specified |
+-------------------------------+--------------------------------------------------------------------+





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

   \mathtt{output}_{i,j} = (\mathtt{input}_{i,j} - \mathtt{zero_point}_{j}) \mathtt{scale}_{j}

C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Dequantize
   :project: ngraph
   :members: 
