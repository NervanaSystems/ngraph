.. quantize.rst: 

########
Quantize
########

.. code-block:: cpp

   Quantize // Maps real input to quantized output

Description
===========

Produces a tensor of element type ``type`` and the same shape as ``input``
where the value of each coordinate :math:`i` of ``output`` is the corresponding 
coordinate of ``input`` divided by ``scale`` rounded as specified by 
``round_mode`` minus ``offset``. The coordinate :math:`j` of ``scale`` and 
``offset`` is the coordinate of ``output`` projected onto ``axes``.

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
| ``type``                      | The output element type, which must be a quantized type        |
+-------------------------------+----------------------------------------------------------------+
| ``axes``                      | Axis positions on which ``scale`` and ``offset`` are specified |
+-------------------------------+----------------------------------------------------------------+
| ``round_mode``                | Refer to ``/src/ngraph/op/quantize.hpp``                       |
+-------------------------------+----------------------------------------------------------------+



Outputs
-------

+-----------------+-------------------------+---------------------------------------+
| Name            | Element Type            | Shape                                 |
+=================+=========================+=======================================+
| ``output``      | type                    | Same as ``input``                     |
+-----------------+-------------------------+---------------------------------------+

Mathematical Definition
=======================

.. math::
  
   \mathtt{output}_{i} = \mathtt{round}(\frac{\mathtt{input}_{i}}{\mathtt{scale}_{j}}) - \mathtt{offset}_{j}    



C++ Interface
=============

.. doxygenclass:: ngraph::op::Quantize
   :project: ngraph
   :members: 
