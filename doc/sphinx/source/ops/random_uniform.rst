.. random_uniform.rst:

#############
RandomUniform
#############

.. code-block:: cpp

   RandomUniform  // Operation that generates a tensor populated with random
                  // values of a uniform distribution.


Description
===========

.. warning:: This op is experimental and subject to change without notice.

Inputs
------

+--------------------+-------------------------+---------------------------------+-------------------------------------------+
| Name               | Element Type            | Shape                           | Notes                                     |
+====================+=========================+=============================================================================+
| ``min_value``      | Any floating point type | Scalar                          | Minimum value for the random distribution |
+--------------------+-------------------------+---------------------------------+-------------------------------------------+
| ``max_value``      | Same as ``max_value``   | Scalar                          | Maximum value for the random distribution |
+--------------------+-------------------------+---------------------------------+-------------------------------------------+
| ``result_shape``   | ``element::i64``        | Vector of any size              | Shape of the output tensor                |
+--------------------+-------------------------+---------------------------------+-------------------------------------------+
| ``use_fixed_seed`` | ``element::boolean``    | Scalar                          | Flag indicating whether to use the fixed  |
|                    |                         |                                 | seed value ``fixed_seed`` (useful for     |
|                    |                         |                                 | testing)                                  |
+--------------------+-------------------------+---------------------------------+-------------------------------------------+

Attributes
-----------

+---------------------+---------------+-----------------------------------------------------------------------------------------+
| Name                | Type          | Notes                                                                                   |
+=====================+===============+=========================================================================================+
| ``fixed_seed``      | ``uint64_t``  | Fixed seed value to use if ``use_fixed_seed`` flag is set to ``1``. This should be used |
|                     |               | only for testing; if ``use_fixed_seed`` is ``1``, ``RandomUniform`` will produce the    |
|                     |               | _same_ values at each iteration.                                                        |
+---------------------+---------------+-----------------------------------------------------------------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------------------+
| Name            | Element Type            | Shape                                      |
+=================+=========================+============================================+
| ``output``      | Same as ``min_value``   | ``result_shape``                           |
+-----------------+-------------------------+--------------------------------------------+


Mathematical Definition
=======================

.. math::

   \mathtt{output}_i = \mathtt{uniform_rand}(\mathtt{min}=\mathtt{min_value}, \mathtt{max}=\mathtt{max_value})


C++ Interface
=============

.. doxygenclass:: ngraph::op::RandomUniform
   :project: ngraph
   :members:
