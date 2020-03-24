.. transpose.rst:

#########
Transpose
#########

.. code-block:: cpp

   Transpose  // Operation that transposes axes of a tensor

Description
===========

.. warning:: This op is not yet implemented in any backend.

.. warning:: This op is experimental and subject to change without notice.

Operation that transposes axes of an input tensor. This operation covers
matrix transposition, and also more general cases on higher-rank tensors.

Inputs
------

+-----------------+-------------------------+---------------------------------------------+
| Name            | Element Type            | Shape                                       |
+=================+=========================+=============================================+
| ``arg``         | Any                     | Any                                         |
+-----------------+-------------------------+---------------------------------------------+
| ``input_order`` | ``element::i64``        | ``[n]``, where `n`` is the rank of ``arg``. |
+-----------------+-------------------------+---------------------------------------------+

Outputs
-------

+-----------------+-------------------------+-------------------------------------------------------------------------------+
| Name            | Element Type            | Shape                                                                         |
+=================+=========================+===============================================================================+
| ``output``      | Same as ``arg``         | ``P(ShapeOf(arg))``, where `P` is the permutation supplied for `input_order`. |
+-----------------+-------------------------+-------------------------------------------------------------------------------+

The input ``input_order`` must be a vector of shape `[n]`, where `n` is the
rank of ``arg``, and must contain every integer in the range ``[0,n-1]``. This
vector represents a permutation of ``arg``'s dimensions. For example,

+---------------+-----------------------+------------------+-----------------------------------------------------+
| ``arg`` Shape | ``input_order`` Value | ``output`` Shape | Comment                                             |
+===============+=======================+==================+=====================================================+
| ``[3,4]``     | ``[1,0]``             | ``[4,3]``        | Transposes the ``arg`` matrix.                      |
+---------------+-----------------------+------------------+-----------------------------------------------------+
| ``[3,3]``     | ``[1,0]``             | ``[3,3]``        | Transposes the ``arg`` matrix.                      |
+---------------+-----------------------+------------------+-----------------------------------------------------+
| ``[3,3]``     | ``[1,0]``             | ``[3,3]``        | Transposes the ``arg`` matrix.                      |
+---------------+-----------------------+------------------+-----------------------------------------------------+
| ``[3,4,8]``   | ``[2,0,1]``           | ``[8,3,4]``      | Moves the "last" dimension to the "first" position. |
+---------------+-----------------------+------------------+-----------------------------------------------------+

Mathematical Definition
=======================

.. math::

   \mathtt{output}_{i_0,i_1,...,i_n} = \mathtt{arg}_{i_{\mathtt{input_order}[0]},i_\mathtt{input_order}[1],...,i_\mathtt{input_order}[n]}.

Backprop
========

Not yet implemented.

C++ Interface
=============

.. doxygenclass:: ngraph::op::v0::Transpose
   :project: ngraph
   :members:
