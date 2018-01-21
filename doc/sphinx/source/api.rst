.. api.rst:


This API documentation covers the public API for 
Intel® nGraph™ library (``ngraph``), organized into 
three main modules:

- ``ngraph``: Contains the core ops for constructing the graph.
- ``ngraph.transformers``: Defines methods for executing a defined graph on hardware.
- ``ngraph.types``: Types in ngraph (for example, ``Axes``, ``Op``, etc.)

Intel Nervana (ngraph) API 
==========================

Several ops are used to create different types of tensors:

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`ngraph.variable` | Create a trainable variable.
	:meth:`ngraph.persistent_tensor` | Tensor that persists across computations.
	:meth:`ngraph.placeholder` | Used for input values, typically from host.
	:meth:`ngraph.constant` | Immutable constant that can be inlined.

Assigning the above tensors requires defining ``Axis``, which can be done using the following methods:

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`ngraph.axes_with_order` | Return a tensor with a different axes order.
	:meth:`ngraph.cast_axes` | Cast the axes of a tensor to new axes.
	:meth:`ngraph.make_axes` | Create an Axes object.
	:meth:`ngraph.make_axis` | Create an Axis.

We also provide several helper function for retrieving information from tensors.

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

	:meth:`ngraph.batch_size` | Returns the batch size
	:meth:`ngraph.is_constant` | Returns true if tensor is constant
	:meth:`ngraph.is_constant_scalar` | Returns true if tensor is a constant scalar
	:meth:`ngraph.constant_value` | Returns the value of a constant tensor
	:meth:`ngraph.tensor_size` | Returns the total size of the tensor

To compose a computational graph, we support the following operations:

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`ngraph.absolute` | :math:`\operatorname{abs}(a)`
    :meth:`ngraph.negative` | :math:`-a`
	:meth:`ngraph.sign` | if :math:`x<0`, :math:`-1`; if :math:`x=0`, :math:`0`; if :math:`x>0`, :math:`1`
	:meth:`ngraph.add` | :math:`a+b`
	:meth:`ngraph.reciprocal` | :math:`1/a`
	:meth:`ngraph.square` | :math:`a^2`
	:meth:`ngraph.sqrt` | :math:`\sqrt{a}`
	:meth:`ngraph.cos` | :math:`\cos(a)`
	:meth:`ngraph.sin` | :math:`\sin(a)`
	:meth:`ngraph.tanh` | :math:`\tanh(a)`
	:meth:`ngraph.sigmoid` | :math:`1/(1+\exp(-a))`
	:meth:`ngraph.exp` | :math:`\exp(a)`
	:meth:`ngraph.log` | :math:`\log(a)`
	:meth:`ngraph.safelog` | :math:`\log(a)`
	:meth:`ngraph.one_hot` | Convert to one-hot
	:meth:`ngraph.variance` | Compute variance
	:meth:`ngraph.stack` | Stack tensors along an axis
	:meth:`ngraph.convolution` | Convolution operation
	:meth:`ngraph.pad` | Pad a tensor with zeros along each dimension
	:meth:`ngraph.pooling` | Pooling operation
	:meth:`ngraph.squared_L2` | dot x with itself


.. Note::
   Additional operations are supported that are not currently documented, and so are not included in the list above. We will continue to populate this API when the documentation is updated.

ngraph.transformers
===================

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`ngraph.transformers.allocate_transformer` | Allocate a transformer.
    :meth:`ngraph.transformers.make_transformer` | Generates a transformer using the factory.
    :meth:`ngraph.transformers.make_transformer_factory` | Creates a new factory with cpu default.
    :meth:`ngraph.transformers.set_transformer_factory` | Sets the Transformer factory used by make_transformer.
    :meth:`ngraph.transformers.transformer_choices` | Return the list of available transformers.
    :meth:`ngraph.transformers.Transformer` | Produce an executable version of op-graphs.

ngraph.types
============

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`ngraph.types.AssignableTensorOp` | Assign a tensor. Used by `ng.placeholder`, and more.
    :meth:`ngraph.types.Axis` | An Axis labels a dimension of a tensor.
    :meth:`ngraph.types.Axes` | Axes represent multiple axis dimensions.
    :meth:`ngraph.types.Computation` | Computations to attach to transformers.
    :meth:`ngraph.types.NameableValue` | Objects that can derive name from the name scope.
    :meth:`ngraph.types.NameScope` | Name scope for objects.
    :meth:`ngraph.types.Op` | Basic class for ops.
    :meth:`ngraph.types.TensorOp` | Base class for ops related to Tensors.

ngraph
------

Graph construction.

.. py:module: ngraph

.. automodule:: ngraph
   :members:

ngraph.transformers
-------------------

Transformer manipulation.

.. py:module: ngraph.transformers

.. automodule:: ngraph.transformers
   :members:

ngraph.types
------------

.. py:module: ngraph.types

.. automodule:: ngraph.types
   :members:
