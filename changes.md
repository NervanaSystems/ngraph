# API Changes

## Passes
* `LikeReplacement` pass must be run by all transformers.

## Nodes, Parameters

* `Nodes` is now `NodeVector`
* `Parameters` is now `ParameterVector`
* `NodeVector`, `ParameterVector`, `AxisVector`, `AxisSet`, `Shape`, `Stride`, `Coordinate`, and `CoordinateDiff` are now classes, not type aliases.
* `PrimaryTensorView` is now `TensorView` (and will merge into `Tensor`)

## Changes to ops

* The namespace `ngraph::op` is only for actual ops. Helpers have been moved into
  `ngraph::op::util`:
  + `BinaryElementwiseArithmetic`
  + `BinaryElementwiseComparison`
  + `BinaryElementwise`
  + `RequiresTensorViewArgs`
  + `UnaryElementwiseArithmetic`
  + `UnaryElementwise`
  Ops defined outside of nGraph core will need to get the base class from `ngraph::op::util` and
  change the include file to `#include "ngraph/ops/util/requires_tensor_view_args.hpp"`, etc.

  See any of the core ops for an example.

## Changes to convolution and pooling ops

* Backprop ops have been added for convolution ops.
* The convolution and pooling ops have had several methods/fields renamed, to reflect a shift
  in terminology from "images" to "data". Generally this just means that you will have to
  `s/image_batch/data_batch/` and `s/image_dilation_strides/data_dilation_strides/`.
* The following functions have been removed:
  + `AvgPool`: `get_channel_count get_input_image_physical_shape get_input_image_virtual_shape get_output_image_shape get_batch_size get_image_dimension_count`
  + `MaxPool`: `get_channel_count get_input_image_shape get_output_image_shape get_batch_size get_image_dimension_count`
  + `Convolution`: `get_input_channel_count get_output_channel_count get_input_image_physical_shape get_input_image_virtual_shape get_output_image_shape get_window_physical_shape get_window_virtual_shape get_batch_size get_image_dimension_count`

  All of the above information can be inferred from the shapes and parameters of the op.

* The `AvgPool` operator has a new attribute governing whether or not padding-region values
  are considered when computing a given window's average: `include_padding_in_avg_computation`.
  One of the class constructors adds this to the parameter list, and the others use a default
  value of `false` which matches the old behavior.

## Negative convolution padding

`Convolution` now allows negative padding. This means that the `padding_below` and `padding_above`
arguments now take type `CoordinateDiff` instead of `Shape`. `CoordinateDiff` is an alias for
`std::vector<std::ptrdiff_t>`, which "is like `size_t` but is allowed to be negative". Callers may
need to be adapted.

## `Parameter` and `Function` no longer take a type argument.

To update, remove the passed argument. For example,
```C++
// Old
make_shared<Parameter>(make_shared<descriptor::TensorViewType>(element::f32, Shape{2, 4}));
// New (remove TensorViewType)
make_shared<Parameter>(element::f32, Shape{2, 4});

// Old
make_shared<Function>(results, result_type, parameters);
// New
make_shared<Function>(results, parameters);
```

The runtime::TensorView methods to get_tensor<> and write<T>(std::vector&) have been removed
to the unit test directory under utils/test_tool.hpp read_vector and write_vector.
