# API Changes

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
