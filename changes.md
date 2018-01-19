# API Changes

## Negative convolution padding

`Convolution` now allows negative padding. This means that the `padding_below` and `padding_above`
arguments now take type `Padding` instead of `Shape`. `Padding` is an alias for
`std::vector<ssize_t>`, which "is like `size_t` but is allowed to be negative". Callers may need
to be adapted.

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

