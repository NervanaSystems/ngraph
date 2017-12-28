# API Changes
`Parameter` and `Function` no longer take a type argument. 
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

