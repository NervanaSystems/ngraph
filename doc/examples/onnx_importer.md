# ONNX Importer API usage examples

This sample demonstrates how to use the ONNX importer API.
This API makes it possible to create an nGraph `Function` object from an imported ONNX model.

All functions of the ONNX importer API are in the [onnx.hpp][onnx_header] header file.

API functions can be divided into two categories:
* helper functions used to check which ONNX ops are supported in current version of ONNX importer.
* function used to read ONNX models from stream or file. Their result is nGraph function which can be executed using the Inference Engine.

## How to check which ONNX ops are supported?

In order to list all supported ONNX ops in a specific version and domain, the `get_supported_operators` function should be used.
The code listing below shows how to do it.
```
const std::int64_t version = 12;
const std::string domain = "ai.onnx";
const std::set<std::string> supported_ops = ngraph::onnx_import::get_supported_operators(version, domain);

for(const auto& op : supported_ops)
{
    std::cout << op << std::endl;
}
```
The above code compiles to produce a list of all the supported operators for the `version` and `domain` you selected. The output should be a list similar to this:
```
Abs
Acos
...
Xor
```

To determine whether a specific ONNX operator in a particular version and domain is supported by the importer, use the function `is_operator_supported`.
The code listing below shows how to do it.
```
const std::string op_name = "Abs";
const std::int64_t version = 12;
const std::string domain = "ai.onnx";
const bool is_abs_op_supported = ngraph::onnx_import::is_operator_supported(op_name, version, domain);

std::cout << "Abs in version 12, domain `ai.onnx`is supported: " << (is_abs_op_supported ? "true" : "false") << std::endl;
```

## How to import ONNX model
In order to import an ONNX model, the `import_onnx_model` function should be used.
The method has two overloads.
The first uses stream as input (e.g. file stream, memory stream).
The code listing below shows how to convert `ResNet50` ONNX model to nGraph function.

> **NOTE** `ResNet50` ONNX model can be downloaded from the [ONNX model zoo][onnx_model_zoo].
```
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
$ tar -xzvf resnet50.tar.gz
```

```
 const std::string resnet50_path = "resnet50/model.onnx";
 std::ifstream resnet50_stream(resnet50_path);
 if(resnet50_stream.is_open())
 {
     try
     {
         const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_stream);

         // Let's check for example shape of first output
         std::cout << ng_function->get_output_shape(0) << std::endl;
         // The output is Shape{1, 1000}
     }
     catch (const ngraph::ngraph_error& error)
     {
         std::cout << "Error during import ONNX model: " << error.what() << std::endl;
     }
 }
 resnet50_stream.close();
```

The second overload of `import_onnx_model` uses file path as argument.
Using this version, the code can be simplified to:
```
const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_path);
```

If `ng_function` is created, it can be used for running computation on Inference Engine.
As it was shown in [Build a Model with nGraph Library][build_ngraph] `std::shared_ptr<ngraph::Function>` can be transformed into a `CNNNetwork`.

[onnx_header]: https://github.com/NervanaSystems/ngraph/blob/master/src/ngraph/frontend/onnx_import/onnx.hpp
[onnx_model_zoo]: https://github.com/onnx/models
[build_ngraph]: https://docs.openvinotoolkit.org/latest/_docs_IE_DG_nGraphTutorial.html
