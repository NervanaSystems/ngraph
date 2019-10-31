//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

static std::mt19937_64 random_generator;

using namespace std;
using namespace ngraph;

// tool to import and run an ONNX model
void help()
{
    cout << R"###(
DESCRIPTION
    Run an ONNX model

SYNOPSIS
        run_onnx_model -m|--model <model file> [-i|--input <input file>]  [-b|--backend <backend name>]

OPTIONS
        -m or --model    Path to ONNX protobuf file with extension .onnx or .prototext  
        -i or --input    Path to a raw binary file with an array of input data. If not provided, model will be executed with random data.
        -b or --backend  nGraph backend name, such as INTERPRETER, CPU, GPU, NNP, PlaidML, INTELGPU, where available. Default backend: CPU

)###";
}

// Load raw binary data from a file to  dynamically allocated memory buffer.
unique_ptr<char[]> load_data_from_file(const string& file_name)
{
    ifstream file(file_name);
    file.seekg(0, ios::end);
    size_t len = file.tellg();
    unique_ptr<char[]> data(new char[len]);
    file.seekg(0, ios::beg);
    file.read(data.get(), len);
    file.close();

    return data;
}

// Prepare input tensors for given model and load them with data from binary files.
vector<shared_ptr<runtime::Tensor>> load_inputs(std::shared_ptr<runtime::Backend> backend,
                                                std::shared_ptr<ngraph::Function> function,
                                                const vector<string>& input_paths)
{
    vector<shared_ptr<runtime::Tensor>> inputs{};
    auto params = function->get_parameters();

    for (int i = 0; i < input_paths.size(); i++)
    {
        cout << "Input " << i << " file path:" << input_paths.at(i) << endl;
        auto tensor =
            backend->create_tensor(params.at(i)->get_element_type(), params.at(i)->get_shape());
        const string input_path = input_paths.at(i);
        const size_t data_size =
            shape_size(tensor->get_shape()) * tensor->get_element_type().size();
        unique_ptr<char[]> data = load_data_from_file(input_paths.at(i));
        tensor->write(data.get(), data_size);
        inputs.push_back(tensor);
    }

    return inputs;
}

// Prepare input tensors for given model and load them with random generated data.
vector<shared_ptr<runtime::Tensor>> random_inputs(std::shared_ptr<runtime::Backend> backend,
                                                  std::shared_ptr<ngraph::Function> function)
{
    vector<shared_ptr<runtime::Tensor>> inputs{};
    std::uniform_int_distribution<int> distribution(0, 255);
    auto params = function->get_parameters();

    for (int i = 0; i < params.size(); i++)
    {
        auto tensor =
            backend->create_tensor(params.at(i)->get_element_type(), params.at(i)->get_shape());
        const size_t data_size =
            shape_size(tensor->get_shape()) * tensor->get_element_type().size();
        unique_ptr<char[]> data(new char[data_size]());
        for (int i = 0; i < data_size; i++)
        {
            data[i] = distribution(random_generator);
        }
        tensor->write(data.get(), data_size);
        inputs.push_back(tensor);
    }
    return inputs;
}

// Prepare output tensors for given model.
vector<shared_ptr<runtime::Tensor>> make_outputs(std::shared_ptr<runtime::Backend> backend,
                                                 std::shared_ptr<ngraph::Function> function)
{
    vector<shared_ptr<runtime::Tensor>> outputs{};
    const size_t outputs_size = function->get_results().size();
    for (int i = 0; i < outputs_size; i++)
    {
        auto tensor = backend->create_tensor(function->get_output_element_type(i),
                                             function->get_output_shape(i));
        outputs.push_back(tensor);
    }
    return outputs;
}

// Validate provided arguments and split them into options and values.
std::tuple<string, string> validate_argument(string arg, string arg2)
{
    size_t index = arg.find_first_of('=');
    string option = arg;
    string value = arg2;
    if (index < string::npos)
    {
        option = arg.substr(0, index);
        value = arg.substr(index + 1, arg.size() - 1);
    }
    return std::make_tuple(option, value);
}

// Print vector as tensor with given shape.
void print_vector(const vector<float>& data,
                  const vector<size_t>& shape,
                  size_t shape_pointer,
                  size_t data_pointer)
{
    cout << "[ ";
    if (shape_pointer == shape.size() - 2)
    {
        const size_t pointed_dim = shape.at(shape_pointer);
        const size_t next_dim = shape.at(shape_pointer + 1);
        for (size_t i = 0; i < pointed_dim; i++)
        {
            cout << "[";
            for (int j = 0; j < next_dim; j++)
            {
                cout << data.at(data_pointer + (i * next_dim) + j);
                if (j != next_dim - 1)
                    cout << ", ";
            }

            if (i == pointed_dim - 1)
            {
                cout << "]";
            }
            else
            {
                cout << "]" << endl;
            }
        }
    }
    else
    {
        size_t data_offset = 1;
        const size_t pointed_dim = shape.at(shape_pointer);
        for (size_t i = shape_pointer + 1; i < shape.size(); i++)
        {
            data_offset *= shape.at(i);
        }
        size_t next_data_pointer;
        for (size_t k = 0; k < pointed_dim; k++)
        {
            next_data_pointer = data_pointer + (k * data_offset);
            if (next_data_pointer != 0)
            {
                cout << endl;
            }

            print_vector(data, shape, shape_pointer + 1, next_data_pointer);
        }
    }

    if (shape_pointer == 0)
    {
        cout << "]" << endl;
    }
    else
    {
        cout << "]";
    }
}

// Print model outputs' metadata and data.
void print_outputs(const vector<shared_ptr<runtime::Tensor>>& outputs)
{
    cout << "Outputs info:" << endl;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        shared_ptr<runtime::Tensor> output = outputs.at(i);
        cout << "Output " << i << endl;
        cout << output->get_shape() << endl;
        cout << output->get_element_type() << endl;
        cout << "Count of elements:" << output->get_element_count() << endl;
        vector<float> output_values = read_float_vector(output);
        print_vector(output_values, output->get_shape(), 0, 0);
    }
}

int main(int argc, char** argv)
{
    vector<string> input_paths{};
    string model;
    string backend_type = "CPU";
    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    std::shared_ptr<ngraph::Function> function;
    std::shared_ptr<runtime::Backend> backend;

    for (size_t i = 1; i < argc; i++)
    {
        string arg = argv[i];
        string arg2 = "";
        tie(arg, arg2) = validate_argument(arg, arg2);

        if (arg2 == "")
        {
            arg2 = argv[++i];
        }

        if (arg == "-m" || arg == "--model")
        {
            model = arg2;
        }
        else if (arg == "-b" || arg == "--backend")
        {
            backend_type = arg2;
        }
        else if (arg == "-i" || arg == "--input")
        {
            input_paths.push_back(arg2);
        }
        else if (arg == "-h" || arg == "--help")
        {
            help();
            return 0;
        }
    }

    ifstream f(model);
    if (f)
    {
        function = ngraph::onnx_import::import_onnx_model(model);
        cout << "Model file path:" << model << endl;
    }
    else
    {
        cout << "Failed to open '" << model << "' for model\n";
        return 1;
    }

    try
    {
        backend = ngraph::runtime::Backend::create(backend_type);
    }
    catch (runtime_error e)
    {
        cout << "Backend " << backend_type << " not supported." << endl;
        return 2;
    }

    const size_t inputs_size = function->get_parameters().size();
    if (input_paths.size() == inputs_size)
    {
        inputs = load_inputs(backend, function, input_paths);
    }
    else if (input_paths.size() == 0)
    {
        inputs = random_inputs(backend, function);
    }
    else
    {
        cout << "Inappropriate number of input files." << endl;
        return 2;
    }

    outputs = make_outputs(backend, function);

    if (function)
    {
        auto handle = backend->compile(function);

        if (handle->call_with_validate(outputs, inputs))
        {
            print_outputs(outputs);
        }
        else
        {
            cout << "FAILED" << endl;
        }
    }

    return 0;
}
