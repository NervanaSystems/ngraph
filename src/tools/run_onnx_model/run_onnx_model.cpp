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
#include <string>

#include "../../../test/util/test_tools.hpp"
#include "gtest/gtest.h"
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
        run_onnx_model [-m|--model <model file>] [-i|--input <input file>] [-m|--model <model file>] [-b|--backend <backend name>]

OPTIONS
        -i or --input  input binary file
        -b or --backend  backend name

)###";
}

int load_inputs(vector<string> input_paths, vector<shared_ptr<runtime::Tensor>> inputs)
{
    for (int i = 0; i < input_paths.size(); i++)
    {
        const string input_path = input_paths.at(i);
        std::vector<float> data = read_binary_file<float>(input_path);
        copy_data(inputs.at(i), data);
    }

    return 0;
}

int random_inputs(vector<shared_ptr<runtime::Tensor>> inputs)
{
    for (int i = 0; i < inputs.size(); i++)
    {
        auto tensor_size = shape_size(inputs.at(i)->get_shape());
        std::uniform_int_distribution<int> distribution(0, 255);
        vector<float> data(tensor_size, 0);
        for (int i = 0; i < tensor_size; i++)
        {
            data[i] = distribution(random_generator);
        }
        copy_data(inputs.at(i), data);
    }
    return 0;
}

int main(int argc, char** argv)
{
    vector<string> input_paths{};
    string model;
    string backend_type = "CPU";
    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    std::shared_ptr<ngraph::Function> function;

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-m" || arg == "--model")
        {
            model = argv[++i];
        }
        else if (arg == "-b" || arg == "--backend")
        {
            backend_type = argv[++i];
        }
        else if (arg == "-i" || arg == "--input")
        {
            input_paths.push_back(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            help();
            return 0;
        }
    }

    auto backend = ngraph::runtime::Backend::create(backend_type);
    ifstream f(model);
    if (f)
    {
        function = ngraph::onnx_import::import_onnx_model(model);

        // Creating inputs
        auto params = function->get_parameters();
        for (int i = 0; i < params.size(); i++)
        {
            auto tensor =
                backend->create_tensor(params.at(i)->get_element_type(), params.at(i)->get_shape());
            inputs.push_back(tensor);
        }

        auto outputs_size = function->get_results().size();
        for (int i = 0; i < outputs_size; i++)
        {
            auto tensor = backend->create_tensor(function->get_output_element_type(i),
                                                 function->get_output_shape(i));
            outputs.push_back(tensor);
        }
    }
    else
    {
        cout << "Failed to open '" << model << "' for model\n";
        return 1;
    }

    if (input_paths.size() == inputs.size())
    {
        load_inputs(input_paths, inputs);
    }
    else if (input_paths.size() == 0)
    {
        random_inputs(inputs);
    }
    else
    {
        cout << "Inappropriate number of input files." << endl;
        return 2;
    }

    if (function)
    {
        auto handle = backend->compile(function);

        if (handle->call_with_validate(outputs, inputs))
        {
            cout << "PASSED" << endl;
        }
        else
        {
            cout << "FAILED" << endl;
        }
    }

    return 0;
}
