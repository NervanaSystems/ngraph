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

// tool to import and run an ONNX model

#include <fstream>
#include <iostream>
#include <string>

#include "../../../test/util/test_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

static std::mt19937_64 random_generator;

using namespace std;
using namespace ngraph;

void help()
{
    cout << R"###(
DESCRIPTION
    Run an ONNX model

SYNOPSIS
        run_onnx [-i|--input <input file>] [-o|--output <output file>]

OPTIONS
        -i or --input  input ONNX file

)###";
}

int main(int argc, char** argv)
{
    string input;
    string model;
    string output;
    string backend;
    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-m" || arg == "--model")
        {
            model = argv[++i];
        }
        else if (arg == "-b" || arg == "--backend")
        {
            backend = argv[++i];
        }
        else if (arg == "-i" || arg == "--input")
        {
            input = argv[++i];
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
        std::shared_ptr<ngraph::Function> function = ngraph::onnx_import::import_onnx_model(model);
        auto backend = ngraph::runtime::Backend::create("CPU");

        std::uniform_int_distribution<int> distribution(1, 6);

        // Creating inputs
        vector<shared_ptr<runtime::Tensor>> inputs;
        auto params = function->get_parameters();
        for (int i = 0; i < params.size(); i++)
        {
            auto tensor =
                backend->create_tensor(params.at(i)->get_element_type(), params.at(i)->get_shape());
            auto tensor_size = params.at(i)->get_shape().size();
            std::uniform_int_distribution<int> distribution(0, 255);
            vector<float> v_a(tensor_size, 0);
            double r = 0;
            for (int i = 0; i < tensor_size; i++)
            {
                v_a[i] = distribution(random_generator);
                r += static_cast<double>(v_a[i]);
            }
            copy_data(tensor, v_a);
            inputs.push_back(tensor);
        }

        // Creating outputs
        vector<shared_ptr<runtime::Tensor>> outputs;
        auto outputs_size = function->get_results().size();
        for (int i = 0; i < outputs_size; i++)
        {
            auto tensor = backend->create_tensor(function->get_output_element_type(i),
                                                 function->get_output_shape(i));
            outputs.push_back(tensor);
        }
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
    else
    {
        cout << "Failed to open '" << model << "' for model\n";
        return 2;
    }
    ifstream d(input);
    if (d)
    {
        const string s = input;
        // auto data = read_binary_file(s);
    }
    else
    {
        cout << "Failed to open '" << input << "' for data\n";
        return 2;
    }

    return 0;
}
