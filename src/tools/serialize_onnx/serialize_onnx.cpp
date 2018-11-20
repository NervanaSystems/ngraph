//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

// tool to serialize an ONNX model to an ngraph json graph

#include <fstream>
#include <iostream>
#include <string>

#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;

void help()
{
    cout << R"###(
DESCRIPTION
    Serialize an ONNX model

SYNOPSIS
        serialize_onnx [-i|--input <input file>] [-o|--output <output file>]

OPTIONS
        -i or --input  input ONNX file
        -o or --output output serialized model
)###";
}

int main(int argc, char** argv)
{
    string input;
    string output;
    for (size_t i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-o" || arg == "--output")
        {
            output = argv[++i];
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

    ifstream f(input);
    if (f)
    {
        std::shared_ptr<ngraph::Function> function = ngraph::onnx_import::import_onnx_model(input);

        ngraph::stopwatch timer;
        timer.start();
        ngraph::serialize(output, function, 2);
        timer.stop();
        cout << "serialize took   " << timer.get_milliseconds() << "ms\n";
    }
    else
    {
        cout << "failed to open '" << input << "' for input\n";
        return 2;
    }

    return 0;
}
