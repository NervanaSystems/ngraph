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

// tool to benchmark any ngraph json model with given backend.
// compile and run with:
// $ g++ ./nbench.cpp
//             -std=c++11
//             -I$HOME/ngraph_dist/include
//             -L$HOME/ngraph_dist/lib
//             -lngraph -o nbench
// $ env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib env NGRAPH_INTERPRETER_EMIT_TIMING=1 ./nbench
// sample models are under ../../test/models

#include <fstream>
#include <iostream>
#include <string>

#include "ngraph/pass/constant_to_broadcast.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;

void help()
{
    cout << R"###(
DESCRIPTION
    Reserialize a serialized model

SYNOPSIS
        reserialize [-i|--input <input file>] [-o|--output <output file>]

OPTIONS
        -i or --input  input serialized model
        -o or --output output serialized model
        -c or --constant_to_broacast Convert large constant constants to broadcast
)###";
}

int main(int argc, char** argv)
{
    string input;
    string output;
    bool c2b = false;
    for (int i = 1; i < argc; i++)
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
        else if (arg == "-c" || arg == "--constant_to_broadcast")
        {
            c2b = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            help();
            return 0;
        }
    }

    if (input.empty())
    {
        cout << "input file missing\n";
        help();
        return 1;
    }

    if (output.empty())
    {
        cout << "output file missing\n";
        help();
        return 1;
    }

    ifstream f(input);
    if (f)
    {
        ngraph::stopwatch timer;
        timer.start();
        shared_ptr<ngraph::Function> function = ngraph::deserialize(f);
        timer.stop();
        cout << "deserialize took " << timer.get_milliseconds() << "ms\n";

        if (c2b)
        {
            ngraph::pass::Manager pass_manager;
            pass_manager.register_pass<ngraph::pass::ConstantToBroadcast>();
            pass_manager.run_passes(function);
        }

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
