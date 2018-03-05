/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// tool to benchmark any ngraph json model with given backend.
// compile and run with:
// g++ ./nbench.cpp -std=c++11 -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -o nbench
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib env NGRAPH_INTERPRETER_EMIT_TIMING=1 ./nbench
// sample models are under ../../test/models

#include <fstream>
#include <ngraph/file_util.hpp>
#include <ngraph/runtime/backend.hpp>
#include <ngraph/runtime/call_frame.hpp>
#include <ngraph/runtime/manager.hpp>

#include "util/benchmark.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

int main(int argc, char** argv)
{
    string model = "model.json";
    string backend = "CPU";
    int iterations = 0;
    bool failed = false;
    bool statistics = false;
    bool timing_detail = false;
    for (size_t i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-f" || arg == "--file")
        {
            model = argv[++i];
        }
        else if (arg == "-b" || arg == "--backend")
        {
            backend = argv[++i];
        }
        else if (arg == "-i" || arg == "--iterations")
        {
            try
            {
                iterations = stoi(argv[++i]);
            }
            catch (...)
            {
                cout << "Invalid Argument\n";
                failed = true;
            }
        }
        else if (arg == "-s" || arg == "--statistics")
        {
            statistics = true;
        }
        else if (arg == "--timing_detail")
        {
            timing_detail = true;
        }
        else
        {
            cout << "Unknown option: " << arg << endl;
            failed = true;
        }
    }
    if (!static_cast<bool>(ifstream(model)))
    {
        cout << "File " << model << " not found\n";
        failed = true;
    }

    if (failed)
    {
        cout << R"###(
DESCRIPTION
    Benchmark ngraph json model with given backend.

SYNOPSIS
        nbench [-f <filename>] [-b <backend>] [-i <iterations>]

OPTIONS
        -f          model json file to use (default: model.json)
        -b          Backend to use (default: INTERPRETER)
        -i          Iterations (default: 10)
)###";
        return 1;
    }

    const string json_string = file_util::read_file_to_string(model);
    stringstream ss(json_string);
    shared_ptr<Function> f = deserialize(ss);
    if (statistics)
    {
        cout << "statistics:" << endl;
        cout << "total nodes: " << f->get_ops().size() << endl;
        unordered_map<string, size_t> op_list;
        for (shared_ptr<Node> node : f->get_ordered_ops())
        {
            string name = node->get_name();
            string op = name.substr(0, name.find('_'));
            op_list[op]++;

            if (op == "Constant")
            {
                cout << "Constant size: " << join(node->get_outputs()[0].get_shape()) << endl;
            }
        }
        for (const pair<string, size_t>& op_info : op_list)
        {
            cout << op_info.first << ": " << op_info.second << endl;
        }
    }

    if (iterations > 0)
    {
        cout << "Benchmarking " << model << ", " << backend << " backend, " << iterations
             << " iterations.\n";
        run_benchmark(f, backend, iterations, timing_detail);
    }

    return 0;
}
