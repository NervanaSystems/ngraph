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

#include "benchmark.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

int main(int argc, char** argv)
{
    string model;
    string backend = "CPU";
    string directory;
    int iterations = 10;
    bool failed = false;
    bool statistics = false;
    bool timing_detail = false;
    bool visualize = false;
    int warmup_iterations = 1;

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
        else if (arg == "--timing_detail" || arg == "--timing-detail")
        {
            timing_detail = true;
        }
        else if (arg == "-v" || arg == "--visualize")
        {
            visualize = true;
        }
        else if (arg == "-d" || arg == "--directory")
        {
            directory = argv[++i];
        }
        else if (arg == "-w" || arg == "--warmup_iterations")
        {
            try
            {
                warmup_iterations = stoi(argv[++i]);
            }
            catch (...)
            {
                cout << "Invalid Argument\n";
                failed = true;
            }
        }
        else
        {
            cout << "Unknown option: " << arg << endl;
            failed = true;
        }
    }
    if (!model.empty() && !file_util::exists(model))
    {
        cout << "File " << model << " not found\n";
        failed = true;
    }
    else if (!directory.empty() && !file_util::exists(directory))
    {
        cout << "Directory " << model << " not found\n";
        failed = true;
    }
    else if (directory.empty() && model.empty())
    {
        cout << "Either file or directory must be specified\n";
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
        -f|--file                 Serialized model file
        -b|--backend              Backend to use (default: CPU)
        -d|--directory            Directory to scan for models. All models are benchmarked.
        -i|--iterations           Iterations (default: 10)
        -s|--statistics           Display op stastics
        -v|--visualize            Visualize a model (WARNING: requires GraphViz installed)
        --timing-detail           Gather detailed timing
        -w|--warmup_iterations    Number of warm-up iterations
)###";
        return 1;
    }

    if (visualize)
    {
        shared_ptr<Function> f = deserialize(model);
        auto model_file_name = ngraph::file_util::get_file_name(model) + std::string(".") +
                               pass::VisualizeTree::get_file_ext();

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::VisualizeTree>(model_file_name);
        pass_manager.run_passes(f);
    }

    if (statistics)
    {
        shared_ptr<Function> f = deserialize(model);

        cout << "statistics:" << endl;
        cout << "total nodes: " << f->get_ops().size() << endl;
        size_t total_constant_bytes = 0;
        unordered_map<string, size_t> op_list;
        for (shared_ptr<Node> node : f->get_ordered_ops())
        {
            string name = node->get_name();
            string op_name = name.substr(0, name.find('_'));
            string shape_name = "{" + join(node->get_outputs()[0].get_shape()) + "}";
            op_list[op_name + shape_name]++;

            if (op_name == "Constant")
            {
                const Shape& shape = node->get_outputs()[0].get_shape();
                size_t const_size = node->get_outputs()[0].get_element_type().size();
                if (shape.size() == 0)
                {
                    total_constant_bytes += const_size;
                }
                else
                {
                    total_constant_bytes +=
                        (const_size * shape_size(node->get_outputs()[0].get_shape()));
                }
            }
        }
        cout << "Total Constant size: " << total_constant_bytes << " bytes\n";
        for (const pair<string, size_t>& op_info : op_list)
        {
            cout << op_info.first << ": " << op_info.second << " ops" << endl;
        }
    }
    else if (!directory.empty())
    {
        vector<string> models;
        file_util::iterate_files(directory,
                                 [&](const string& file, bool is_dir) {
                                     if (!is_dir)
                                     {
                                         models.push_back(file);
                                     }
                                 },
                                 true);
        for (const string& m : models)
        {
            try
            {
                shared_ptr<Function> f = deserialize(m);
                cout << "Benchmarking " << m << ", " << backend << " backend, " << iterations
                     << " iterations.\n";
                run_benchmark(f, backend, iterations, timing_detail, warmup_iterations);
            }
            catch (exception e)
            {
                cout << "Exception caught on '" << m << "'\n" << e.what() << endl;
            }
        }
    }
    else if (iterations > 0)
    {
        shared_ptr<Function> f = deserialize(model);
        cout << "Benchmarking " << model << ", " << backend << " backend, " << iterations
             << " iterations.\n";
        run_benchmark(f, backend, iterations, timing_detail, warmup_iterations);
    }

    return 0;
}
