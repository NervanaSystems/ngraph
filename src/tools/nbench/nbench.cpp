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

// tool to benchmark any ngraph json model with given backend.
// compile and run with:
// g++ ./nbench.cpp -std=c++11 -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -o nbench
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib env NGRAPH_INTERPRETER_EMIT_TIMING=1 ./nbench
// sample models are under ../../test/models

#include <fstream>
#include <iomanip>

#include "benchmark.hpp"
#include "ngraph/except.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

class PerfShape : public ngraph::runtime::PerformanceCounter
{
public:
    PerfShape(const runtime::PerformanceCounter& p, Shape s)
        : PerformanceCounter(p)
        , shape(s)
    {
    }
    Shape shape;
};

unordered_map<string, shared_ptr<Node>> get_node_map(shared_ptr<Function> func)
{
    unordered_map<string, shared_ptr<Node>> node_map;
    vector<shared_ptr<Function>> fs;
    traverse_functions(func, [&](shared_ptr<Function> f) { fs.push_back(f); });
    for (shared_ptr<Function> f : fs)
    {
        for (shared_ptr<Node> node : f->get_ops())
        {
            node_map.insert({node->get_name(), node});
        }
    }
    return node_map;
}

vector<PerfShape> to_perf_shape(shared_ptr<Function> f,
                                const vector<runtime::PerformanceCounter>& perf_data)
{
    vector<PerfShape> result;
    auto node_map = get_node_map(f);
    for (const runtime::PerformanceCounter& p : perf_data)
    {
        auto node = node_map[p.name()];
        Shape shape = node->get_outputs()[0].get_shape();
        result.push_back(PerfShape(p, shape));
    }
    return result;
}

multimap<size_t, string> aggregate_timing_details(const vector<PerfShape>& perf_data)
{
    unordered_map<string, size_t> timing;
    unordered_map<string, size_t> count;
    for (const PerfShape& p : perf_data)
    {
        string op = p.name().substr(0, p.name().find('_'));
        string shape_name = " {" + join(p.shape) + "} ";
        timing[op + shape_name] += p.microseconds();
        count[op + shape_name] += 1;
    }

    multimap<size_t, string> rc;
    for (const pair<string, size_t>& t : timing)
    {
        rc.insert({t.second, t.first + to_string(count[t.first])});
    }
    return rc;
}

multimap<size_t, string> aggregate_timing(const vector<PerfShape>& perf_data)
{
    unordered_map<string, size_t> timing;
    for (const PerfShape& p : perf_data)
    {
        string op = p.name().substr(0, p.name().find('_'));
        timing[op] += p.microseconds();
    }

    multimap<size_t, string> rc;
    for (const pair<string, size_t>& t : timing)
    {
        rc.insert({t.second, t.first});
    }
    return rc;
}

void print_times(const multimap<size_t, string>& timing)
{
    // set the column widths
    int name_width = 0;
    int time_width = 0;
    for (const pair<size_t, string>& p : timing)
    {
        name_width = max(name_width, static_cast<int>(p.second.size()));
        stringstream ss;
        ss.imbue(locale(""));
        ss << p.first;
        time_width = max(time_width, static_cast<int>(ss.str().size()));
    }
    for (auto it = timing.rbegin(); it != timing.rend(); it++)
    {
        cout << setw(name_width + 2) << left << it->second << " " << setw(time_width + 2) << right
             << it->first << "us\n";
    }
}

void print_results(vector<PerfShape> perf_data, bool timing_detail)
{
    sort(perf_data.begin(), perf_data.end(), [](const PerfShape& p1, const PerfShape& p2) {
        return p1.total_microseconds() > p2.total_microseconds();
    });
    multimap<size_t, string> timing = aggregate_timing(perf_data);
    multimap<size_t, string> timing_details = aggregate_timing_details(perf_data);

    if (timing_detail)
    {
        cout << "\n---- Aggregate times per op type ----\n";
        print_times(timing);

        cout << "\n---- Aggregate times per op type/shape/count ----\n";
        print_times(timing_details);
    }
}

int main(int argc, char** argv)
{
    string model;
    string backend;
    string directory;
    int iterations = 10;
    bool failed = false;
    bool statistics = false;
    bool timing_detail = false;
    bool visualize = false;
    int warmup_iterations = 1;
    bool copy_data = true;

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
        else if (arg == "--no_copy_data")
        {
            copy_data = false;
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
    else if (backend.empty())
    {
        cout << "Backend missing\n";
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
        --timing_detail           Gather detailed timing
        -w|--warmup_iterations    Number of warm-up iterations
        --no_copy_data            Disable copy of input/result data every iteration
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
        vector<PerfShape> aggregate_perf_data;
        file_util::iterate_files(directory,
                                 [&](const string& file, bool is_dir) {
                                     if (!is_dir)
                                     {
                                         models.push_back(file);
                                     }
                                 },
                                 true);
        unordered_map<string, Shape> shape_info;
        cout << "Benchmarking " << endl;
        cout << "    Backend: " << backend << endl;
        cout << "    Iterations: " << iterations << endl;
        cout << "    Warmup: " << warmup_iterations << endl;
        cout << "    Copy Data: " << (copy_data ? "true" : "false") << endl;
        for (const string& m : models)
        {
            cout << "Benchmarking " << m << endl;
            try
            {
                shared_ptr<Function> f = deserialize(m);
                auto perf_data = run_benchmark(
                    f, backend, iterations, timing_detail, warmup_iterations, copy_data);
                auto perf_shape = to_perf_shape(f, perf_data);
                aggregate_perf_data.insert(
                    aggregate_perf_data.end(), perf_shape.begin(), perf_shape.end());
            }
            catch (ngraph::unsupported_op ue)
            {
                cout << "Unsupported op '" << ue.what() << "' in model " << m << endl;
            }
            catch (exception e)
            {
                cout << "Exception caught on '" << m << "'\n" << e.what() << endl;
            }
        }
        print_results(aggregate_perf_data, timing_detail);
    }
    else if (iterations > 0)
    {
        try
        {
            shared_ptr<Function> f = deserialize(model);
            cout << "Benchmarking " << model << endl;
            cout << "    Backend: " << backend << endl;
            cout << "    Iterations: " << iterations << endl;
            cout << "    Warmup: " << warmup_iterations << endl;
            cout << "    Copy Data: " << (copy_data ? "true" : "false") << endl;
            auto perf_data =
                run_benchmark(f, backend, iterations, timing_detail, warmup_iterations, copy_data);
            auto perf_shape = to_perf_shape(f, perf_data);
            print_results(perf_shape, timing_detail);
        }
        catch (ngraph::unsupported_op ue)
        {
            cout << "Unsupported op '" << ue.what() << endl;
        }
        catch (exception e)
        {
            cout << e.what() << endl;
        }
    }

    return 0;
}
