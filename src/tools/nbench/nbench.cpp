//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
//             -lngraph
//             -o nbench
// $ env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib env NGRAPH_INTERPRETER_EMIT_TIMING=1 ./nbench
// sample models are under ../../test/models

#include <fstream>
#include <iomanip>

#include "benchmark.hpp"
#include "benchmark_pipelined.hpp"
#include "ngraph/component_manager.hpp"
#include "ngraph/distributed.hpp"
#include "ngraph/except.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#ifdef NGRAPH_MLIR_ENABLE
#include "contrib/mlir/utils.hpp"
#endif

using namespace std;
using namespace ngraph;

static void configure_static_backends()
{
#ifdef NGRAPH_CPU_ENABLE
    ngraph_register_cpu_backend();
#endif
#ifdef NGRAPH_INTERPRETER_ENABLE
    ngraph_register_interpreter_backend();
#endif

#ifdef NGRAPH_MLIR_ENABLE
    // Initialize MLIR
    ngraph::runtime::ngmlir::initializeNGraphMLIR();
#endif
}

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

vector<PerfShape> to_perf_shape(shared_ptr<Function> f,
                                const vector<runtime::PerformanceCounter>& perf_data)
{
    vector<PerfShape> result;
    for (const runtime::PerformanceCounter& p : perf_data)
    {
        auto node = p.get_node();
        if (node == nullptr)
        {
            ostringstream os;
            os << "Can't find \"" << node->get_name() << "\" in Function \"" << f->get_name()
               << "\".";
            throw runtime_error(os.str());
        }

        Shape shape = node->get_output_shape(0);
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
        auto node = p.get_node();
        string op = node->description();
        string shape_name = " {" + join(p.shape) + "} ";
        timing[op + shape_name] += p.microseconds();
        count[op + shape_name] += 1;
    }

    multimap<size_t, string> rc;
    for (auto& t : timing)
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
        auto node = p.get_node();
        string op = node->description();
        timing[op] += p.microseconds();
    }

    multimap<size_t, string> rc;
    for (auto& t : timing)
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
    for (auto& p : timing)
    {
        name_width = max(name_width, static_cast<int>(p.second.size()));
        time_width = max(time_width, static_cast<int>(locale_string(p.first).size()));
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
    string model_arg;
    string backend;
    string directory;
    int iterations = 10;
    bool failed = false;
    bool statistics = false;
    bool timing_detail = false;
    bool visualize = false;
    int warmup_iterations = 1;
    bool copy_data = true;
    bool dot_file = false;
    bool double_buffer = false;

    configure_static_backends();
    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-f" || arg == "--file")
        {
            model_arg = argv[++i];
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
        else if (arg == "--dot")
        {
            dot_file = true;
        }
        else if (arg == "-d" || arg == "--directory")
        {
            directory = argv[++i];
        }
        else if (arg == "--double_buffer")
        {
            double_buffer = true;
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
    if (!model_arg.empty() && !file_util::exists(model_arg))
    {
        cout << "File " << model_arg << " not found\n";
        failed = true;
    }
    else if (!directory.empty() && !file_util::exists(directory))
    {
        cout << "Directory " << directory << " not found\n";
        failed = true;
    }
    else if (directory.empty() && model_arg.empty())
    {
        cout << "Either file or directory must be specified\n";
        failed = true;
    }

    if (failed)
    {
        cout << R"###(
DESCRIPTION
    Benchmark nGraph JSON model with given backend.

SYNOPSIS
        nbench [-f <filename>] [-b <backend>] [-i <iterations>]

OPTIONS
        -f|--file                 Serialized model file
        -b|--backend              Backend to use (default: CPU)
        -d|--directory            Directory to scan for models. All models are benchmarked.
        -i|--iterations           Iterations (default: 10)
        -s|--statistics           Display op statistics
        -v|--visualize            Visualize a model (WARNING: requires Graphviz installed)
        --timing_detail           Gather detailed timing
        -w|--warmup_iterations    Number of warm-up iterations
        --no_copy_data            Disable copy of input/result data every iteration
        --dot                     Generate Graphviz dot file
        --double_buffer           Double buffer inputs and outputs
)###";
        return 1;
    }

    vector<string> models;
    if (!directory.empty())
    {
        vector<PerfShape> aggregate_perf_data;
        file_util::iterate_files(directory,
                                 [&](const string& file, bool is_dir) {
                                     if (!is_dir)
                                     {
                                         models.push_back(file);
                                     }
                                 },
                                 true);
    }
    else
    {
        // Error case where model is missing already checked above
        models.push_back(model_arg);
    }

    vector<PerfShape> aggregate_perf_data;
    int rc = 0;
    for (const string& model : models)
    {
        cout << "\n";
        cout << "============================================================================\n";
        cout << "---- Processing '" << model << "'\n";
        cout << "============================================================================\n";
        try
        {
            if (visualize)
            {
                shared_ptr<Function> f = deserialize(model);
                auto model_file_name = ngraph::file_util::get_file_name(model) +
                                       (dot_file ? ".dot" : ngraph::file_util::get_file_ext(model));

                pass::Manager pass_manager;
                pass_manager.register_pass<pass::VisualizeTree>(model_file_name, nullptr, true);
                pass_manager.run_passes(f);
            }

            if (statistics)
            {
                shared_ptr<Function> f = deserialize(model);

                pass::Manager pass_manager;
                pass_manager.register_pass<pass::Liveness>();
                pass_manager.register_pass<pass::MemoryLayout>();
                pass_manager.run_passes(f);

                cout << "\n---- Source Graph Statistics ----\n";
                cout << "Total nodes: " << locale_string(f->get_ops().size()) << endl;
                size_t total_constant_bytes = 0;
                size_t total_parameter_bytes = 0;
                size_t total_result_bytes = 0;
                size_t total_temporary_bytes = 0;
                size_t total_constant_count = 0;
                size_t total_parameter_count = 0;
                size_t total_result_count = 0;
                size_t total_temporary_count = 0;
                unordered_map<string, size_t> op_list;
                set<string> type_list;
                for (shared_ptr<Node> node : f->get_ordered_ops())
                {
                    for (auto value : node->outputs())
                    {
                        type_list.insert(value.get_element_type().c_type_string());
                    }
                    for (descriptor::Tensor* tensor : node->liveness_new_list)
                    {
                        total_temporary_bytes += tensor->size();
                        total_temporary_count++;
                    }
                    string op_name = node->description();
                    string shape_name = "{" + join(node->get_output_shape(0)) + "}";
                    op_list[op_name + shape_name]++;

                    if (node->is_constant())
                    {
                        total_constant_count++;
                        const Shape& shape = node->get_output_shape(0);
                        size_t const_size = node->output(0).get_element_type().size();
                        if (shape.size() == 0)
                        {
                            total_constant_bytes += const_size;
                        }
                        else
                        {
                            total_constant_bytes +=
                                (const_size * shape_size(node->get_output_shape(0)));
                        }
                    }
                    else if (node->is_parameter())
                    {
                        total_parameter_count++;
                        const Shape& shape = node->get_output_shape(0);
                        size_t size = node->output(0).get_element_type().size() * shape_size(shape);
                        total_parameter_bytes += size;
                    }
                    else if (is_type<op::Result>(node))
                    {
                        total_result_count++;
                        const Shape& shape = node->get_input_shape(0);
                        size_t size = node->get_input_element_type(0).size() * shape_size(shape);
                        total_result_bytes += size;
                    }
                }
                cout << "--\n";
                cout << "Total Constant size: " << locale_string(total_constant_bytes)
                     << " bytes in " << total_constant_count << " constants\n";
                cout << "Total Parameter size: " << locale_string(total_parameter_bytes)
                     << " bytes in " << total_parameter_count << " parameters\n";
                cout << "Total Result size: " << locale_string(total_result_bytes) << " bytes in "
                     << total_result_count << " results\n";
                cout << "Total Temporary size: " << locale_string(total_temporary_bytes)
                     << " bytes in " << total_temporary_count << " temporaries\n";
                cout << "Temporary size with reuse : "
                     << locale_string(f->get_temporary_pool_size()) << " bytes\n";
                cout << "--\n";
                cout << "Types used:\n";
                for (const string& type : type_list)
                {
                    cout << "    " << type << "\n";
                }
                cout << "--\n";
                for (auto& op_info : op_list)
                {
                    cout << op_info.first << ": " << op_info.second << " ops" << endl;
                }
            }

            if (!backend.empty())
            {
                cout << "\n---- Benchmark ----\n";
                stopwatch t1;
                t1.start();
                shared_ptr<Function> f = deserialize(model);
                stringstream ss;
                ss.imbue(locale(""));
                ss << t1.get_milliseconds();
                cout << "deserialize took " << ss.str() << "ms\n";
                vector<runtime::PerformanceCounter> perf_data;
                if (double_buffer)
                {
                    perf_data = run_benchmark_pipelined(
                        f, backend, iterations, timing_detail, warmup_iterations, copy_data);
                }
                else
                {
                    perf_data = run_benchmark(
                        f, backend, iterations, timing_detail, warmup_iterations, copy_data);
                }
                auto perf_shape = to_perf_shape(f, perf_data);
                aggregate_perf_data.insert(
                    aggregate_perf_data.end(), perf_shape.begin(), perf_shape.end());
                print_results(perf_shape, timing_detail);
            }
        }
        catch (ngraph::unsupported_op& ue)
        {
            cout << "Unsupported op '" << ue.what() << "' in model " << model << endl;
            rc += 1;
        }
        catch (exception& e)
        {
            cout << "Exception caught on '" << model << "'\n" << e.what() << endl;
            rc += 1;
        }
    }

    if (models.size() > 1)
    {
        cout << "\n";
        cout << "============================================================================\n";
        cout << "---- Aggregate over all models\n";
        cout << "============================================================================\n";
        print_results(aggregate_perf_data, timing_detail);
    }

    return rc;
}
