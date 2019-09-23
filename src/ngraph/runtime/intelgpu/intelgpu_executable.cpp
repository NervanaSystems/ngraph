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

#include <iomanip>

#include "ngraph/runtime/intelgpu/intelgpu_executable.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static void
    memory_size_check(size_t memory_size, const shared_ptr<Node>& node, const string& function_name)
{
    const size_t tensor_size = shape_size(node->get_shape()) * node->get_element_type().size();

    if (memory_size != tensor_size)
    {
        ostringstream os;
        os << "IntelGPU backend failed memory check. In \"" << function_name << "\" with Node \""
           << node->get_name() << "\" and " << node->get_shape() << " mismatched memory sizes "
           << tensor_size << " and " << memory_size;
        throw invalid_argument(os.str());
    }
}

static const string& get_input_name(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_tensor().get_name();
}

// The cldnn::network contains something like "generic_layer_0_Parameter_254_0" names
// This function should return "Parameter_254" from the example above
static string convert_cldnn_names(shared_ptr<Function> func, const string& cldnn_name)
{
    const string key("_");
    string result;

    const size_t last_key = cldnn_name.rfind(key);
    const size_t pre_last_key = cldnn_name.rfind(key, last_key - 1);
    const size_t pre_pre_last_key = cldnn_name.rfind(key, pre_last_key - 1);

    if (pre_pre_last_key == std::string::npos)
    {
        result = cldnn_name.substr(0, last_key);
    }
    else
    {
        result = cldnn_name.substr(pre_pre_last_key + 1, last_key - pre_pre_last_key - 1);
    }

    return result;
}

runtime::intelgpu::IntelGPUExecutable::IntelGPUExecutable(shared_ptr<Function> func,
                                                          shared_ptr<cldnn::network> network,
                                                          bool enable_timing,
                                                          bool enable_profile,
                                                          double compilation_time,
                                                          double consumed_memory,
                                                          size_t profile_lines_limit_count)
{
    m_function = func;
    m_cldnn_network = network;
    m_performance_counters_enabled = enable_timing;
    m_profile_enable = enable_profile;
    m_compilation_time = compilation_time;
    m_consumed_memory = consumed_memory;
    m_profile_lines_limit_count = profile_lines_limit_count;

    set_parameters_and_results(*func);
}

bool runtime::intelgpu::IntelGPUExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                                 const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    double mem_call_consumed = 0.0f;
    stopwatch timer_call;

    if (m_cldnn_network == nullptr)
    {
        throw runtime_error("compile() must be called before call().");
    }

    if (m_profile_enable)
    {
        mem_call_consumed = runtime::intelgpu::get_max_memory_rss();
        timer_call.start();
    }

    // Process input parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_parameters and inputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < inputs.size(); i++)
    {
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> tv =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(inputs[i]);
        const ParameterVector& input_params = get_parameters();
        const string& tensor_name = input_params[i]->get_output_tensor().get_name();
        m_cldnn_network->set_input_data(tensor_name, *tv->get_data_ptr());
    }

    // Execute network
    map<cldnn::primitive_id, cldnn::network_output> result = m_cldnn_network->execute();

    // Process output parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_results and outputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < m_function->get_output_size(); i++)
    {
        const shared_ptr<Node>& dst_node = m_function->get_output_op(i);
        const size_t dst_shape_size = shape_size(dst_node->get_shape());

        // We should not touch destination memory if it is not existed
        if (!dst_shape_size)
        {
            continue;
        }

        shared_ptr<runtime::intelgpu::IntelGPUTensorView> ngraph_res =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(outputs[i]);
        const string& tensor_name = get_input_name(dst_node);
        auto result_memory = result.at(tensor_name).get_memory().pointer<char>();

        memory_size_check(result_memory.size(), dst_node, m_function->get_name());

        ngraph_res->write(result_memory.data(), result_memory.size());
    }

    if (m_profile_enable)
    {
        timer_call.stop();
        mem_call_consumed = runtime::intelgpu::get_max_memory_rss() - mem_call_consumed;

        print_call_performance(m_cldnn_network,
                               m_function,
                               m_compilation_time,
                               timer_call.get_milliseconds(),
                               m_consumed_memory,
                               mem_call_consumed,
                               runtime::intelgpu::get_max_memory_rss());

        // Output compile time only once
        m_compilation_time = 0.0;
        m_consumed_memory = 0.0;
    }

    return true;
}

vector<runtime::PerformanceCounter>
    runtime::intelgpu::IntelGPUExecutable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;

    if (m_cldnn_network != nullptr && m_performance_counters_enabled)
    {
        const map<cldnn::primitive_id, cldnn::event>& primitives =
            m_cldnn_network->get_executed_primitives();
        map<string, shared_ptr<const Node>> name_map;
        for (auto n : m_function->get_ops())
        {
            name_map.insert({n->get_name(), n});
        }
        for (const auto& p : primitives)
        {
            // Let's generate the primitive name that matches to the name in Function
            const string primitive_name = convert_cldnn_names(m_function, p.first);
            size_t usec = 0;
            for (const auto& q : p.second.get_profiling_info())
            {
                if (q.name == string("executing"))
                {
                    usec += chrono::duration_cast<
                                chrono::duration<size_t, chrono::milliseconds::period>>(
                                q.value->value())
                                .count();
                }
            }
            shared_ptr<const Node> n = name_map[primitive_name];
            const runtime::PerformanceCounter perf_counter(n, usec, 1);
            rc.push_back(perf_counter);
        }
    }
    return rc;
}

static Node* get_node_by_name(const shared_ptr<Function> func, const string& name)
{
    for (shared_ptr<Node> node : func->get_ops())
    {
        if (node->get_name() == name)
        {
            return node.get();
        }
    }

    return nullptr;
}

void runtime::intelgpu::IntelGPUExecutable::print_call_performance(
    const shared_ptr<cldnn::network> network,
    const shared_ptr<Function> func,
    double time_compile,
    double time_call,
    double mem_compilation_consumed,
    double mem_call_consumed,
    double mem_current) const
{
    struct data_item
    {
        string item_name;
        map<string, double> item_times;
    };
    const string& func_name = func->get_name();
    const map<cldnn::primitive_id, cldnn::event>& primitives = network->get_executed_primitives();
    size_t limit_count = m_profile_lines_limit_count;
    multimap<double, data_item> data;
    map<string, double> total_interval_times;
    double total_executing_time = 0;
    size_t total_items_count = 0;
    size_t max_item_name_size = 0;

    ios_base::fmtflags saved_stream_flags(cout.flags()); // Save stream flags to restore them later

    if (m_profile_lines_limit_count > 0)
    {
        // Extract profiling statistic, calculate summary and sort
        for (auto& prim : primitives)
        {
            double executing_time = 0;
            data_item item;
            item.item_name = prim.first;
            max_item_name_size = max(max_item_name_size, prim.first.size());

            for (auto& prof_info : prim.second.get_profiling_info())
            {
                const string& interval_name = prof_info.name;
                double interval =
                    chrono::duration_cast<chrono::duration<double, chrono::milliseconds::period>>(
                        prof_info.value->value())
                        .count();

                item.item_times[interval_name] = interval;

                // Get the Key time to sort by
                if (interval_name == "executing")
                {
                    executing_time += interval;
                }

                // Accumulate total time for each interval
                if (total_interval_times.find(interval_name) == total_interval_times.end())
                {
                    total_interval_times[interval_name] = interval;
                }
                else
                {
                    total_interval_times[interval_name] += interval;
                }
            }
            data.emplace(executing_time, item);
            total_executing_time += executing_time;
            ++total_items_count;
        }

        // Print statistic for each primitive in the cldnn::network
        for (auto it = data.rbegin(); (it != data.rend()) && (limit_count > 0); ++it, --limit_count)
        {
            const string ngraph_node_name = convert_cldnn_names(func, it->second.item_name);
            const Node* ngraph_node = get_node_by_name(func, ngraph_node_name);

            cout << func_name << delim << setw(max_item_name_size) << it->second.item_name << delim
                 << "time(ms)" << delim << scientific << setprecision(2) << it->first;
            for (auto item : it->second.item_times)
            {
                cout << delim << item.first << "(ms)" << delim << item.second;
            }
            cout << delim << ngraph_node_name;

            if (ngraph_node) // it might be initialized by nullptr
            {
                // print all input shapes for the Node
                size_t arg_idx = 0;
                for (const descriptor::Input& op_input : ngraph_node->get_inputs())
                {
                    cout << delim << op_input.get_element_type().c_type_string() << " input"
                         << arg_idx << vector_to_string(op_input.get_shape());
                    ++arg_idx;
                }

                // print all output shapes for the Node
                arg_idx = 0;
                for (const descriptor::Output& op_output : ngraph_node->get_outputs())
                {
                    cout << delim << op_output.get_element_type().c_type_string() << " output"
                         << arg_idx << vector_to_string(op_output.get_shape());
                    ++arg_idx;
                }
            }

            cout << "\n";
        }

        // Print bottom line summary
        const string total_items_count_string = "Total(cldnn " + to_string(total_items_count) +
                                                ", ngraph " + to_string(func->get_ops().size()) +
                                                ")";
        cout << func_name << delim << setw(max_item_name_size) << total_items_count_string << delim
             << "time(ms)" << delim << scientific << setprecision(2) << total_executing_time;
        for (auto item_times : total_interval_times)
        {
            cout << delim << item_times.first << "(ms)" << delim << item_times.second;
        }
        cout << "\n";
    }

    // Print time and memory consumed in ::call function
    cout << func_name << delim << " Backend compilation(ms)" << delim << time_compile << delim
         << "call(ms)" << delim << time_call << delim << "memory consumption compile(B)" << delim
         << mem_compilation_consumed << delim << "call(B)" << delim << mem_call_consumed << delim
         << "RSS(B)" << delim << mem_current << endl;

    cout.flags(saved_stream_flags); // Restore stream configuration to leave it in original state
}
