// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cstdlib>
#include <iomanip>

#include "ngraph/runtime/gpu_interpreter/int_call_frame.hpp"
#include "ngraph/runtime/gpu_interpreter/int_tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::gpu_interpreter::INT_CallFrame::INT_CallFrame(shared_ptr<ExternalFunction> external_function,
                                                   shared_ptr<Function> func)
    : m_external_function(external_function)
    , m_function(func)
    , m_emit_timing(std::getenv("NGRAPH_INTERPRETER_EMIT_TIMING") != nullptr)
    , m_nan_check(std::getenv("NGRAPH_INTERPRETER_NAN_CHECK") != nullptr)
{
}

void runtime::gpu_interpreter::INT_CallFrame::call(
    std::shared_ptr<Function> function,
    const vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>>& input_tvs,
    const vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>>& output_tvs)
{
    if (m_nan_check)
    {
        perform_nan_check(input_tvs);
    }
    unordered_map<descriptor::TensorView*, shared_ptr<runtime::gpu_interpreter::INT_TensorView>>
        tensor_map;

    size_t arg_index = 0;
    for (shared_ptr<op::Parameter> param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = param->get_output_tensor_view(i).get();
            string name = tv->get_tensor().get_name();
            tensor_map.insert({tv, input_tvs[arg_index++]});
        }
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        descriptor::TensorView* tv = function->get_output_op(i)->get_output_tensor_view(0).get();
        string name = tv->get_tensor().get_name();
        if (contains_key(tensor_map, tv))
        {
            // Here we handle the special case where an output is just a copy of an input
            memcpy(output_tvs[i]->get_data_ptr(),
                   tensor_map.at(tv)->get_data_ptr(),
                   tv->get_tensor().size());
        }
        else
        {
            tensor_map.insert({tv, output_tvs[i]});
        }
    }

    // create alias list
    size_t output_index = 0;
    unordered_map<descriptor::TensorView*, vector<size_t>> output_alias_map;
    vector<size_t> aliases;
    for (size_t i = 0; i < function->get_output_size(); ++i)
    {
        shared_ptr<descriptor::TensorView> otv =
            function->get_output_op(i)->get_output_tensor_view(0);
        vector<size_t>& al = output_alias_map[otv.get()];
        al.push_back(output_index);
        if (al.size() > 1)
        {
            aliases.push_back(output_index);
        }
        output_index++;
    }

    // Invoke computation
    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        if (op->description() == "Parameter")
        {
            continue;
        }

        vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>> inputs;
        vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>> outputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::TensorView* tv = input.get_output().get_tensor_view().get();
            string name = tv->get_tensor().get_name();
            inputs.push_back(tensor_map.at(tv));
        }
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = op->get_output_tensor_view(i).get();
            string name = tv->get_tensor().get_name();
            shared_ptr<runtime::gpu_interpreter::INT_TensorView> itv;
            if (!contains_key(tensor_map, tv))
            {
                // The output tensor is not in the tensor map so create a new tensor
                const Shape& shape = op->get_output_shape(i);
                const element::Type& element_type = op->get_output_element_type(i);
                string tensor_name = op->get_output_tensor(i).get_name();
                itv = make_shared<runtime::gpu_interpreter::INT_TensorView>(
                    element_type, shape, tensor_name);
                tensor_map.insert({tv, itv});
            }
            else
            {
                itv = tensor_map.at(tv);
            }
            outputs.push_back(itv);
        }

        element::Type base_type;
        element::Type secondary_type;
        if (op->get_inputs().empty())
        {
            base_type = op->get_element_type();
        }
        else
        {
            base_type = op->get_inputs().at(0).get_tensor().get_element_type();
        }
        secondary_type = op->get_element_type();

        // Some ops have unusual intput/output types so handle those special cases here
        if (op->description() == "Select")
        {
            base_type = op->get_inputs().at(1).get_tensor().get_element_type();
            secondary_type = op->get_inputs().at(0).get_tensor().get_element_type();
        }

        if (m_emit_timing)
        {
            m_timer_map[op.get()].start();
        }
        generate_calls(base_type, secondary_type, *op, inputs, outputs);
        if (m_emit_timing)
        {
            stopwatch& timer = m_timer_map[op.get()];
            timer.stop();
        }
        if (m_nan_check)
        {
            perform_nan_check(outputs, op.get());
        }

        handle_output_alias(*op, output_alias_map, output_tvs);

        // Delete any obsolete tensors
        for (const descriptor::Tensor* t : op->liveness_free_list)
        {
            for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it)
            {
                if (it->second->get_tensor().get_name() == t->get_name())
                {
                    tensor_map.erase(it);
                    break;
                }
            }
        }
    }
}

void runtime::gpu_interpreter::INT_CallFrame::handle_output_alias(
    const Node& node,
    const unordered_map<descriptor::TensorView*, vector<size_t>>& output_alias_map,
    const vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>>& output_tvs)
{
    for (size_t i = 0; i < node.get_output_size(); ++i)
    {
        shared_ptr<descriptor::TensorView> otv = node.get_output_tensor_view(i);
        auto it = output_alias_map.find(otv.get());
        if (it != output_alias_map.end())
        {
            const vector<size_t>& outputs = it->second;
            if (outputs.size() > 1)
            {
                for (size_t j = 1; j < outputs.size(); j++)
                {
                    memcpy(static_cast<void*>(output_tvs[j]->get_data_ptr()),
                           static_cast<void*>(output_tvs[0]->get_data_ptr()),
                           otv->get_tensor().size());
                }
            }
        }
    }
}

void runtime::gpu_interpreter::INT_CallFrame::generate_calls(
    const element::Type& base_type,
    const element::Type& secondary_type,
    ngraph::Node& op,
    const std::vector<std::shared_ptr<INT_TensorView>>& args,
    const std::vector<std::shared_ptr<INT_TensorView>>& out)
{
    if (base_type == element::boolean)
    {
        generate_calls<char>(secondary_type, op, args, out);
    }
    else if (base_type == element::f32)
    {
        generate_calls<float>(secondary_type, op, args, out);
    }
    else if (base_type == element::f64)
    {
        generate_calls<double>(secondary_type, op, args, out);
    }
    else if (base_type == element::i8)
    {
        generate_calls<int8_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::i16)
    {
        generate_calls<int16_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::i32)
    {
        generate_calls<int32_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::i64)
    {
        generate_calls<int64_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::u8)
    {
        generate_calls<uint8_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::u16)
    {
        generate_calls<uint16_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::u32)
    {
        generate_calls<uint32_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::u64)
    {
        generate_calls<uint64_t>(secondary_type, op, args, out);
    }
    else
    {
        stringstream ss;
        ss << "unsupported element type " << base_type << " op " << op.get_name();
        throw runtime_error(ss.str());
    }
}

void runtime::gpu_interpreter::INT_CallFrame::tensor_call(
    const vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>>& input_tvs,
    const vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>>& output_tvs)
{
    call(m_function, input_tvs, output_tvs);
}

void runtime::gpu_interpreter::INT_CallFrame::tensor_call(
    const vector<shared_ptr<runtime::TensorView>>& input_tvs,
    const vector<shared_ptr<runtime::TensorView>>& output_tvs)
{
    vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>> args;
    vector<shared_ptr<runtime::gpu_interpreter::INT_TensorView>> out;
    for (auto tv : input_tvs)
    {
        args.push_back(static_pointer_cast<runtime::gpu_interpreter::INT_TensorView>(tv));
    }
    for (auto tv : output_tvs)
    {
        out.push_back(static_pointer_cast<runtime::gpu_interpreter::INT_TensorView>(tv));
    }
    tensor_call(args, out);
}

void runtime::gpu_interpreter::INT_CallFrame::call(
    const vector<shared_ptr<runtime::TensorView>>& arguments,
    const vector<shared_ptr<runtime::TensorView>>& results)
{
    vector<shared_ptr<runtime::TensorView>> inputs;
    for (shared_ptr<runtime::TensorView> argument : arguments)
    {
        argument->collect_tensor_views(inputs, argument);
    }

    vector<shared_ptr<runtime::TensorView>> outputs;
    for (shared_ptr<runtime::TensorView> result : results)
    {
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}

vector<runtime::PerformanceCounter>
    runtime::gpu_interpreter::INT_CallFrame::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    for (const pair<const Node*, stopwatch> p : m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}

void runtime::gpu_interpreter::INT_CallFrame::perform_nan_check(
    const vector<shared_ptr<INT_TensorView>>& tvs, const Node* op)
{
    size_t arg_number = 1;
    for (shared_ptr<INT_TensorView> tv : tvs)
    {
        const element::Type& type = tv->get_tensor().get_element_type();
        if (type == element::f32)
        {
            const float* data = reinterpret_cast<float*>(tv->get_data_ptr());
            for (size_t i = 0; i < tv->get_element_count(); i++)
            {
                if (isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        else if (type == element::f64)
        {
            const double* data = reinterpret_cast<double*>(tv->get_data_ptr());
            for (size_t i = 0; i < tv->get_element_count(); i++)
            {
                if (isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        arg_number++;
    }
}

void runtime::gpu_interpreter::INT_CallFrame::set_nan_check(bool value)
{
    m_nan_check = value;
}
