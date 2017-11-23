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

#include "ngraph/runtime/interpreter/int_call_frame.hpp"
#include "ngraph/runtime/interpreter/int_tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::interpreter::INT_CallFrame::INT_CallFrame(shared_ptr<ExternalFunction> external_function,
                                                   shared_ptr<Function> func)
    : m_external_function(external_function)
    , m_function(func)
{
}

void runtime::interpreter::INT_CallFrame::call(
    std::shared_ptr<Function> function,
    const vector<shared_ptr<runtime::interpreter::INT_TensorView>>& input_tvs,
    const vector<shared_ptr<runtime::interpreter::INT_TensorView>>& output_tvs)
{
    unordered_map<string, shared_ptr<runtime::interpreter::INT_TensorView>> tensor_map;
    const std::vector<std::shared_ptr<op::Parameter>>& params = function->get_parameters();

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        string name = params[i]->get_name();
        tensor_map.insert({name, input_tvs[i]});
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        string name = function->get_result()->get_name();
        tensor_map.insert({name, output_tvs[i]});
    }

    // Invoke computation
    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        vector<shared_ptr<runtime::interpreter::INT_TensorView>> inputs;
        vector<shared_ptr<runtime::interpreter::INT_TensorView>> outputs;

        element::Type base_type;
        if (op->get_inputs().empty())
        {
            base_type = op->get_element_type();
        }
        else
        {
            base_type = op->get_inputs().at(0).get_tensor().get_element_type();
        }
        element::Type secondary_type = op->get_element_type();

        // Some ops have unusual intput/output types so handle those special cases here
        if (op->description() == "Select")
        {
            base_type = op->get_inputs().at(1).get_tensor().get_element_type();
            secondary_type = op->get_inputs().at(0).get_tensor().get_element_type();
        }

        for (const descriptor::Input& input : op->get_inputs())
        {
            string name = input.get_output().get_node()->get_name();
            shared_ptr<runtime::interpreter::INT_TensorView> tv = tensor_map.at(name);
            inputs.push_back(tv);
            // NGRAPH_INFO << "Op Inputs " << name;
        }
        for (descriptor::Output& output : op->get_outputs())
        {
            string name = output.get_node()->get_name();
            shared_ptr<runtime::interpreter::INT_TensorView> tv;
            if (!contains_key(tensor_map, name))
            {
                // The output tensor is not in the tensor map so create a new tensor
                const Shape& shape = output.get_tensor_view_type()->get_shape();
                element::Type element_type = output.get_tensor_view_type()->get_element_type();
                string tensor_name = output.get_tensor().get_name();
                tv = make_shared<runtime::interpreter::INT_TensorView>(
                    element_type, shape, tensor_name);
                tensor_map.insert({name, tv});
            }
            else
            {
                tv = tensor_map.at(name);
            }
            outputs.push_back(tv);
            // NGRAPH_INFO << "Op Outputs " << name;
        }

        if (base_type == element::boolean)
        {
            generate_calls<char>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::f32)
        {
            generate_calls<float>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::f64)
        {
            generate_calls<double>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::i8)
        {
            generate_calls<int8_t>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::i16)
        {
            generate_calls<int16_t>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::i32)
        {
            generate_calls<int32_t>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::i64)
        {
            generate_calls<int64_t>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::u8)
        {
            generate_calls<uint8_t>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::u16)
        {
            generate_calls<uint16_t>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::u32)
        {
            generate_calls<uint32_t>(secondary_type, *op, inputs, outputs);
        }
        else if (base_type == element::u64)
        {
            generate_calls<uint64_t>(secondary_type, *op, inputs, outputs);
        }
        else
        {
            stringstream ss;
            ss << "unsupported element type " << base_type << " op " << op->get_name();
            throw runtime_error(ss.str());
        }

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

void runtime::interpreter::INT_CallFrame::tensor_call(
    const vector<shared_ptr<runtime::interpreter::INT_TensorView>>& input_tvs,
    const vector<shared_ptr<runtime::interpreter::INT_TensorView>>& output_tvs)
{
    call(m_function, input_tvs, output_tvs);
}

void runtime::interpreter::INT_CallFrame::tensor_call(
    const vector<shared_ptr<runtime::TensorView>>& input_tvs,
    const vector<shared_ptr<runtime::TensorView>>& output_tvs)
{
    vector<shared_ptr<runtime::interpreter::INT_TensorView>> args;
    vector<shared_ptr<runtime::interpreter::INT_TensorView>> out;
    for (auto tv : input_tvs)
    {
        args.push_back(static_pointer_cast<runtime::interpreter::INT_TensorView>(tv));
    }
    for (auto tv : output_tvs)
    {
        out.push_back(static_pointer_cast<runtime::interpreter::INT_TensorView>(tv));
    }
    tensor_call(args, out);
}

void runtime::interpreter::INT_CallFrame::call(const vector<shared_ptr<runtime::Value>>& arguments,
                                               const vector<shared_ptr<runtime::Value>>& results)
{
    // TODO: Check types of args and result
    vector<shared_ptr<runtime::TensorView>> inputs;
    for (shared_ptr<runtime::Value> argument : arguments)
    {
        argument->collect_tensor_views(inputs, argument);
    }

    vector<shared_ptr<runtime::TensorView>> outputs;
    for (shared_ptr<runtime::Value> result : results)
    {
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}
