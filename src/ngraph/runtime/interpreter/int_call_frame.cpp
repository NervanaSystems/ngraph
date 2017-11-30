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
#include "ngraph/runtime/interpreter/int_op_engine.hpp"
#include "ngraph/runtime/interpreter/int_tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::interpreter::INT_CallFrame::INT_CallFrame(shared_ptr<ExternalFunction> external_function,
                                                   shared_ptr<Function> func)
    : m_external_function(external_function)
    , m_function(func)
{
    NGRAPH_INFO;
}

void runtime::interpreter::INT_CallFrame::tensor_call(
    const vector<shared_ptr<runtime::TensorView>>& input_tvs,
    const vector<shared_ptr<runtime::TensorView>>& output_tvs)
{
    NGRAPH_INFO << "----------------------------------";
    unordered_map<string, shared_ptr<runtime::interpreter::INT_TensorView>> tensor_map;
    const std::vector<std::shared_ptr<op::Parameter>>& params = m_function->get_parameters();

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::interpreter::INT_TensorView> tv =
            static_pointer_cast<runtime::interpreter::INT_TensorView>(input_tvs[i]);
        string name = params[i]->get_name();
        NGRAPH_INFO << "Funtion Inputs " << name;
        tensor_map.insert({name, tv});
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::interpreter::INT_TensorView> tv =
            static_pointer_cast<runtime::interpreter::INT_TensorView>(output_tvs[i]);
        string name = m_function->get_result()->get_name();
        NGRAPH_INFO << "Funtion Outputs " << name;
        tensor_map.insert({name, tv});
    }

    // Invoke computation
    for (shared_ptr<Node> op : m_function->get_ordered_ops())
    {
        NGRAPH_INFO << "op " << *op;

        vector<shared_ptr<runtime::interpreter::INT_TensorView>> inputs;
        vector<shared_ptr<runtime::interpreter::INT_TensorView>> outputs;

        // Allocate any new tensors
        for (const descriptor::Tensor* t : op->liveness_new_list)
        {
            NGRAPH_INFO << "new " << *t;
        }

        for (const descriptor::Input& input : op->get_inputs())
        {
            string name = input.get_output().get_node()->get_name();
            shared_ptr<runtime::interpreter::INT_TensorView> tv = tensor_map.at(name);
            inputs.push_back(tv);
            NGRAPH_INFO << "Op Inputs " << name;
        }
        for (descriptor::Output& output : op->get_outputs())
        {
            string name = output.get_node()->get_name();
            shared_ptr<runtime::interpreter::INT_TensorView> tv;
            if (!contains_key(tensor_map, name))
            {
                const Shape& shape = output.get_tensor_view_type()->get_shape();
                element::Type element_type = output.get_tensor_view_type()->get_element_type();
                // make the output tensor;
                tv = make_shared<runtime::interpreter::INT_TensorView>(element_type, shape);
                tensor_map.insert({name, tv});
            }
            else
            {
                tv = tensor_map.at(name);
            }
            outputs.push_back(tv);
            NGRAPH_INFO << "Op Outputs " << name;
            NGRAPH_INFO << output.get_tensor_view()->get_name();
            NGRAPH_INFO << output.get_tensor().get_name();
        }

        if (op->get_element_type() == element::boolean)
        {
            op_engine<char>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::f32)
        {
            op_engine<float>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::f64)
        {
            op_engine<double>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::i8)
        {
            op_engine<int8_t>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::i16)
        {
            op_engine<int16_t>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::i32)
        {
            op_engine<int32_t>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::i64)
        {
            op_engine<int64_t>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::u8)
        {
            op_engine<uint8_t>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::u16)
        {
            op_engine<uint16_t>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::u32)
        {
            op_engine<uint32_t>(*op, inputs, outputs);
        }
        else if (op->get_element_type() == element::u64)
        {
            op_engine<uint64_t>(*op, inputs, outputs);
        }

        // Delete any obsolete tensors
        // for (const descriptor::Tensor* t : op->liveness_free_list)
        // {
        //     NGRAPH_INFO << "free " << *t;
        // }
    }
    NGRAPH_INFO << "----------------------------------";
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

    NGRAPH_INFO;
    tensor_call(inputs, outputs);
    NGRAPH_INFO;
}
