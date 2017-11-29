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

#include "call_frame.hpp"
#include "ngraph/runtime/interpreter/op_engine.hpp"
#include "ngraph/runtime/interpreter/tensor_view.hpp"

using namespace std;
using namespace ngraph::runtime::interpreter;

CallFrame::CallFrame(std::shared_ptr<ExternalFunction> external_function,
                     shared_ptr<ngraph::Function> func)
    : m_external_function(external_function)
    , m_function(func)
{
    NGRAPH_INFO;
}

void CallFrame::tensor_call(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& input_tvs,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& output_tvs)
{
    NGRAPH_INFO << "----------------------------------";
    unordered_map<string, shared_ptr<Node>> node_map;
    vector<shared_ptr<runtime::interpreter::CPUTensorView>> inputs;
    vector<shared_ptr<runtime::interpreter::CPUTensorView>> outputs;
    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::interpreter::CPUTensorView> tv =
            static_pointer_cast<runtime::interpreter::CPUTensorView>(input_tvs[i]);
        inputs.push_back(tv);
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::interpreter::CPUTensorView> tv =
            static_pointer_cast<runtime::interpreter::CPUTensorView>(output_tvs[i]);
        outputs.push_back(tv);
    }

    // Invoke computation
    for (shared_ptr<Node> op : m_function->get_ordered_ops())
    {
        NGRAPH_INFO << *op;
        NGRAPH_INFO << op->get_element_type();
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
    }
}

void CallFrame::call(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& arguments,
                     const std::vector<std::shared_ptr<ngraph::runtime::Value>>& results)
{
    NGRAPH_INFO;
    // TODO: Check types of args and result
    vector<shared_ptr<ngraph::runtime::TensorView>> inputs;
    for (shared_ptr<ngraph::runtime::Value> argument : arguments)
    {
        argument->collect_tensor_views(inputs, argument);
    }

    vector<shared_ptr<ngraph::runtime::TensorView>> outputs;
    for (shared_ptr<ngraph::runtime::Value> result : results)
    {
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}
