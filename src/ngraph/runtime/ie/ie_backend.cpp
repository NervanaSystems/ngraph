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

#include "ngraph/runtime/ie/ie_backend.hpp"

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorViewLayout;

static bool static_init()
{
    runtime::Backend::register_backend("IE", make_shared<runtime::ie::IE_Backend>());
    return true;
};

bool runtime::ie::IE_Backend::init = static_init();

shared_ptr<runtime::TensorView> runtime::ie::IE_Backend::create_tensor(const element::Type& type,
                                                                       const Shape& shape)
{
    return make_shared<runtime::HostTensorView>(type, shape, "external");
}

shared_ptr<runtime::TensorView> runtime::ie::IE_Backend::create_tensor(const element::Type& type,
                                                                       const Shape& shape,
                                                                       void* memory_pointer)
{
    return make_shared<runtime::HostTensorView>(type, shape, memory_pointer, "external");
}

bool runtime::ie::IE_Backend::compile(shared_ptr<Function> function)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignLayout<DenseTensorViewLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.run_passes(function);

    return true;
}

bool runtime::ie::IE_Backend::call(shared_ptr<Function> function,
                                   const vector<shared_ptr<runtime::TensorView>>& outputs,
                                   const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    validate_call(function, outputs, inputs);

    // TODO: check if function already compiled?
    compile(function);

    // convert inputs to HostTensorView
    vector<shared_ptr<runtime::HostTensorView>> func_inputs;
    for (auto tv : inputs)
    {
        func_inputs.push_back(static_pointer_cast<runtime::HostTensorView>(tv));
    }

    // convert outputs to HostTensorView
    vector<shared_ptr<runtime::HostTensorView>> func_outputs;
    for (auto tv : outputs)
    {
        func_outputs.push_back(static_pointer_cast<runtime::HostTensorView>(tv));
    }

    // map function params -> HostTensorView
    unordered_map<descriptor::TensorView*, shared_ptr<runtime::HostTensorView>> tensor_map;
    size_t input_count = 0;
    for (auto param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = param->get_output_tensor_view(i).get();
            tensor_map.insert({tv, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensorView
    for (size_t output_count = 0; output_count < function->get_output_size(); ++output_count)
    {
        auto output = function->get_output_op(output_count);
        if (!dynamic_pointer_cast<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::TensorView* tv = output->get_output_tensor_view(0).get();
        tensor_map.insert({tv, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        if (op->description() == "Parameter")
        {
            continue;
        }

        // get op inputs from map
        vector<shared_ptr<runtime::HostTensorView>> op_inputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::TensorView* tv = input.get_output().get_tensor_view().get();
            op_inputs.push_back(tensor_map.at(tv));
        }

        // get op outputs from map or create
        vector<shared_ptr<runtime::HostTensorView>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = op->get_output_tensor_view(i).get();
            shared_ptr<runtime::HostTensorView> htv;
            if (!contains_key(tensor_map, tv))
            {
                // the output tensor is not in the tensor map so create a new tensor
                const Shape& shape = op->get_output_shape(i);
                const element::Type& type = op->get_output_element_type(i);
                string name = op->get_output_tensor(i).get_name();
                htv = make_shared<runtime::HostTensorView>(type, shape, name);
                tensor_map.insert({tv, htv});
            }
            else
            {
                htv = tensor_map.at(tv);
            }
            op_outputs.push_back(htv);
        }

        // get op type
        element::Type type = op->get_element_type();
        if (!op->get_inputs().empty())
        {
            type = op->get_inputs().at(0).get_tensor().get_element_type();
        }

        generate_calls(type, *op, op_outputs, op_inputs);

        // delete any obsolete tensors
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

    return true;
}

void runtime::ie::IE_Backend::generate_calls(const element::Type& type,
                                             Node& op,
                                             const vector<shared_ptr<HostTensorView>>& outputs,
                                             const vector<shared_ptr<HostTensorView>>& inputs)
{
    if (type == element::boolean)
    {
        op_engine<char>(op, outputs, inputs);
    }
    else if (type == element::f32)
    {
        op_engine<float>(op, outputs, inputs);
    }
    else if (type == element::f64)
    {
        op_engine<double>(op, outputs, inputs);
    }
    else if (type == element::i8)
    {
        op_engine<int8_t>(op, outputs, inputs);
    }
    else if (type == element::i16)
    {
        op_engine<int16_t>(op, outputs, inputs);
    }
    else if (type == element::i32)
    {
        op_engine<int32_t>(op, outputs, inputs);
    }
    else if (type == element::i64)
    {
        op_engine<int64_t>(op, outputs, inputs);
    }
    else if (type == element::u8)
    {
        op_engine<uint8_t>(op, outputs, inputs);
    }
    else if (type == element::u16)
    {
        op_engine<uint16_t>(op, outputs, inputs);
    }
    else if (type == element::u32)
    {
        op_engine<uint32_t>(op, outputs, inputs);
    }
    else if (type == element::u64)
    {
        op_engine<uint64_t>(op, outputs, inputs);
    }
    else
    {
        stringstream ss;
        ss << "unsupported element type " << type << " op " << op.get_name();
        throw ngraph_error(ss.str());
    }
}
