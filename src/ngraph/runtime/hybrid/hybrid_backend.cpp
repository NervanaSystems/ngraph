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

#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    ngraph::runtime::hybrid::backend_map_t backend_map = {{0, "INTERPRETER:0"},
                                                          {1, "INTERPRETER:1"}};
    return new runtime::hybrid::HYBRIDBackend(backend_map);
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

runtime::hybrid::HYBRIDBackend::HYBRIDBackend(const backend_map_t& backend_map)
    : m_backend_map{backend_map}
{
}

shared_ptr<runtime::TensorView>
    runtime::hybrid::HYBRIDBackend::create_tensor(const element::Type& type, const Shape& shape)
{
    return make_shared<runtime::HostTensorView>(type, shape, "external");
}

shared_ptr<runtime::TensorView> runtime::hybrid::HYBRIDBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::HostTensorView>(type, shape, memory_pointer, "external");
}

bool runtime::hybrid::HYBRIDBackend::compile(shared_ptr<Function> function)
{
    FunctionInstance& instance = m_function_map[function];
    if (!instance.m_is_compiled)
    {
        instance.m_is_compiled = true;
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::LikeReplacement>();
        pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
        pass_manager.register_pass<pass::Liveness>();
        pass_manager.run_passes(function);
        instance.m_function = function;
    }

    return true;
}

bool runtime::hybrid::HYBRIDBackend::call(shared_ptr<Function> function,
                                          const vector<shared_ptr<runtime::TensorView>>& outputs,
                                          const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    validate_call(function, outputs, inputs);

    compile(function);
    FunctionInstance& instance = m_function_map[function];

    // // convert inputs to HostTensorView
    // vector<shared_ptr<runtime::HostTensorView>> func_inputs;
    // for (auto tv : inputs)
    // {
    //     func_inputs.push_back(static_pointer_cast<runtime::HostTensorView>(tv));
    // }
    // if (instance.m_nan_check_enabled)
    // {
    //     perform_nan_check(func_inputs);
    // }

    // // convert outputs to HostTensorView
    // vector<shared_ptr<runtime::HostTensorView>> func_outputs;
    // for (auto tv : outputs)
    // {
    //     func_outputs.push_back(static_pointer_cast<runtime::HostTensorView>(tv));
    // }

    // // map function params -> HostTensorView
    // unordered_map<descriptor::TensorView*, shared_ptr<runtime::HostTensorView>> tensor_map;
    // size_t input_count = 0;
    // for (auto param : function->get_parameters())
    // {
    //     for (size_t i = 0; i < param->get_output_size(); ++i)
    //     {
    //         descriptor::Tensor* tv = param->get_output_tensor_ptr(i).get();
    //         tensor_map.insert({tv, func_inputs[input_count++]});
    //     }
    // }

    // // map function outputs -> HostTensorView
    // for (size_t output_count = 0; output_count < function->get_output_size(); ++output_count)
    // {
    //     auto output = function->get_output_op(output_count);
    //     if (!dynamic_pointer_cast<op::Result>(output))
    //     {
    //         throw ngraph_error("One of function's outputs isn't op::Result");
    //     }
    //     descriptor::TensorView* tv = output->get_output_tensor_ptr(0).get();
    //     tensor_map.insert({tv, func_outputs[output_count]});
    // }

    // // for each ordered op in the graph
    // for (const NodeWrapper& wrapped : instance.m_wrapped_nodes)
    // {
    //     const Node* op = &wrapped.get_node();
    //     auto type_id = wrapped.get_typeid();
    //     if (type_id == OP_TYPEID::Parameter)
    //     {
    //         continue;
    //     }
    //     // get op inputs from map
    //     vector<shared_ptr<runtime::HostTensorView>> op_inputs;
    //     for (const descriptor::Input& input : op->get_inputs())
    //     {
    //         descriptor::TensorView* tv = input.get_output().get_tensor_ptr().get();
    //         op_inputs.push_back(tensor_map.at(tv));
    //     }

    //     // get op outputs from map or create
    //     vector<shared_ptr<runtime::HostTensorView>> op_outputs;
    //     for (size_t i = 0; i < op->get_output_size(); ++i)
    //     {
    //         descriptor::TensorView* tv = op->get_output_tensor_ptr(i).get();
    //         shared_ptr<runtime::HostTensorView> htv;
    //         auto it = tensor_map.find(tv);
    //         if (it == tensor_map.end())
    //         {
    //             // the output tensor is not in the tensor map so create a new tensor
    //             const Shape& shape = op->get_output_shape(i);
    //             const element::Type& type = op->get_output_element_type(i);
    //             string name = op->get_output_tensor(i).get_name();
    //             htv = make_shared<runtime::HostTensorView>(type, shape, name);
    //             tensor_map.insert({tv, htv});
    //         }
    //         else
    //         {
    //             htv = it->second;
    //         }
    //         op_outputs.push_back(htv);
    //     }

    //     // get op type
    //     element::Type type;
    //     switch (type_id)
    //     {
    //     case OP_TYPEID::Convert: type = op->get_input_element_type(0); break;
    //     case OP_TYPEID::Equal:
    //     case OP_TYPEID::Greater:
    //     case OP_TYPEID::GreaterEq:
    //     case OP_TYPEID::Less:
    //     case OP_TYPEID::LessEq:
    //     case OP_TYPEID::NotEqual:
    //         // Get the type of the second input, not the first
    //         // All BinaryElementwiseComparision ops have the same type for inputs
    //         // Select has bool for first input and the type we are interested in for the second
    //         type = op->get_input_element_type(1);
    //         break;
    //     default: type = op->get_outputs().at(0).get_element_type(); break;
    //     }

    //     if (instance.m_performance_counters_enabled)
    //     {
    //         instance.m_timer_map[op].start();
    //     }
    //     generate_calls(type, wrapped, op_outputs, op_inputs);
    //     if (instance.m_performance_counters_enabled)
    //     {
    //         instance.m_timer_map[op].stop();
    //     }
    //     if (instance.m_nan_check_enabled)
    //     {
    //         perform_nan_check(op_outputs, op);
    //     }

    //     // delete any obsolete tensors
    //     for (const descriptor::Tensor* t : op->liveness_free_list)
    //     {
    //         for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it)
    //         {
    //             if (it->second->get_tensor().get_name() == t->get_name())
    //             {
    //                 tensor_map.erase(it);
    //                 break;
    //             }
    //         }
    //     }
    // }

    return true;
}

void runtime::hybrid::HYBRIDBackend::enable_performance_data(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::hybrid::HYBRIDBackend::get_performance_data(shared_ptr<Function> func) const
{
    vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_map.at(func);
    for (const pair<const Node*, stopwatch> p : instance.m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}
