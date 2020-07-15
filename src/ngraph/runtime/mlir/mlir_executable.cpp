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

#include "ngraph/runtime/mlir/mlir_executable.hpp"
#include "ngraph/chrome_trace.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_downgrade.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"


#include "ngraph/env_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder_registry.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
// #include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor.hpp"
#include "ngraph/runtime/cpu/static_initialize.hpp"
#include "ngraph/util.hpp"


// #include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Module.h"
// #include "mlir/IR/Verifier.h"
// #include "mlir/Parser.h"

// #include "llvm/ADT/StringRef.h"
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/ErrorOr.h"
// #include "llvm/Support/MemoryBuffer.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/raw_ostream.h"



// #include "contrib/mlir/runtime/cpu/cpu_runtime.hpp"
// #include "contrib/mlir/core/pass/mlir_subgraph_extraction.hpp"
// #include "contrib/mlir/core/compiler.hpp"
#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "contrib/mlir/core/compiler.hpp"
// #include "contrib/mlir/backend/backend.hpp"



using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

runtime::mlir::OP_TYPEID runtime::mlir::MlirExecutable::get_typeid(const Node& node)
{
    const NodeTypeInfo& type_info = node.get_type_info();
    // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
    // {Abs::type_info, OP_TYPEID::Abs},
    // {Acos::type_info, OP_TYPEID::Acos},
    // ...
    static const map<NodeTypeInfo, OP_TYPEID> type_info_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, OP_TYPEID::ID_SUFFIX(NAME)},
#include "ngraph/runtime/interpreter/opset_int_tbl.hpp"
#undef NGRAPH_OP
    };
    OP_TYPEID rc = OP_TYPEID::UnknownOp;

    auto it = type_info_map.find(type_info);
    if (it != type_info_map.end())
    {
        rc = it->second;
    }
    return rc;
}

runtime::mlir::MlirExecutable::MlirExecutable(const shared_ptr<Function>& function,
                                              bool enable_performance_collection)
{
    ngmlir::MLIRCompiler::init();
    ngmlir::MLIRCPUBackend::init();

#ifndef NGRAPH_JSON_DISABLE
    // To verify that the serializer and deserializer work correctly let's just run this
    // graph round-trip
    m_function = deserialize(serialize(function));
#else
    m_function = clone_function(*function);
#endif

    auto is_supported = [](const Node& node) {
        bool retval = false;
        switch (MlirExecutable::get_typeid(node))
        {
        case OP_TYPEID::Clamp:
        case OP_TYPEID::MatMul:
        case OP_TYPEID::Mod_v1:
        case OP_TYPEID::Squeeze:
        case OP_TYPEID::Unsqueeze: retval = true; break;
        default: break;
        }
        return retval;
    };
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();
    pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.register_pass<pass::Opset1Downgrade>();
    pass_manager.register_pass<pass::Opset0Downgrade>();
    // Need to decompose any v0 fused ops, which were produced by the downgrade pass
    pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.run_passes(m_function);
    for (auto node : m_function->get_ordered_ops())
    {
        m_nodes.push_back(node);
    }
    set_parameters_and_results(*m_function);
}

bool runtime::mlir::MlirExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                         const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    event::Duration d1("call", "Interpreter");

    // convert inputs to HostTensor
    vector<shared_ptr<HostTensor>> func_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs.push_back(host_tensor);
    }

    // convert outputs to HostTensor
    vector<shared_ptr<HostTensor>> func_outputs;
    for (auto tensor : outputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_outputs.push_back(host_tensor);
    }

    // map function params -> HostTensor
    unordered_map<descriptor::Tensor*, shared_ptr<HostTensor>> tensor_map;
    size_t input_count = 0;
    for (auto param : get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &param->output(i).get_tensor();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < get_results().size(); ++output_count)
    {
        auto output = get_results()[output_count];
        if (!is_type<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::Tensor* tensor = &output->get_output_tensor(0);
        tensor_map.insert({tensor, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (auto op : m_nodes)
    {
        event::Duration d2(op->description(), "Interpreter");
        if (op->is_parameter())
        {
            continue;
        }

        // get op inputs from map
        vector<shared_ptr<HostTensor>> op_inputs;
        for (auto input : op->inputs())
        {
            descriptor::Tensor* tensor = &input.get_tensor();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        vector<shared_ptr<HostTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &op->output(i).get_tensor();
            shared_ptr<HostTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                host_tensor = make_shared<HostTensor>(op->output(i));
                tensor_map.insert({tensor, host_tensor});
            }
            else
            {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }

        // get op type
        element::Type type;
        if (is_type<op::Convert>(op) || is_type<op::Quantize>(op) || is_type<op::Dequantize>(op) ||
            is_type<op::ArgMin>(op) || is_type<op::ArgMax>(op))
        {
            type = op->get_input_element_type(0);
        }
        else if (is_type<op::Equal>(op) || is_type<op::Greater>(op) || is_type<op::GreaterEq>(op) ||
                 is_type<op::Less>(op) || is_type<op::LessEq>(op) || is_type<op::NotEqual>(op))
        {
            // Get the type of the second input, not the first
            // All BinaryElementwiseComparision ops have the same type for inputs
            // Select has bool for first input and the type we are interested in for the second
            type = op->get_input_element_type(1);
        }
        else if (is_type<op::TopK>(op))
        {
            type = op->get_output_element_type(1);
        }
        else
        {
            type = op->get_output_element_type(0);
        }

        // generate_calls(type, *op.get(), op_outputs, op_inputs);
    }

    return true;
}

shared_ptr<ngraph::op::Parameter> runtime::mlir::MlirExecutable::get_parameter(size_t index) const
{
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

shared_ptr<ngraph::op::Result> runtime::mlir::MlirExecutable::get_result(size_t index) const
{
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}
shared_ptr<runtime::Tensor> runtime::mlir::MlirExecutable::create_input_tensor(size_t input_index)
{
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    return make_shared<runtime::HostTensor>(parameter->get_output_element_type(0),
                                            parameter->get_output_shape(0));
}

shared_ptr<runtime::Tensor> runtime::mlir::MlirExecutable::create_output_tensor(size_t output_index)
{
    shared_ptr<op::Result> result = get_result(output_index);
    return make_shared<runtime::HostTensor>(result->get_output_element_type(0),
                                            result->get_output_shape(0));
}

vector<shared_ptr<runtime::Tensor>>
    runtime::mlir::MlirExecutable::create_input_tensor(size_t input_index, size_t pipeline_depth)
{
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::HostTensor> tensor;
        auto t = make_shared<runtime::HostTensor>(parameter->get_output_element_type(0),
                                                  parameter->get_output_shape(0));
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

vector<shared_ptr<runtime::Tensor>>
    runtime::mlir::MlirExecutable::create_output_tensor(size_t output_index, size_t pipeline_depth)
{
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Result> result = get_result(output_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::HostTensor> tensor;
        auto t = make_shared<runtime::HostTensor>(result->get_output_element_type(0),
                                                  result->get_output_shape(0));
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}
