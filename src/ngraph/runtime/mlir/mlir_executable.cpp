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
#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "contrib/mlir/core/compiler.hpp"
#include "contrib/mlir/runtime/cpu/cpu_runtime.hpp"
#include "ngraph/chrome_trace.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
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

    m_function = clone_function(*function);

    for (auto op : m_function->get_ordered_ops())
    {
        NGRAPH_INFO << *op;
    }

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
    pass_manager.register_pass<pass::FusedOpDecomposition>();

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

    static bool is_compiled = false;
    if (!is_compiled)
    {
        is_compiled = true;

        // // Tensors haven't been allocated yet so we have to keep a pointer to the pointer
        // // that will hold the future memory address.
        // std::vector<size_t> buffer_indices;
        // std::vector<std::vector<size_t>> shape_vec;
        // std::vector<std::vector<size_t>> strides_vec;
        // for (const TensorWrapper& arg : args)
        // {
        //     auto buffer_index = external_function->get_buffer_index(arg.get_name());
        //     buffer_indices.push_back(buffer_index);
        //     // Get shape and strides
        //     auto tensor_shape = arg.get_shape();
        //     std::vector<size_t> shape(tensor_shape.size());
        //     for (auto i = 0; i < tensor_shape.size(); i++)
        //     {
        //         shape[i] = tensor_shape[i];
        //     }
        //     shape_vec.push_back(shape);
        //     auto tensor_strides = arg.get_strides();
        //     std::vector<size_t> strides(tensor_strides.size());
        //     for (auto i = 0; i < tensor_strides.size(); i++)
        //     {
        //         strides[i] = tensor_strides[i];
        //     }
        //     strides_vec.push_back(strides);
        // }

        // for (const TensorWrapper& result : out)
        // {
        //     auto buffer_index = external_function->get_buffer_index(result.get_name());
        //     buffer_indices.push_back(buffer_index);
        //     // Get shape and strides
        //     auto tensor_shape = result.get_shape();
        //     std::vector<size_t> shape(tensor_shape.size());
        //     for (auto i = 0; i < tensor_shape.size(); i++)
        //     {
        //         shape[i] = tensor_shape[i];
        //     }
        //     shape_vec.push_back(shape);
        //     auto tensor_strides = result.get_strides();
        //     std::vector<size_t> strides(tensor_strides.size());
        //     for (auto i = 0; i < tensor_strides.size(); i++)
        //     {
        //         strides[i] = tensor_strides[i];
        //     }
        //     strides_vec.push_back(strides);
        // }

        // // MLIR requires a list of type-erased pointer to arguments. Tensors must have
        // // been allocated at this point so we can get rid of the extra reference.
        // std::vector<runtime::ngmlir::MemRefArg> mem_ref_arg_vec;
        // int i = 0;
        // for (auto& buffer_index : buffer_indices)
        // {
        //     runtime::ngmlir::MemRefArg mem_ref_arg;
        //     mem_ref_arg.m_tensor = ctx->buffer_data[buffer_index];
        //     mem_ref_arg.m_shape = shape_vec[i];
        //     mem_ref_arg.m_strides = strides_vec[i];
        //     mem_ref_arg_vec.push_back(mem_ref_arg);
        //     i++;
        // }

        NGRAPH_INFO;
        ::mlir::MLIRContext& context = m_mlir_runtime.get_context();
        NGRAPH_INFO;
        runtime::ngmlir::MLIRCompiler mlir_compiler(m_function, context);
        // Compile to NG dialect
        NGRAPH_INFO;
        mlir_compiler.compile();
        // Grab a context and initialize a CPU backend using same context
        NGRAPH_INFO;
        runtime::ngmlir::MLIRCPUBackend mlir_backend(mlir_compiler.get_module(), context);
        // Codegen to LLVM dialect
        NGRAPH_INFO;
        mlir_backend.codegen();
        // Store module into runtime, and invoke.
        NGRAPH_INFO;
        m_mlir_runtime.set_module(mlir_backend.get_module());
        // bool first_iteration = true;
        // m_mlir_runtime.run(mem_ref_arg_vec, first_iteration);
        NGRAPH_INFO;
    }

    std::vector<runtime::ngmlir::MemRefArg> mem_ref_arg_vec;
    NGRAPH_INFO;
    for (auto tensor : inputs)
    {
        auto host_tensor = dynamic_pointer_cast<runtime::HostTensor>(tensor);
        if (!host_tensor)
        {
            throw runtime_error("call args are not HostTensor");
        }
        runtime::ngmlir::MemRefArg mem_ref_arg;
        mem_ref_arg.m_tensor = host_tensor->get_data_ptr();
        mem_ref_arg.m_shape = tensor->get_shape();
        mem_ref_arg.m_strides = tensor->get_strides();
        mem_ref_arg_vec.push_back(mem_ref_arg);
    }

    // convert outputs to HostTensor
    NGRAPH_INFO;
    for (auto tensor : outputs)
    {
        auto host_tensor = dynamic_pointer_cast<runtime::HostTensor>(tensor);
        if (!host_tensor)
        {
            throw runtime_error("call args are not HostTensor");
        }
        runtime::ngmlir::MemRefArg mem_ref_arg;
        mem_ref_arg.m_tensor = host_tensor->get_data_ptr();
        mem_ref_arg.m_shape = tensor->get_shape();
        mem_ref_arg.m_strides = tensor->get_strides();
        mem_ref_arg_vec.push_back(mem_ref_arg);
    }

    static bool first_iteration = true;
    m_mlir_runtime.run(mem_ref_arg_vec, first_iteration);
    first_iteration = false;

    // // map function params -> HostTensor
    // NGRAPH_INFO;
    // unordered_map<descriptor::Tensor*, shared_ptr<HostTensor>> tensor_map;
    // size_t input_count = 0;
    // for (auto param : get_parameters())
    // {
    //     NGRAPH_INFO << *param;
    //     for (size_t i = 0; i < param->get_output_size(); ++i)
    //     {
    //         descriptor::Tensor* tensor = &param->output(i).get_tensor();
    //         NGRAPH_INFO << static_cast<void*>(tensor);
    //         NGRAPH_INFO << tensor->get_tensor_layout()->get_shape();
    //         NGRAPH_INFO;
    //         NGRAPH_INFO << tensor->get_tensor_layout()->get_strides();
    //         NGRAPH_INFO;
    //         tensor_map.insert({tensor, func_inputs[input_count++]});
    //     }
    // }

    // // map function outputs -> HostTensor
    // NGRAPH_INFO;
    // for (size_t output_count = 0; output_count < get_results().size(); ++output_count)
    // {
    //     auto output = get_results()[output_count];
    //     if (!is_type<op::Result>(output))
    //     {
    //         throw ngraph_error("One of function's outputs isn't op::Result");
    //     }
    //     descriptor::Tensor* tensor = &output->get_output_tensor(0);
    //     tensor_map.insert({tensor, func_outputs[output_count]});
    // }

    NGRAPH_INFO;
    // MLIR requires a list of type-erased pointer to arguments. Tensors must have
    // been allocated at this point so we can get rid of the extra reference.
    int i = 0;
    // for (auto& buffer_index : buffer_indices)
    // {
    //     runtime::ngmlir::MemRefArg mem_ref_arg;
    //     mem_ref_arg.m_tensor = ctx->buffer_data[buffer_index];
    //     mem_ref_arg.m_shape = shape_vec[i];
    //     mem_ref_arg.m_strides = strides_vec[i];
    //     mem_ref_arg_vec.push_back(mem_ref_arg);
    //     i++;
    // }

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

// void ngraph::runtime::mlir::MlirRuntime::run(const std::vector<runtime::ngmlir::MemRefArg>& args,
//                                              bool firstIteration)
// {
//     NGRAPH_INFO << "in run";
// }
