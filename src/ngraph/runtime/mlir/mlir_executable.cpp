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
#include "ngraph/pass/convert_opset_1_to_0.hpp"
#include "ngraph/pass/convert_opset_3_to_1.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
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
#define NGRAPH_OP(NAME, VERSION)                                                                   \
    {ngraph::op::v##VERSION::NAME::type_info, OP_TYPEID::NAME##_v##VERSION},
#include "ngraph/op_version_tbl.hpp"
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

    auto is_supported = [](const Node& node) {
        bool retval = false;
        switch (MlirExecutable::get_typeid(node))
        {
        case OP_TYPEID::Clamp_v0:
        case OP_TYPEID::MatMul_v0:
        case OP_TYPEID::Mod_v1:
        case OP_TYPEID::Squeeze_v0:
        case OP_TYPEID::Unsqueeze_v0: retval = true; break;
        default: break;
        }
        return retval;
    };
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();

    pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.register_pass<pass::FusedOpDecomposition>();

    pass_manager.register_pass<pass::ConvertOpset3To1>();
    pass_manager.register_pass<pass::ConvertOpset1To0>();
    // Need to decompose any v0 fused ops, which were produced by the downgrade pass
    pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.register_pass<pass::ZeroDimTensorElimination>();
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

    if (m_first_iteration)
    {
        ::mlir::MLIRContext& context = m_mlir_runtime.get_context();
        runtime::ngmlir::MLIRCompiler mlir_compiler(m_function, context);
        // Compile to NG dialect
        mlir_compiler.compile();
        // Grab a context and initialize a CPU backend using same context
        runtime::ngmlir::MLIRCPUBackend mlir_backend(mlir_compiler.get_module(), context);
        // Codegen to LLVM dialect
        mlir_backend.codegen();
        // Store module into runtime, and invoke.
        m_mlir_runtime.set_module(mlir_backend.get_module());
    }

    std::vector<runtime::ngmlir::MemRefArg> mem_ref_arg_vec;
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

    m_mlir_runtime.run(mem_ref_arg_vec, m_first_iteration);
    m_first_iteration = false;

    return true;
}

shared_ptr<ngraph::op::v0::Parameter>
    runtime::mlir::MlirExecutable::get_parameter(size_t index) const
{
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

shared_ptr<ngraph::op::v0::Result> runtime::mlir::MlirExecutable::get_result(size_t index) const
{
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}
shared_ptr<runtime::Tensor> runtime::mlir::MlirExecutable::create_input_tensor(size_t input_index)
{
    shared_ptr<op::v0::Parameter> parameter = get_parameter(input_index);
    return make_shared<runtime::HostTensor>(parameter->get_output_element_type(0),
                                            parameter->get_output_shape(0));
}

shared_ptr<runtime::Tensor> runtime::mlir::MlirExecutable::create_output_tensor(size_t output_index)
{
    shared_ptr<op::v0::Result> result = get_result(output_index);
    return make_shared<runtime::HostTensor>(result->get_output_element_type(0),
                                            result->get_output_shape(0));
}

vector<shared_ptr<runtime::Tensor>>
    runtime::mlir::MlirExecutable::create_input_tensor(size_t input_index, size_t pipeline_depth)
{
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::v0::Parameter> parameter = get_parameter(input_index);
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
    shared_ptr<op::v0::Result> result = get_result(output_index);
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
