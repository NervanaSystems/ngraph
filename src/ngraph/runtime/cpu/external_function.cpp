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

#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/codegen/compiler.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_tuple_element.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/ops/tuple.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/cpu/call_frame.hpp"
#include "ngraph/runtime/cpu/emitter.hpp"
#include "ngraph/runtime/cpu/external_function.hpp"
#include "ngraph/runtime/utils.hpp"

using namespace std;
using namespace ngraph::runtime::cpu;

using ngraph::descriptor::layout::DenseTensorViewLayout;

#define TI(x) type_index(typeid(x))

static const OpMap dispatcher{
    {TI(ngraph::op::Add), &Emitter::EmitAdd},
    {TI(ngraph::op::Dot), &Emitter::EmitDot},
    {TI(ngraph::op::Multiply), &Emitter::EmitMultiply},
    {TI(ngraph::op::Parameter), &Emitter::EmitNop},
    {TI(ngraph::op::GetTupleElement), &Emitter::EmitGetTupleElement},
    {TI(ngraph::op::Tuple), &Emitter::EmitTuple},
    {TI(ngraph::op::Abs), &Emitter::EmitAbs},
    {TI(ngraph::op::Concat), &Emitter::EmitConcat},
    {TI(ngraph::op::Divide), &Emitter::EmitDivide},
    {TI(ngraph::op::Equal), &Emitter::EmitEqual},
    {TI(ngraph::op::Greater), &Emitter::EmitGreater},
    {TI(ngraph::op::GreaterEq), &Emitter::EmitGreaterEq},
    {TI(ngraph::op::Less), &Emitter::EmitLess},
    {TI(ngraph::op::LessEq), &Emitter::EmitLessEq},
    {TI(ngraph::op::Log), &Emitter::EmitLog},
    {TI(ngraph::op::Maximum), &Emitter::EmitMaximum},
    {TI(ngraph::op::Minimum), &Emitter::EmitMinimum},
    {TI(ngraph::op::Negative), &Emitter::EmitNegative},
    {TI(ngraph::op::NotEqual), &Emitter::EmitNotEqual},
    {TI(ngraph::op::Select), &Emitter::EmitSelect},
    {TI(ngraph::op::Subtract), &Emitter::EmitSubtract},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::Bool>),
     &Emitter::EmitParameterizedConstantBool},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::Float32>),
     &Emitter::EmitParameterizedConstantFloat32},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::Int8>),
     &Emitter::EmitParameterizedConstantInt8},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::Int32>),
     &Emitter::EmitParameterizedConstantInt32},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::Int64>),
     &Emitter::EmitParameterizedConstantInt64},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::UInt8>),
     &Emitter::EmitParameterizedConstantUInt8},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::UInt32>),
     &Emitter::EmitParameterizedConstantUInt32},
    {TI(ngraph::op::ParameterizedConstant<ngraph::element::UInt64>),
     &Emitter::EmitParameterizedConstantUInt64},
    {TI(ngraph::op::Broadcast), &Emitter::EmitBroadcast},
    {TI(ngraph::op::Convert), &Emitter::EmitConvert},
    {TI(ngraph::op::Constant), &Emitter::EmitConstant},
    {TI(ngraph::op::Reshape), &Emitter::EmitReshape},
    {TI(ngraph::op::FunctionCall), &Emitter::EmitFunctionCall},
    {TI(ngraph::op::Reduce), &Emitter::EmitReduce},
    {TI(ngraph::op::Sign), &Emitter::EmitSign},
    {TI(ngraph::op::Slice), &Emitter::EmitSlice},
    {TI(ngraph::op::Sum), &Emitter::EmitSum},
    {TI(ngraph::op::Exp), &Emitter::EmitExp},
    {TI(ngraph::op::Sin), &Emitter::EmitSin},
    {TI(ngraph::op::Sinh), &Emitter::EmitSinh},
    {TI(ngraph::op::Cos), &Emitter::EmitCos},
    {TI(ngraph::op::Cosh), &Emitter::EmitCosh},
    {TI(ngraph::op::Tan), &Emitter::EmitTan},
    {TI(ngraph::op::Tanh), &Emitter::EmitTanh},
    {TI(ngraph::op::Asin), &Emitter::EmitAsin},
    {TI(ngraph::op::Acos), &Emitter::EmitAcos},
    {TI(ngraph::op::Atan), &Emitter::EmitAtan},
};

#undef TI

ExternalFunction::ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                   bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
    , compiled_function(nullptr)
{
}

void ExternalFunction::compile(FunctionMap& function_map)
{
    if (m_is_compiled)
    {
        return;
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<pass::PropagateTypes>();
    pass_manager.register_pass<pass::AssignTensors>();
    // For now, just make everyone row-major.
    pass_manager.register_pass<pass::AssignLayout<DenseTensorViewLayout>>();
    pass_manager.run_passes(m_function);

    // Determine tensor requirements for the call frame
    unordered_map<shared_ptr<ngraph::descriptor::TensorView>, size_t> tensor_index;

    // First come the function outputs
    for (const descriptor::Output& output : m_function->get_result()->get_outputs())
    {
        auto tv = output.get_tensor_view();
        size_t index = tensor_index.size();
        tensor_index[tv] = index;
    }
    m_n_outputs = tensor_index.size();

    // Next are the function inputs
    for (auto param : m_function->get_parameters())
    {
        for (const descriptor::Output& output : param->get_outputs())
        {
            auto tv = output.get_tensor_view();
            size_t index = tensor_index.size();
            tensor_index[tv] = index;
        }
    }
    m_n_inputs = tensor_index.size() - m_n_outputs;

    // All remaining tensor views
    for (shared_ptr<Node> node : m_function->get_ordered_ops())
    {
        for (const descriptor::Output& output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            if (0 == tensor_index.count(tv))
            {
                size_t index = tensor_index.size();
                tensor_index[tv] = index;
                m_temp_views.push_back(tv);
            }
        }
    }

    // Now we build the TU
    Emitter emitter;
    auto& TU = emitter.GetTU();
    TU += R"(// Generated by the NGraph CPU backend
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"
#include "ngraph/runtime/utils.hpp"
#include "ngraph/runtime/cpu/call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/eigen_utils.hpp"

using namespace ngraph::element;
using namespace ngraph::runtime;
using namespace ngraph::runtime::cpu::eigen;

void *__dso_handle = 0;

extern "C" void __entrypoint(ngraph::runtime::cpu::CallFrame* call_frame,
                             ngraph::runtime::TensorViewPtrs& tensor_views,
                             const std::vector<std::shared_ptr<ngraph::runtime::cpu::CallFrame>>& callees)
{
)";

    for (shared_ptr<Node> node : m_function->get_ordered_ops())
    {
        auto& n = *node; // Work around a compiler warning (*node inside typeid may have effects
                         // with shared pointers, which is fine here but clang doesn't like it.)
        auto handler = dispatcher.find(type_index(typeid(n)));
        if (handler == dispatcher.end())
        {
            throw ngraph_error("Unhandled op during code generation : " + node->description());
        }
        std::vector<TensorViewInfo> in;
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            auto tv = output.get_tensor_view();
            in.push_back({tensor_index.at(tv), tv});
        }
        std::vector<TensorViewInfo> out;
        for (const descriptor::Output& output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            out.push_back({tensor_index.at(tv), tv});
        }
        handler->second(&emitter, node.get(), this, function_map, in, out);
    }

    // End TU
    TU += "}\n";

    // TODO: Cleanup and make this a utility function
    ofstream out("__ngcpu_codegen.cpp");
    out << TU;
    out.close();

    ngraph::codegen::execution_state estate;

#if NGCPU_PCH
    estate.set_precompiled_headers_enabled(true);
#endif

#if NGCPU_DEBUGINFO
    estate.set_debuginfo_enabled(true);
#endif

    auto llvm_module = estate.compile(TU, "__ngcpu_codegen.cpp");
    assert(llvm_module);
    estate.add_module(llvm_module);
    estate.finalize();
    compiled_function = estate.find_function<void(
                            ngraph::runtime::cpu::CallFrame*,
                            ngraph::runtime::TensorViewPtrs&,
                            const std::vector<std::shared_ptr<CallFrame>>&)>("__entrypoint");
    assert(compiled_function);

    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

// Suppress Clang's complaints about the ,##__VA_ARGS__ token-pasting hack, which is a GNU extension
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#define DO_ON_ELEMENT_TYPE(et, err_msg, macro, ...)                                                \
    {                                                                                              \
        if (et == element::Bool::element_type())                                                   \
        {                                                                                          \
            macro(element::Bool, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::Float32::element_type())                                           \
        {                                                                                          \
            macro(element::Float32, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::Int8::element_type())                                              \
        {                                                                                          \
            macro(element::Int8, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::Int32::element_type())                                             \
        {                                                                                          \
            macro(element::Int32, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::Int64::element_type())                                             \
        {                                                                                          \
            macro(element::Int64, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::UInt8::element_type())                                             \
        {                                                                                          \
            macro(element::UInt8, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::UInt32::element_type())                                            \
        {                                                                                          \
            macro(element::UInt32, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else if (et == element::UInt64::element_type())                                            \
        {                                                                                          \
            macro(element::UInt64, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw ngraph_error(err_msg);                                                           \
        }                                                                                          \
    }

// Turn off complaint suppression (see above)
#pragma clang diagnostic pop

shared_ptr<ngraph::runtime::CallFrame> ExternalFunction::make_call_frame()
{
    FunctionMap function_map;

    if (!m_is_compiled)
    {
        compile(function_map);
    }

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> temps;
    for (auto tv : m_temp_views)
    {
        auto& et = tv->get_tensor_view_type()->get_element_type();
        auto shape = tv->get_tensor_view_type()->get_shape();

#define M(T) temps.push_back(ngraph::runtime::make_tensor<T>(shape));
        DO_ON_ELEMENT_TYPE(
            et, "Internal error: tried to create temporary for unhandled element type", M);
#undef M
    }
    return make_shared<ngraph::runtime::cpu::CallFrame>(
        compiled_function, m_n_outputs, m_n_inputs, temps, callees);
}
