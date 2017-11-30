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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/file_util.hpp"
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
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
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
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/cpu/call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/emitter.hpp"
#include "ngraph/runtime/cpu/external_function.hpp"
#include "ngraph/runtime/utils.hpp"

using namespace std;
using namespace ngraph::runtime::cpu;

static const std::string s_output_dir = "cpu_codegen";

class StaticInitializers
{
public:
    StaticInitializers() { ngraph::file_util::remove_directory(s_output_dir); }
};

static StaticInitializers s_static_initializers;

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
    {TI(ngraph::op::Power), &Emitter::EmitPower},
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
    {TI(ngraph::op::ReplaceSlice), &Emitter::EmitReplaceSlice},
};

ExternalFunction::ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                   bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
    , m_compiled_function(nullptr)
{
}

void ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    string function_name = m_function->get_name();
    string dump_filename = file_util::path_join(s_output_dir, function_name + "_ops.txt");

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::TopologicalSort>();
    // For now, just make everyone row-major.
    pass_manager.register_pass<pass::AssignLayout<DenseTensorViewLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.register_pass<pass::MemoryLayout>(64);
    pass_manager.register_pass<pass::DumpSorted>(dump_filename);
    pass_manager.run_passes(m_function);

    // Now we build the TU
    Emitter emitter;
    codegen::CodeWriter& TU = emitter.get_code_writer();

    TU +=
        R"(// Generated by the NGraph CPU backend
#include <cmath>

#include <Eigen/Dense>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/eigen_utils.hpp"

using namespace ngraph::runtime::cpu::eigen;

)";
    string pch_header_source = TU.get_code();

    // The "dso_handle" symbol is required by __cxa_atexit()
    // which is enabled because the JIT uses it as the default mechanism
    // to register cleanup handlers. We use it, and not atexit(), because
    // atexit() happens too late, when the JIT is no longer alive

    TU << "void *__dso_handle = 0;\n\n";

    TU << "// Declare all functions\n";
    for (shared_ptr<Function> f : pass_manager.get_state().get_functions())
    {
        TU << "extern \"C\" void " << f->get_name() << "(void** inputs, void** outputs);\n";
    }
    TU << "\n";

    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        TU << "extern \"C\" void " << current_function->get_name();
        TU << "(void** inputs, void** outputs)\n";
        TU << "{\n";
        TU.indent++;

        bool temporaries_used = false;
        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            if (node->liveness_new_list.size() > 0)
            {
                temporaries_used = true;
                break;
            }
        }
        if (temporaries_used)
        {
            size_t temp_pool_size = current_function->get_temporary_pool_size();
            TU << "// Allocate the memory pool\n";
            TU << "ngraph::runtime::AlignedBuffer memory_handler(" << temp_pool_size << ", "
               << ngraph::runtime::cpu::alignment << ");\n";
            TU << "\n";

            TU << "// Define temporary tensors\n";
            for (shared_ptr<Node> node : current_function->get_ordered_ops())
            {
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    TU << tensor->get_element_type().c_type_string() << "* " << tensor->get_name()
                       << " = (" << tensor->get_element_type().c_type_string()
                       << "*)(memory_handler.get_ptr(" << tensor->get_pool_offset() << "));\n";
                }
            }
            TU << "\n";
        }

        TU << "// Define inputs\n";
        size_t arg_index = 0;
        for (shared_ptr<op::Parameter> param : current_function->get_parameters())
        {
            for (const descriptor::Output& output : param->get_outputs())
            {
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                const element::Type& et = tv->get_tensor_view_type()->get_element_type();
                string type = et.c_type_string();
                TU << "" << type << "* " << tv->get_tensor().get_name() << " = static_cast<" << type
                   << "*>(inputs[" << arg_index << "]);\n";
                arg_index++;
            }
        }
        TU << "\n";

        TU << "// Define outputs\n";
        size_t output_index = 0;
        for (const descriptor::Output& output : current_function->get_result()->get_outputs())
        {
            shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
            const element::Type& et = tv->get_tensor_view_type()->get_element_type();
            string type = et.c_type_string();
            TU << type << "* " << tv->get_tensor().get_name() << " = static_cast<" << type
               << "*>(outputs[" << output_index << "]);\n";
            output_index++;
        }
        TU << "\n";

        for (shared_ptr<Node> node : current_function->get_ordered_ops())
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
                in.push_back({0, tv});
            }
            std::vector<TensorViewInfo> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                auto tv = output.get_tensor_view();
                out.push_back({0, tv});
            }
            handler->second(&emitter, node.get(), in, out);
        }

        TU.indent--;

        // End TU
        TU += "}\n\n";
    }

    // TODO: Cleanup and make this a utility function

    file_util::make_directory(s_output_dir);
    string filename = file_util::path_join(s_output_dir, function_name + "_codegen.cpp");
    ofstream out(filename);
    string code = TU.get_code();
    out << code;
    out.close();

    compiler.reset(new codegen::Compiler());
    execution_engine.reset(new codegen::ExecutionEngine());

    compiler->set_precompiled_header_source(pch_header_source);

    auto llvm_module = compiler->compile(code);

    if (llvm_module == nullptr)
    {
        throw runtime_error("function failed to compile");
    }
    execution_engine->add_module(llvm_module);
    execution_engine->finalize();
    m_compiled_function = execution_engine->find_function<EntryPoint_t>(function_name);
    assert(m_compiled_function);

    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

shared_ptr<ngraph::runtime::CallFrame> ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }

    return make_shared<ngraph::runtime::cpu::CallFrame>(shared_from_this(), m_compiled_function);
}
