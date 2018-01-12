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
#include <cmath>
#include <iostream>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static string eigen_vector_format(const runtime::gpu::GPU_TensorViewWrapper& tvi)
{
    return "fmt::V{" + to_string(tvi.get_size()) + "}";
}

static string eigen_matrix_format(const ngraph::Shape& shape, const ngraph::Strides& strides)
{
    stringstream ss;
    ss << "fmt::M{{" << join(shape) << "}, {" << join(strides) << "}}";
    return ss.str();
}

void runtime::gpu::GPU_Emitter::EmitNop(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAdd(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] + "
          << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitDot(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    // const Shape& arg0_shape = args[0].get_shape();
    // const Shape& arg1_shape = args[1].get_shape();
    // if (arg0_shape.empty() || arg1_shape.empty())
    // {
    //     auto& first = (arg0_shape.empty() ? args[0] : args[1]);
    //     auto& second = (arg0_shape.empty() ? args[1] : args[0]);

    //     m_out << "{   // " << n->get_name() << "\n";
    //     m_out.indent++;
    //     m_out << emit_vector(out[0]) << "\n    = ";
    //     m_out << first.get_name() << "[0]\n    * " << emit_vector(second) << ";\n";
    //     m_out.indent--;
    //     m_out << "}\n";
    // }
    // else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1))
    // {
    //     m_out << "{   // " << n->get_name() << "\n";
    //     m_out.indent++;
    //     m_out << emit_vector(out[0]) << " << \n"
    //           << "    " << emit_vector(args[0]) << ".dot(" << emit_vector(args[1]) << ");\n";
    //     m_out.indent--;
    //     m_out << "}\n";
    // }
    // else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1))
    // {
    //     m_out << "{   // " << n->get_name() << "\n";
    //     m_out.indent++;
    //     m_out << emit_vector(out[0]) << " = \n"
    //           << "    " << emit_matrix(args[0]) << " * " << emit_vector(args[1]) << ";\n";
    //     m_out.indent--;
    //     m_out << "}\n";
    // }
    // else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2))
    // {
    //     // Emit an MKL SGEMM call if possible
    //     // clang-format off
    //     if (args[0].get_element_type() == element::f32)
    //     {
    //         m_out << "{   // " << n->get_name() << "\n";
    //         m_out.indent++;
    //         m_out << "cblas::cblas_sgemm("
    //            << "cblas::Layout::RowMajor, "
    //            << "cblas::Transpose::None, "
    //            << "cblas::Transpose::None, "
    //            << arg0_shape[0] << ", " << arg1_shape[1] << ", " << arg0_shape[1] << ",\n" <<
    //             "        1.0f, " << args[0].get_name() << ", " << max(1UL, arg0_shape[1]) << ", " << args[1].get_name() << ", " << max(1UL, arg1_shape[1]) << ", 0.0f,\n" <<
    //             "        " << out[0].get_name() << ", " << max(1UL, arg1_shape[1]) << ");\n";
    //         m_out.indent--;
    //         m_out << "}\n";
    //     }
    //     // clang-format on
    //     else
    //     {
    //         m_out << "{   // " << n->get_name() << "\n";
    //         m_out.indent++;
    // m_out << emit_matrix(out[0]) << " = \n"
    // << "    " << emit_concat(m_out,
    //                     args[0].get_element_type().c_type_string(),
    //                     arg_names,
    //                     out[0].get_name(),
    //                     arg_shapes,
    //                     result_shape,
    //                     axis);
}

void runtime::gpu::GPU_Emitter::EmitDivide(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    if (n->get_element_type().is_real() == false)
    {
        // Check for divide by zero for integer types only
        size_t element_count = args[1].get_size();
        m_out << "for (size_t i=0; i<" << element_count << "; i++)\n";
        m_out << "{\n";
        m_out << "    if (" << args.at(1).get_name()
              << "[i] == 0) throw std::runtime_error(\"integer divide by zero\");\n";
        m_out << "}\n";
    }
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] / "
          << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitEqual(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
          << "[i] == " << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitGreater(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << " xxx\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > "
          << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitGreaterEq(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
          << "[i] >= " << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitLess(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] < "
          << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitLessEq(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
          << "[i] <= " << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitLog(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = log(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitMaximum(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > "
          << args[1].get_name() << "[i] ? " << args[0].get_name() << "[i] : " << args[1].get_name()
          << "[i] ;\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitMinimum(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] < "
          << args[1].get_name() << "[i] ? " << args[0].get_name() << "[i] : " << args[1].get_name()
          << "[i] ;\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitNegative(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = -" << args[0].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitNotEqual(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
          << "[i] != " << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitSelect(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    // m_out << emit_array1d(out[0]) << " =\n"
    //       << "   " << emit_array1d(args[0]) << "\n"
    //       << "    .select(" << emit_array1d(args[1]) << ",\n"
    //       << "       " << emit_array1d(args[2]) << ");\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitSubtract(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] - "
          << args[1].get_name() << "[i];\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitBroadcast(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto broadcast = static_cast<const op::Broadcast*>(n);

    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    // kernel::emit_broadcast(m_out,
    //                        args[0].get_element_type().c_type_string(),
    //                        args[0].get_name(),
    //                        out[0].get_name(),
    //                        args[0].get_shape(),
    //                        out[0].get_shape(),
    //                        broadcast->get_broadcast_axes());
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitConvert(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto& result_element_type = out[0].get_element_type();

    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = (" << result_element_type.c_type_string() << ")("
          << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitConstant(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReshape(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto reshape = static_cast<const op::Reshape*>(n);
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    // kernel::emit_reshape(m_out,
    //                      args[0].get_element_type().c_type_string(),
    //                      args[0].get_name(),
    //                      out[0].get_name(),
    //                      args[0].get_shape(),
    //                      out[0].get_shape(),
    //                      reshape->get_input_order());
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitFunctionCall(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto function_call = static_cast<const op::FunctionCall*>(n);
    shared_ptr<Function> function = function_call->get_functions()[0];

    m_out << "{   // Call " << function->get_name() << "\n";
    m_out.indent++;
    generate_call(args, out, function);
    m_out.indent--;
    m_out << "}\n";
}

// TODO: This and other ops include comments/notes that
// we don't want to just copy-paste here. Figure out a better way
// or just point to ngvm/external_function.cpp with a note that
// the compiled version of these ops is intended to have semantics identical
// to what's seen there (for now atleast)

void runtime::gpu::GPU_Emitter::EmitReduce(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    // auto reduce = static_cast<const op::Reduce*>(n);
    // auto reduction_function = reduce->get_functions()[0];

    // auto reductee_shape = args[0].get_shape();

    // auto& f_result_element_type = out[0].get_element_type();
    // auto result_shape = out[0].get_shape();

    // auto& reduction_axes = reduce->get_reduction_axes();

    // // Trivial case: no reduction axes (this includes the scalar-reductee case).
    // if (reduction_axes.empty())
    // {
    //     m_out << "{   // " << n->get_name() << " 1\n";
    //     m_out.indent++;
    //     m_out << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
    //           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
    //     m_out.indent--;
    //     m_out << "}\n";
    // }
    // // Behavior for zero-size axes bears some explanation here. XLA's reduce
    // // operator provides an "gpu" element (usually, but not necessarily,
    // // an identity element) that it apparently *may* choose to insert anywhere
    // // in the reduction any number of times. For example, given:
    // //
    // //   reduce{{1,2,3},b,+)
    // //
    // // any of the following are valid reductions (I think!):
    // //
    // //   b+(b+1+2)+3
    // //   b+(1+(2+3))
    // //   (1+2)+3 (I think!)
    // //
    // // etc. Here we will choose never to instantiate the gpu element, which
    // // works well with Eigen's default behavior for non-zero-length axes. The
    // // exceptional case is when we reduce on a zero-length axis. In this case,
    // // Eigen's default behavior is to put a zero in the output,  which is not
    // // what we want, so we detect that case here and override with a copy
    // // instruction (for reduce-to-scalar) or a broadcast (for reduce-to-vector)
    // // from the gpu element.
    // //
    // // What I'm actually not sure about is whether the identity element is
    // // required to appear at least once. If so, this will need to be reworked,
    // // assuming we actually want to mimic XLA's semantics that closely, which
    // // we may not.
    // else if ((reductee_shape.size() == 1 && reduction_axes == AxisSet{0}) ||
    //          (reductee_shape.size() == 2 && reduction_axes == AxisSet{0, 1}))
    // {
    //     if (reductee_shape.at(0) == 0 || (reductee_shape.size() == 2 && reductee_shape.at(1) == 0))
    //     {
    //         m_out << "{   // " << n->get_name() << " 2\n";
    //         m_out.indent++;
    //         m_out << "memcpy(" << out[0].get_name() << ", " << args[1].get_name() << ", "
    //               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
    //         m_out.indent--;
    //         m_out << "}\n";
    //     }
    //     else
    //     {
    //         m_out << "{   // " << n->get_name() << " 3\n";
    //         m_out.indent++;
    //         string type = f_result_element_type.c_type_string();
    //         m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
    //         m_out.indent++;
    //         m_out << "\n";
    //         m_out << type << " result;\n";
    //         m_out << "void* args[] = {&x, &y};\n";
    //         m_out << "void* out[] = {&result};\n";
    //         m_out << reduction_function->get_name() << "(args, out);\n";
    //         m_out << "return result;\n";
    //         m_out.indent--;
    //         m_out << "};\n";
    //         m_out << emit_array1d(out[0]) << " =\n"
    //               << "    " << emit_array1d(args[0]) << ".redux(f);\n";
    //         m_out.indent--;
    //         m_out << "}\n";
    //     }
    // }
    // else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{1})
    // {
    //     if (reductee_shape.at(1) == 0)
    //     {
    //         m_out << "{   // " << n->get_name() << " 4\n";
    //         m_out.indent++;
    //         m_out << emit_array1d(out[0]) << " =\n"
    //               << "    " << emit_array1d(args[1]) << "(0, 0);\n";
    //         m_out.indent--;
    //         m_out << "}\n";
    //     }
    //     else
    //     {
    //         // shared_ptr<CallFrame> cf =
    //         //     dynamic_pointer_cast<CallFrame>(external->make_call_frame());
    //         // ef->get_callees().emplace_back(cf);

    //         m_out << "{   // " << n->get_name() << " 5\n";
    //         m_out.indent++;
    //         string type = f_result_element_type.c_type_string();
    //         m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
    //         m_out.indent++;
    //         m_out << "\n";
    //         m_out << type << " result;\n";
    //         m_out << "void* args[] = {&x, &y};\n";
    //         m_out << "void* out[] = {&result};\n";
    //         m_out << reduction_function->get_name() << "(args, out);\n";
    //         m_out << "return result;\n";
    //         m_out.indent--;
    //         m_out << "};\n";
    //         m_out << emit_vector(out[0]) << " =\n"
    //               << "        " << emit_matrix(args[0]) << ".rowwise().redux(f);\n";
    //         m_out.indent--;
    //         m_out << "}\n";
    //     }
    // }
    // else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{0})
    // {
    //     if (reductee_shape.at(0) == 0)
    //     {
    //         m_out << "{   // " << n->get_name() << " 6\n";
    //         m_out.indent++;
    //         m_out << emit_array1d(out[0]) << " =\n"
    //               << "    " << emit_array1d(args[1]) << "(0, 0);\n";
    //         m_out.indent--;
    //         m_out << "}\n";
    //     }
    //     else
    //     {
    //         m_out << "{   // " << n->get_name() << " 7\n";
    //         m_out.indent++;
    //         string type = f_result_element_type.c_type_string();
    //         m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
    //         m_out.indent++;
    //         m_out << "\n";
    //         m_out << type << " result;\n";
    //         m_out << "void* args[] = {&x, &y};\n";
    //         m_out << "void* out[] = {&result};\n";
    //         m_out << reduction_function->get_name() << "(args, out);\n";
    //         m_out << "return result;\n";
    //         m_out.indent--;
    //         m_out << "};\n";
    //         m_out << emit_vector(out[0]) << " =\n"
    //               << "    " << emit_matrix(args[0]) << ".colwise().redux(f);\n";
    //         m_out.indent--;
    //         m_out << "}\n";
    //     }
    // }
    // else
    // {
    //     string type = f_result_element_type.c_type_string();
    //     m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
    //     m_out.indent++;
    //     m_out << "\n";
    //     m_out << type << " result;\n";
    //     m_out << "void* args[] = {&x, &y};\n";
    //     m_out << "void* out[] = {&result};\n";
    //     m_out << reduction_function->get_name() << "(args, out);\n";
    //     m_out << "return result;\n";
    //     m_out.indent--;
    //     m_out << "};\n";

    //     m_out << "kernel::reduce<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    //     m_out << "               " << args[1].get_name() << ",\n";
    //     m_out << "               " << out[0].get_name() << ",\n";
    //     m_out << "               {" << join(args[0].get_shape()) << "},\n";
    //     m_out << "               {" << join(out[0].get_shape()) << "},\n";
    //     m_out << "               {" << join(reduce->get_reduction_axes()) << "},\n";
    //     m_out << "               f);\n";
    // }
}

void runtime::gpu::GPU_Emitter::EmitSign(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = (0 < " << args[0].get_name() << "[i]) - ("
          << args[0].get_name() << "[i] < 0);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitSlice(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    const op::Slice* slice = static_cast<const op::Slice*>(n);

    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    // kernel::emit_slice(m_out,
    //                    args[0].get_element_type().c_type_string(),
    //                    args[0].get_name(),
    //                    out[0].get_name(),
    //                    args[0].get_shape(),
    //                    out[0].get_shape(),
    //                    slice->get_lower_bounds(),
    //                    slice->get_upper_bounds(),
    //                    slice->get_strides());
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitSum(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    const op::Sum* sum = static_cast<const op::Sum*>(n);
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    // kernel::emit_sum(m_out,
    //                  args[0].get_element_type().c_type_string(),
    //                  args[0].get_name(),
    //                  out[0].get_name(),
    //                  args[0].get_shape(),
    //                  out[0].get_shape(),
    //                  sum->get_reduction_axes());
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitExp(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = exp(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitSin(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = sin(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitSinh(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = sinh(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitCos(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = cos(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitCosh(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = cosh(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitTan(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = tan(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitTanh(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i=0; i<" << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = tanh(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitAsin(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = asin(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitAcos(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = acos(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitAtan(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = atan(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitPower(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = pow(" << args[0].get_name() << "[i], "
          << args[1].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitReplaceSlice(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto replace_slice = static_cast<const op::Slice*>(n);
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    // kernel::emit_replace_slice(m_out,
    //                            args[0].get_element_type().c_type_string(),
    //                            args[0].get_name(),
    //                            args[1].get_name(),
    //                            out[0].get_name(),
    //                            args[1].get_shape(),
    //                            out[0].get_shape(),
    //                            replace_slice->get_lower_bounds(),
    //                            replace_slice->get_upper_bounds(),
    //                            replace_slice->get_strides());
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitOneHot(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto oh = static_cast<const op::OneHot*>(n);

    auto arg_rank = args[0].get_shape().size();

    size_t bounds = out[0].get_shape()[oh->get_one_hot_axis()];

    if (arg_rank == 0)
    {
        m_out << "{   // " << n->get_name() << " 1\n";
        m_out.indent++;

        // m_out << emit_vector(out[0], "out_vector") << ";\n";

        m_out << "out_vector.setZero();\n"
              << ""
              // << "auto pos_raw = " << emit_vector(args[0]) << "(0, 0);\n"
              << "if (floor(pos_raw) != pos_raw)\n"
              << "{\n";
        m_out.indent++;
        m_out << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
        m_out.indent--;
        m_out << "}\n";

        m_out << "size_t pos = pos_raw;\n"
              << "if (pos >= " << bounds << ")\n";

        m_out << "{\n";
        m_out.indent++;
        m_out << "throw(std::range_error(\"One-hot: value is out of category range\"));\n";
        m_out.indent--;
        m_out << "}\n";

        m_out << "out_vector(pos, 0) = 1;\n";

        m_out.indent--;
        m_out << "}\n";
    }
    else if (arg_rank == 1)
    {
        m_out << "{   // " << n->get_name() << " 1\n";
        m_out.indent++;

        // m_out << emit_vector(args[0], "arg_vector") << ";\n";

        // m_out << emit_matrix(out[0], "out_vector") << ";\n";
        m_out << "out_vector.setZero();\n";

        m_out << "for (size_t i = 0; i < " << args[0].get_shape()[0] << "; i++)\n"
              << "{\n";
        m_out.indent++;

        m_out << "auto pos_raw = arg_vector(i, 0);\n";

        m_out << "if (floor(pos_raw) != pos_raw)\n"
              << "{\n";
        m_out.indent++;
        m_out << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
        m_out.indent--;
        m_out << "}\n";

        m_out << "size_t pos = pos_raw;\n";
        m_out << "bool found = false;\n";

        m_out << "if (pos >= " << bounds << ")\n"
              << "{\n";
        m_out.indent++;
        m_out << "throw(std::range_error(\"One-hot: value is out of category range\"));\n";
        m_out.indent--;
        m_out << "}\n";

        m_out << "out_vector" << (oh->get_one_hot_axis() == 0 ? "(pos, i)" : "(i, pos)")
              << " = 1;\n";

        m_out.indent--;
        m_out << "}\n";

        m_out.indent--;
        m_out << "}\n";
    }
    // Other cases are not handled yet.
    else
    {
        m_out << "kernel::one_hot<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        m_out << "                   " << out[0].get_name() << ",\n";
        m_out << "                   {" << join(args[0].get_shape()) << "},\n";
        m_out << "                   {" << join(out[0].get_shape()) << "},\n";
        m_out << "                   " << oh->get_one_hot_axis() << ");\n";
    }
}

void runtime::gpu::GPU_Emitter::EmitCeiling(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    size_t element_count = out[0].get_size();

    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = ceil(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitFloor(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    size_t element_count = out[0].get_size();
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = floor(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitSqrt(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    size_t element_count = out[0].get_size();
    m_out << "#pragma omp parallel for\n";
    m_out << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = sqrt(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitConvolution(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto convolution = static_cast<const op::Convolution*>(n);

    auto arg0_shape = args[0].get_shape();
    auto arg1_shape = args[1].get_shape();
    auto result_shape = out[0].get_shape();

    m_out << "kernel::convolution<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    m_out << "                         " << args[1].get_name() << ",\n";
    m_out << "                         " << out[0].get_name() << ",\n";
    m_out << "                         {" << join(arg0_shape) << "},\n";
    m_out << "                         {" << join(arg1_shape) << "},\n";
    m_out << "                         {" << join(result_shape) << "},\n";
    m_out << "                         {" << join(convolution->get_window_movement_strides())
          << "},\n";
    m_out << "                         {" << join(convolution->get_window_dilation_strides())
          << "},\n";
    m_out << "                         {" << join(convolution->get_padding_below()) << "},\n";
    m_out << "                         {" << join(convolution->get_padding_above()) << "});\n";
}

void runtime::gpu::GPU_Emitter::EmitNot(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    m_out << "kernel::logical_not(" << args[0].get_name() << ",\n"
          << "                    " << out[0].get_name() << ",\n"
          << "                    " << out[0].get_size() << ");\n";
}

void runtime::gpu::GPU_Emitter::EmitMaxPool(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto max_pool = static_cast<const op::MaxPool*>(n);

    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    m_out << "kernel::max_pool<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    m_out << "                 " << out[0].get_name() << ",\n";
    m_out << "                 {" << join(arg_shape) << "},\n";
    m_out << "                 {" << join(result_shape) << "},\n";
    m_out << "                 {" << join(max_pool->get_window_shape()) << "},\n";
    m_out << "                 {" << join(max_pool->get_window_movement_strides()) << "});\n";
}

void runtime::gpu::GPU_Emitter::EmitReverse(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    auto reverse = static_cast<const op::Reverse*>(n);

    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    m_out << "kernel::reverse<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    m_out << "                " << out[0].get_name() << ",\n";
    m_out << "                {" << join(arg_shape) << "},\n";
    m_out << "                {" << join(result_shape) << "},\n";
    m_out << "                {" << join(reverse->get_reversed_axes()) << "});\n";
}

//------------------------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------------------------

void runtime::gpu::GPU_Emitter::generate_call(
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out,
    shared_ptr<Function> function)
{
    vector<string> input_names;
    vector<string> output_names;

    for (const runtime::gpu::GPU_TensorViewWrapper& input : args)
    {
        input_names.push_back(input.get_name());
    }

    for (const runtime::gpu::GPU_TensorViewWrapper& output : out)
    {
        output_names.push_back(output.get_name());
    }

    m_out << "void* args[] =\n{";
    m_out.indent++;
    m_out << "\n" << join(input_names, ",\n");
    m_out.indent--;
    m_out << "\n};\n";

    m_out << "void* out[] =\n{";
    m_out.indent++;
    m_out << "\n" << join(output_names, ",\n");
    m_out.indent--;
    m_out << "\n};\n";

    m_out << "\n";
    m_out << function->get_name() << "(args, out);\n";
}

static string format_name(const string& name)
{
    string rc;
    if (!name.empty())
    {
        rc = " " + name;
    }
    return rc;
}

string runtime::gpu::GPU_Emitter::emit_vector(const runtime::gpu::GPU_TensorViewWrapper& tvi,
                                              const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_element_type();
    ss << "EigenVector<" << et.c_type_string() << ">" << format_name(name) << "(" << tvi.get_name()
       << ", " << eigen_vector_format(tvi) << ")";
    return ss.str();
}

string runtime::gpu::GPU_Emitter::emit_array1d(const runtime::gpu::GPU_TensorViewWrapper& tvi,
                                               const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_element_type();
    ss << "EigenArray1d<" << et.c_type_string() << ">" << format_name(name) << "(" << tvi.get_name()
       << ", " << eigen_vector_format(tvi) << ")";
    return ss.str();
}

string runtime::gpu::GPU_Emitter::emit_matrix(const runtime::gpu::GPU_TensorViewWrapper& tvi,
                                              const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_element_type();
    ss << "EigenMatrix<" << et.c_type_string() << ">" << format_name(name) << "(" << tvi.get_name()
       << ", " << eigen_matrix_format(tvi.get_shape(), tvi.get_strides()) << ")";
    return ss.str();
}

void runtime::gpu::GPU_Emitter::EmitAbs(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    //     m_out << "{   // " << n->get_name() << "\n";
    //     m_out.indent++;
    // #if PREFER_EIGEN == 1
    //     m_out << emit_array1d(out[0]) << " =\n";
    //     m_out << "Eigen::abs(" << emit_array1d(args[0]) << ");\n";
    // #else
    //     m_out << "#pragma omp parallel for\n";
    //     m_out << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    //     m_out << "{\n";
    //     m_out << "    " << out[0].get_name() << "[i] = std::abs(" << args[0].get_name() << "[i]);\n";
    //     m_out << "}\n";
    // #endif
    //     m_out.indent--;
    //     m_out << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitConcat(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMultiply(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}
