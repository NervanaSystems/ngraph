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
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/pad.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/reduce_window.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/select_and_scatter.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_emitters.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

#define PREFER_EIGEN 0

static bool s_use_ref_kernels = (std::getenv("NGRAPH_CPU_USE_REF_KERNELS") != nullptr);

static string eigen_vector_format(const runtime::cpu::TensorViewWrapper& tvi)
{
    return "fmt::V{" + to_string(tvi.get_size()) + "}";
}

static string eigen_matrix_format(const ngraph::Shape& shape, const ngraph::Strides& strides)
{
    stringstream ss;
    ss << "fmt::M{{" << join(shape) << "}, {" << join(strides) << "}}";
    return ss.str();
}

// Mapping from POD types to MKLDNN data types
// An empty string implies the corresponding MKLDNN data type
// is not supported
static const unordered_map<string, const string> mkldnn_data_type_map{
    {"char", "memory::data_type::s8"},
    {"float", "memory::data_type::f32"},
    {"double", ""},
    {"int8_t", "memory::data_type::s8"},
    {"int16_t", "memory::data_type::s16"},
    {"int32_t", "memory::data_type::s32"},
    {"int64_t", ""},
    {"uint8_t", "memory::data_type::u8"},
    {"uint16_t", ""},
    {"uint32_t", ""},
    {"uint64_t", ""}};

static const string& get_mkldnn_data_type(const string& type)
{
    auto it = mkldnn_data_type_map.find(type);
    if (it == mkldnn_data_type_map.end() || it->second.empty())
        throw ngraph_error("No MKLDNN data type exists for the given element type");
    return it->second;
}

void runtime::cpu::CPU_Emitter::EmitMKLDNNPreamble(codegen::CodeWriter& writer)
{
    writer << "using namespace mkldnn;\n";
    writer << "auto cpu_engine = engine(engine::cpu, 0);\n";
    writer.emitted_mkldnn_preamble = true;
}

void runtime::cpu::CPU_Emitter::EmitNop(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
}

void runtime::cpu::CPU_Emitter::EmitAdd(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    // TODO: Audit all uses of Add and fix this to use
    // the right alignment instead of Eigen::Unaligned
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << "Eigen::Map<Eigen::Array<" << out[0].get_element_type().c_type_string() << ", "
           << out[0].get_size() << ", 1>, Eigen::Unaligned> out(" << out[0].get_name() << ");\n";
    writer << "Eigen::Map<Eigen::Array<" << args[0].get_element_type().c_type_string() << ", "
           << args[0].get_size() << ", 1>, Eigen::Unaligned> arg0(" << args[0].get_name() << ");\n";
    writer << "Eigen::Map<Eigen::Array<" << args[1].get_element_type().c_type_string() << ", "
           << args[1].get_size() << ", 1>, Eigen::Unaligned> arg1(" << args[1].get_name() << ");\n";
    writer << "out = arg0 + arg1;\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] + "
           << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitDot(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    const Shape& arg0_shape = args[0].get_shape();
    const Shape& arg1_shape = args[1].get_shape();
    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& first = (arg0_shape.empty() ? args[0] : args[1]);
        auto& second = (arg0_shape.empty() ? args[1] : args[0]);

        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_vector(out[0]) << "\n    = ";
        writer << first.get_name() << "[0]\n    * " << emit_vector(second) << ";\n";
        writer.indent--;
        writer << "}\n";
    }
    else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1))
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_vector(out[0]) << " << \n"
               << "    " << emit_vector(args[0]) << ".dot(" << emit_vector(args[1]) << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1))
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_vector(out[0]) << " = \n"
               << "    " << emit_matrix(args[0]) << " * " << emit_vector(args[1]) << ";\n";
        writer.indent--;
        writer << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2))
    {
        // Emit an MKL SGEMM call if possible
        // clang-format off
        if (args[0].get_element_type() == element::f32)
        {
            writer << "{   // " << n->get_name() << "\n";
            writer.indent++;
            writer << "cblas::cblas_sgemm("
               << "cblas::Layout::RowMajor, "
               << "cblas::Transpose::None, "
               << "cblas::Transpose::None, "
               << arg0_shape[0] << ", " << arg1_shape[1] << ", " << arg0_shape[1] << ",\n" <<
                "        1.0f, " << args[0].get_name() << ", " << max(1UL, arg0_shape[1]) << ", " << args[1].get_name() << ", " << max(1UL, arg1_shape[1]) << ", 0.0f,\n" <<
                "        " << out[0].get_name() << ", " << max(1UL, arg1_shape[1]) << ");\n";
            writer.indent--;
            writer << "}\n";
        }
        // clang-format on
        else
        {
            writer << "{   // " << n->get_name() << "\n";
            writer.indent++;
            writer << emit_matrix(out[0]) << " = \n"
                   << "    " << emit_matrix(args[0]) << " * " << emit_matrix(args[1]) << ";\n";
            writer.indent--;
            writer << "}\n";
        }
    }
    else
    {
        const ngraph::op::Dot* dot = static_cast<const ngraph::op::Dot*>(n);

        writer << "kernel::dot(" << args[0].get_name() << ",\n";
        writer << "            " << args[1].get_name() << ",\n";
        writer << "            " << out[0].get_name() << ",\n";
        writer << "            {" << join(args[0].get_shape()) << "},\n";
        writer << "            {" << join(args[1].get_shape()) << "},\n";
        writer << "            {" << join(out[0].get_shape()) << "},\n";
        writer << "            " << dot->get_reduction_axes_count() << ");\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitMultiply(codegen::CodeWriter& writer,
                                             const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "   " << emit_array1d(args[0]) << " *\n"
           << "   " << emit_array1d(args[1]) << ";\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] * "
           << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitGetOutputElement(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto get_tuple_element = static_cast<const op::GetOutputElement*>(n);

    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
    writer << "memcpy(" << out[0].get_name() << ", " << args[get_tuple_element->get_n()].get_name()
           << ", " << out[0].get_size() * out[0].get_element_type().size() << ");\n";
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitTuple(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
    for (size_t i = 0; i < args.size(); ++i)
    {
        writer << "memcpy(" << out.at(i).get_name() << ", " << args.at(i).get_name() << ", "
               << out[i].get_size() * out[i].get_element_type().size() << ");\n";
    }
    writer.indent--;
    writer += "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAbs(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n";
    writer << "Eigen::abs(" << emit_array1d(args[0]) << ");\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = std::abs(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitConcat(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto result_shape = out[0].get_shape();

#if PREFER_EIGEN == 1
    if (result_shape.size() == 1)
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_vector(out[0], "out_vector") << ";\n";

        size_t concat_pos = 0;
        for (size_t i = 0; i < args.size(); i++)
        {
            writer << "out_vector.segment(" << concat_pos << ", " << args[i].get_shape().at(0)
                   << ") << " << emit_vector(args[i]) << ";\n";
            concat_pos += args[i].get_shape().at(0);
        }
        writer.indent--;
        writer << "}\n";
    }
    else if (result_shape.size() == 2)
    {
        auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_matrix(out[0], "out_matrix") << ";\n";

        size_t concat_pos[2]{0, 0};
        for (size_t i = 0; i < args.size(); i++)
        {
            auto& arg_shape = args[i].get_shape();

            writer << "out_matrix.block(" << concat_pos[0] << ", " << concat_pos[1] << ", "
                   << arg_shape.at(0) << ", " << arg_shape.at(1) << ") << " << emit_matrix(args[i])
                   << ";\n";

            concat_pos[axis] += arg_shape.at(axis);
        }

        writer.indent--;
        writer << "}\n";
    }
    else
    {
        if (s_use_ref_kernels)
        {
            auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

            std::vector<std::string> arg_names;
            std::vector<std::string> arg_shape_strings;

            for (auto arg : args)
            {
                arg_names.push_back(arg.get_name());
                arg_shape_strings.push_back("{" + join(arg.get_shape()) + "}");
            }

            writer << "kernel::concat<" << out[0].get_type() << ">({" << join(arg_names) << "},\n";
            writer << "                         " << out[0].get_name() << ",\n";
            writer << "                         {" << join(arg_shape_strings) << "},\n";
            writer << "                         {" << join(result_shape) << "},\n";
            writer << "                         " << axis << ");\n";
        }
        else
        {
            auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

            std::vector<std::string> arg_names;
            std::vector<Shape> arg_shapes;

            for (auto arg : args)
            {
                arg_names.push_back(arg.get_name());
                arg_shapes.push_back(arg.get_shape());
            }

            kernel::emit_concat(writer,
                                args[0].get_element_type().c_type_string(),
                                arg_names,
                                out[0].get_name(),
                                arg_shapes,
                                result_shape,
                                axis);
        }
    }
#else
    auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

    std::vector<std::string> arg_names;
    std::vector<Shape> arg_shapes;

    for (auto arg : args)
    {
        arg_names.push_back(arg.get_name());
        arg_shapes.push_back(arg.get_shape());
    }

    kernel::emit_concat(writer,
                        args[0].get_element_type().c_type_string(),
                        arg_names,
                        out[0].get_name(),
                        arg_shapes,
                        result_shape,
                        axis);
#endif
}

void runtime::cpu::CPU_Emitter::EmitDivide(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
    if (n->get_element_type().is_real() == false)
    {
        // Check for divide by zero for integer types only
        size_t element_count = args[1].get_size();
        writer << "for (size_t i=0; i<" << element_count << "; i++)\n";
        writer << "{\n";
        writer << "    if (" << args.at(1).get_name()
               << "[i] == 0) throw std::runtime_error(\"integer divide by zero\");\n";
        writer << "}\n";
    }
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << " /\n"
           << "    " << emit_array1d(args[1]) << ";\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] / "
           << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitEqual(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    (" << emit_array1d(args[0]) << " ==\n"
           << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
           << "[i] == " << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitGreater(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << " xxx\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    (" << emit_array1d(args[0]) << " >\n"
           << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > "
           << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitGreaterEq(codegen::CodeWriter& writer,
                                              const ngraph::Node* n,
                                              const vector<runtime::cpu::TensorViewWrapper>& args,
                                              const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    (" << emit_array1d(args[0]) << " >=\n"
           << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
           << "[i] >= " << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitLess(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    (" << emit_array1d(args[0]) << " <\n"
           << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] < "
           << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitLessEq(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    (" << emit_array1d(args[0]) << " <=\n"
           << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
           << "[i] <= " << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitLog(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    Eigen::log(" << emit_array1d(args[0]) << ");\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = log(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitMaximum(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "        " << emit_array1d(args[0]) << ".max(\n"
           << "        " << emit_array1d(args[1]) << ");\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > "
           << args[1].get_name() << "[i] ? " << args[0].get_name() << "[i] : " << args[1].get_name()
           << "[i] ;\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitMinimum(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".min(\n"
           << "    " << emit_array1d(args[1]) << ");\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] < "
           << args[1].get_name() << "[i] ? " << args[0].get_name() << "[i] : " << args[1].get_name()
           << "[i] ;\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitNegative(codegen::CodeWriter& writer,
                                             const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    -" << emit_array1d(args[0]) << ";\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = -" << args[0].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitNotEqual(codegen::CodeWriter& writer,
                                             const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    (" << emit_array1d(args[0]) << " !=\n"
           << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
           << "[i] != " << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSelect(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "   " << emit_array1d(args[0]) << "\n"
           << "    .select(" << emit_array1d(args[1]) << ",\n"
           << "       " << emit_array1d(args[2]) << ");\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] ? "
           << args[1].get_name() << "[i] : " << args[2].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSubtract(codegen::CodeWriter& writer,
                                             const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << " -\n"
           << "    " << emit_array1d(args[1]) << ";\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] - "
           << args[1].get_name() << "[i];\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitBroadcast(codegen::CodeWriter& writer,
                                              const ngraph::Node* n,
                                              const vector<runtime::cpu::TensorViewWrapper>& args,
                                              const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto broadcast = static_cast<const op::Broadcast*>(n);

    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    if (broadcast->get_broadcast_axes().empty())
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (arg_shape.size() == 0)
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_array1d(out[0]) << " =\n"
               << "    " << emit_array1d(args[0]) << "(0, 0);\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (arg_shape.size() == 1 && result_shape.size() == 2)
    {
        if (broadcast->get_broadcast_axes() == AxisSet{1})
        {
            writer << "{   // " << n->get_name() << "\n";
            writer.indent++;
            writer << emit_matrix(out[0]) << ".colwise() =\n"
                   << "    " << emit_vector(args[0]) << ";\n";
            writer.indent--;
            writer << "}\n";
        }
        else if (broadcast->get_broadcast_axes() == AxisSet{0})
        {
            writer << "{   // " << n->get_name() << "\n";
            writer.indent++;

            writer << "Eigen::Map<Eigen::Matrix<" << out[0].get_element_type().c_type_string()
                   << ", " << join(out[0].get_shape())
                   << ", Eigen::RowMajor>, Eigen::Aligned64, Eigen::Stride<"
                   << join(out[0].get_strides()) << ">> out(" << out[0].get_name() << ");\n";
            writer << "Eigen::Map<Eigen::Matrix<" << args[0].get_element_type().c_type_string()
                   << ", 1, " << args[0].get_size()
                   << ", Eigen::RowMajor>, Eigen::Aligned64, Eigen::Stride<" << args[0].get_size()
                   << ", 1>> arg0(" << args[0].get_name() << ");\n";
            writer << "out = arg0.replicate<" << out[0].get_shape().at(0) << ", 1>();\n";

            writer.indent--;
            writer << "}\n";
        }
        else
        {
            throw ngraph_error(
                "Internal error: axis set for vector-matrix broadcast is neither {0} nor "
                "{1}");
        }
    }
    else
    {
        writer << "kernel::broadcast<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        writer << "                         " << out[0].get_name() << ",\n";
        writer << "                         {" << join(arg_shape) << "},\n";
        writer << "                         {" << join(result_shape) << "},\n";
        writer << "                         {" << join(broadcast->get_broadcast_axes()) << "});\n";
    }
#else
    kernel::emit_broadcast(writer,
                           args[0].get_element_type().c_type_string(),
                           args[0].get_name(),
                           out[0].get_name(),
                           args[0].get_shape(),
                           out[0].get_shape(),
                           broadcast->get_broadcast_axes());
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitConvert(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto& result_element_type = out[0].get_element_type();

    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << "\n"
           << "    .template cast<" << result_element_type.c_type_string() << ">();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = (" << result_element_type.c_type_string()
           << ")(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitConstant(codegen::CodeWriter& writer,
                                             const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
}

void runtime::cpu::CPU_Emitter::EmitReshape(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto reshape = static_cast<const op::Reshape*>(n);
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    auto arg_shape = args[0].get_shape();
    auto arg_rank = arg_shape.size();

    auto result_shape = out[0].get_shape();
    auto& result_element_type = out[0].get_element_type();

    auto input_order = reshape->get_input_order();

    bool same_layout = is_sorted(input_order.begin(), input_order.end());

    size_t result_shape_product = 1;
    for (auto i : result_shape)
    {
        result_shape_product *= i;
    }

    // If there is no layout change or we are just going from 1^n to 1^m or a zero-size tensor,
    //  we can just copy.
    if (same_layout || result_shape_product < 2)
    {
        writer << "{   // " << n->get_name() << " 1\n";
        writer.indent++;
        writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    // If there *is* a layout change in the 2D case, we transpose the input.
    else if (arg_rank == 2)
    {
        // Emit an MKL transpose call if possible
        // clang-format off
        if (result_element_type == ngraph::element::f32)
        {
            writer << "{   // " << n->get_name() << " 2\n";
            writer.indent++;
            writer << "mkl::MKL_Somatcopy('R', 'T', " << to_string(arg_shape[0]) << ",\n" <<
                "                   " << to_string(arg_shape[1]) << ", 1.0f,\n" <<
                "                   " << args[0].get_name() << ", "
                << to_string(arg_shape[1]) << ",\n" <<
                "                   " << out[0].get_name()
                << ", " << to_string(arg_shape[0]) << ");\n";
                writer.indent--;
                writer << "}\n";
        }
        // clang-format on
        else
        {
            writer << "{   // " << n->get_name() << " 3\n";
            writer.indent++;
            writer << emit_matrix(out[0]) << " =\n"
                   << "        " << emit_matrix(args[0]) << ".transpose();\n";
            writer.indent--;
            writer << "}\n";
        }
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error(
            "Axis permutation in reshape is not implemented yet for tensors with rank>2");
    }
#else
    kernel::emit_reshape(writer,
                         args[0].get_element_type().c_type_string(),
                         args[0].get_name(),
                         out[0].get_name(),
                         args[0].get_shape(),
                         out[0].get_shape(),
                         reshape->get_input_order());
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitFunctionCall(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto function_call = static_cast<const op::FunctionCall*>(n);
    shared_ptr<Function> function = function_call->get_functions()[0];

    writer << "{   // Call " << function->get_name() << "\n";
    writer.indent++;
    {
        vector<string> input_names;
        vector<string> output_names;

        for (const runtime::cpu::TensorViewWrapper& input : args)
        {
            input_names.push_back(input.get_name());
        }

        for (const runtime::cpu::TensorViewWrapper& output : out)
        {
            output_names.push_back(output.get_name());
        }

        writer << "void* args[] =\n{";
        writer.indent++;
        writer << "\n" << join(input_names, ",\n");
        writer.indent--;
        writer << "\n};\n";

        writer << "void* out[] =\n{";
        writer.indent++;
        writer << "\n" << join(output_names, ",\n");
        writer.indent--;
        writer << "\n};\n";

        writer << "\n";
        writer << function->get_name() << "(args, out);\n";
    }
    writer.indent--;
    writer << "}\n";
}

// TODO: This and other ops include comments/notes that
// we don't want to just copy-paste here. Figure out a better way
// or just point to ngvm/external_function.cpp with a note that
// the compiled version of these ops is intended to have semantics identical
// to what's seen there (for now atleast)

void runtime::cpu::CPU_Emitter::EmitReduce(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto reduce = static_cast<const op::Reduce*>(n);
    auto reduction_function = reduce->get_functions()[0];

    auto reductee_shape = args[0].get_shape();

    auto& f_result_element_type = out[0].get_element_type();
    auto result_shape = out[0].get_shape();

#if PREFER_EIGEN == 1
    auto& reduction_axes = reduce->get_reduction_axes();
    // Trivial case: no reduction axes (this includes the scalar-reductee case).
    if (reduction_axes.empty())
    {
        writer << "{   // " << n->get_name() << " 1\n";
        writer.indent++;
        writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    // Behavior for zero-size axes bears some explanation here. XLA's reduce
    // operator provides an "base" element (usually, but not necessarily,
    // an identity element) that it apparently *may* choose to insert anywhere
    // in the reduction any number of times. For example, given:
    //
    //   reduce{{1,2,3},b,+)
    //
    // any of the following are valid reductions (I think!):
    //
    //   b+(b+1+2)+3
    //   b+(1+(2+3))
    //   (1+2)+3 (I think!)
    //
    // etc. Here we will choose never to instantiate the base element, which
    // works well with Eigen's default behavior for non-zero-length axes. The
    // exceptional case is when we reduce on a zero-length axis. In this case,
    // Eigen's default behavior is to put a zero in the output,  which is not
    // what we want, so we detect that case here and override with a copy
    // instruction (for reduce-to-scalar) or a broadcast (for reduce-to-vector)
    // from the base element.
    //
    // What I'm actually not sure about is whether the identity element is
    // required to appear at least once. If so, this will need to be reworked,
    // assuming we actually want to mimic XLA's semantics that closely, which
    // we may not.
    else if ((reductee_shape.size() == 1 && reduction_axes == AxisSet{0}) ||
             (reductee_shape.size() == 2 && reduction_axes == AxisSet{0, 1}))
    {
        if (reductee_shape.at(0) == 0 || (reductee_shape.size() == 2 && reductee_shape.at(1) == 0))
        {
            writer << "{   // " << n->get_name() << " 2\n";
            writer.indent++;
            writer << "memcpy(" << out[0].get_name() << ", " << args[1].get_name() << ", "
                   << out[0].get_size() * out[0].get_element_type().size() << ");\n";
            writer.indent--;
            writer << "}\n";
        }
        else
        {
            writer << "{   // " << n->get_name() << " 3\n";
            writer.indent++;
            string type = f_result_element_type.c_type_string();
            writer << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            writer.indent++;
            writer << "\n";
            writer << type << " result;\n";
            writer << "void* args[] = {&x, &y};\n";
            writer << "void* out[] = {&result};\n";
            writer << reduction_function->get_name() << "(args, out);\n";
            writer << "return result;\n";
            writer.indent--;
            writer << "};\n";
            writer << emit_array1d(out[0]) << " =\n"
                   << "    " << emit_array1d(args[0]) << ".redux(f);\n";
            writer.indent--;
            writer << "}\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{1})
    {
        if (reductee_shape.at(1) == 0)
        {
            writer << "{   // " << n->get_name() << " 4\n";
            writer.indent++;
            writer << emit_array1d(out[0]) << " =\n"
                   << "    " << emit_array1d(args[1]) << "(0, 0);\n";
            writer.indent--;
            writer << "}\n";
        }
        else
        {
            // shared_ptr<CallFrame> cf =
            //     dynamic_pointer_cast<CallFrame>(external->make_call_frame());
            // ef->get_callees().emplace_back(cf);

            writer << "{   // " << n->get_name() << " 5\n";
            writer.indent++;
            string type = f_result_element_type.c_type_string();
            writer << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            writer.indent++;
            writer << "\n";
            writer << type << " result;\n";
            writer << "void* args[] = {&x, &y};\n";
            writer << "void* out[] = {&result};\n";
            writer << reduction_function->get_name() << "(args, out);\n";
            writer << "return result;\n";
            writer.indent--;
            writer << "};\n";
            writer << emit_vector(out[0]) << " =\n"
                   << "        " << emit_matrix(args[0]) << ".rowwise().redux(f);\n";
            writer.indent--;
            writer << "}\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{0})
    {
        if (reductee_shape.at(0) == 0)
        {
            writer << "{   // " << n->get_name() << " 6\n";
            writer.indent++;
            writer << emit_array1d(out[0]) << " =\n"
                   << "    " << emit_array1d(args[1]) << "(0, 0);\n";
            writer.indent--;
            writer << "}\n";
        }
        else
        {
            writer << "{   // " << n->get_name() << " 7\n";
            writer.indent++;
            string type = f_result_element_type.c_type_string();
            writer << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            writer.indent++;
            writer << "\n";
            writer << type << " result;\n";
            writer << "void* args[] = {&x, &y};\n";
            writer << "void* out[] = {&result};\n";
            writer << reduction_function->get_name() << "(args, out);\n";
            writer << "return result;\n";
            writer.indent--;
            writer << "};\n";
            writer << emit_vector(out[0]) << " =\n"
                   << "    " << emit_matrix(args[0]) << ".colwise().redux(f);\n";
            writer.indent--;
            writer << "}\n";
        }
    }
    else
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;

        string type = f_result_element_type.c_type_string();
        writer << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
        writer.indent++;
        writer << "\n";
        writer << type << " result;\n";
        writer << "void* args[] = {&x, &y};\n";
        writer << "void* out[] = {&result};\n";
        writer << reduction_function->get_name() << "(args, out);\n";
        writer << "return result;\n";
        writer.indent--;
        writer << "};\n";

        writer << "kernel::reduce<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        writer << "               " << args[1].get_name() << ",\n";
        writer << "               " << out[0].get_name() << ",\n";
        writer << "               {" << join(args[0].get_shape()) << "},\n";
        writer << "               {" << join(out[0].get_shape()) << "},\n";
        writer << "               {" << join(reduce->get_reduction_axes()) << "},\n";
        writer << "               f);\n";

        writer.indent--;
        writer << "}\n";
    }
#else
    writer << "{   // " << n->get_name() << " 1\n";
    writer.indent++;

    string type = f_result_element_type.c_type_string();

    writer << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
    writer.indent++;
    writer << "\n";
    writer << type << " result;\n";
    writer << "void* args[] = {&x, &y};\n";
    writer << "void* out[] = {&result};\n";
    writer << reduction_function->get_name() << "(args, out);\n";
    writer << "return result;\n";
    writer.indent--;
    writer << "};\n";

    kernel::emit_reduce(writer,
                        args[0].get_element_type().c_type_string(),
                        args[0].get_name(),
                        args[1].get_name(),
                        out[0].get_name(),
                        args[0].get_shape(),
                        out[0].get_shape(),
                        reduce->get_reduction_axes());

    writer.indent--;
    writer << "}\n";
#endif
}

void runtime::cpu::CPU_Emitter::EmitSign(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".sign();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = (0 < " << args[0].get_name() << "[i]) - ("
           << args[0].get_name() << "[i] < 0);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSlice(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    const op::Slice* slice = static_cast<const op::Slice*>(n);

    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    size_t arg_rank = args[0].get_shape().size();

    const Coordinate& lower_bounds = slice->get_lower_bounds();
    const Coordinate& upper_bounds = slice->get_upper_bounds();

    bool strided = false;
    for (size_t stride : slice->get_strides())
    {
        if (stride != 1)
        {
            strided = true;
            break;
        }
    }

    // Scalar slice is necessarily just a copy.
    if (!strided && arg_rank == 0)
    {
        writer << "{   // " << n->get_name() << " 1\n";
        writer.indent++;
        writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (!strided && arg_rank == 1)
    {
        writer << "{   // " << n->get_name() << " 2\n";
        writer.indent++;
        writer << emit_vector(out[0]) << " =\n"
               << "    " << emit_vector(args[0]) << ".segment(\n"
               << "        " << to_string(lower_bounds[0]) << ", "
               << to_string(upper_bounds[0] - lower_bounds[0]) << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (!strided && arg_rank == 2)
    {
        writer << "{   // " << n->get_name() << " 3\n";
        writer.indent++;
        writer << emit_matrix(out[0]) << " = \n"
               << "        " << emit_matrix(args[0]) << ".block(" << to_string(lower_bounds[0])
               << ", " << to_string(lower_bounds[1]) << ",\n"
               << "        " << to_string(upper_bounds[0] - lower_bounds[0]) << ",\n"
               << "        " << to_string(upper_bounds[1] - lower_bounds[1]) << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        writer << "kernel::slice<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        writer << "                         " << out[0].get_name() << ",\n";
        writer << "                         {" << join(args[0].get_shape()) << "},\n";
        writer << "                         {" << join(slice->get_lower_bounds()) << "},\n";
        writer << "                         {" << join(slice->get_upper_bounds()) << "},\n";
        writer << "                         {" << join(slice->get_strides()) << "},\n";
        writer << "                         {" << join(out[0].get_shape()) << "});\n";
    }
#else
    kernel::emit_slice(writer,
                       args[0].get_element_type().c_type_string(),
                       args[0].get_name(),
                       out[0].get_name(),
                       args[0].get_shape(),
                       out[0].get_shape(),
                       slice->get_lower_bounds(),
                       slice->get_upper_bounds(),
                       slice->get_strides());
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSum(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    const op::Sum* sum = static_cast<const op::Sum*>(n);
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    const Shape& arg_shape = args[0].get_shape();
    size_t arg_rank = arg_shape.size();
    const AxisSet& reduction_axes = sum->get_reduction_axes();

    // Trivial case: no reduction axes.
    if (reduction_axes.size() == 0)
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    // Full reduction? Then sum to scalar.
    else if ((arg_rank == 1 && reduction_axes == AxisSet{0}) ||
             (arg_rank == 2 && reduction_axes == AxisSet{0, 1}))
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_array1d(out[0]) << " =\n"
               << "    " << emit_array1d(args[0]) << ".sum();\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{1})
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_vector(out[0]) << " =\n"
               << "    " << emit_matrix(args[0]) << ".rowwise().sum();\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{0})
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << emit_vector(out[0]) << " =\n"
               << "    " << emit_matrix(args[0]) << ".colwise().sum();\n";
        writer.indent--;
        writer << "}\n";
    }
    else
    {
        writer << "kernel::sum<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        writer << "                         " << out[0].get_name() << ",\n";
        writer << "                         {" << join(args[0].get_shape()) << "},\n";
        writer << "                         {" << join(out[0].get_shape()) << "},\n";
        writer << "                         {" << join(sum->get_reduction_axes()) << "});\n";
    }
#else
    kernel::emit_sum(writer,
                     args[0].get_element_type().c_type_string(),
                     args[0].get_name(),
                     out[0].get_name(),
                     args[0].get_shape(),
                     out[0].get_shape(),
                     sum->get_reduction_axes());
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitExp(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".exp();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = exp(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSin(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".sin();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = sin(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSinh(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".sinh();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = sinh(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitCos(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".cos();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = cos(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitCosh(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".cosh();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = cosh(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitTan(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".tan();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = tan(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitTanh(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    // Eigen's generic_fast_tanh_float<float> is currently miscompiled by Clang/LLVM
    // so we fall-back to tanh
    // TODO: Implement our own internal fast/approximate tanh if this actually gets used
    // by models
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 0
    writer << "#pragma omp parallel for\n";
#endif
    writer << "for (size_t i=0; i<" << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = tanh(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAsin(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".asin();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = asin(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAcos(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".acos();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = acos(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAtan(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " =\n"
           << "    " << emit_array1d(args[0]) << ".atan();\n";
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = atan(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitPower(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    writer << emit_array1d(out[0]) << " = \n";
    writer.indent++;
    writer << emit_array1d(args[0]) << ".pow(\n ";
    writer << emit_array1d(args[1]) << ");\n";
    writer.indent--;
#else
    writer << "#pragma omp parallel for\n";
    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = pow(" << args[0].get_name() << "[i], "
           << args[1].get_name() << "[i]);\n";
    writer << "}\n";
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitReplaceSlice(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto replace_slice = static_cast<const op::Slice*>(n);
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
#if PREFER_EIGEN == 1
    size_t arg0_rank = args[0].get_shape().size();

    auto& lower_bounds = replace_slice->get_lower_bounds();
    auto& upper_bounds = replace_slice->get_upper_bounds();

    bool strided = false;
    for (size_t stride : replace_slice->get_strides())
    {
        if (stride != 1)
        {
            strided = true;
            break;
        }
    }

    // Scalar slice is necessarily just a copy.
    if (!strided && arg0_rank == 0)
    {
        writer << "{   // " << n->get_name() << " 1\n";
        writer.indent++;
        writer << "memcpy(" << out[0].get_name() << ", " << args[1].get_name() << ", "
               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (!strided && arg0_rank == 1)
    {
        writer << "{   // " << n->get_name() << " 2\n";
        writer.indent++;
        writer << emit_vector(out[0]) << " =\n"
               << "    " << emit_vector(args[0]) << ";\n"
               << emit_vector(out[0]) << ".segment(\n"
               << "    " << to_string(lower_bounds[0]) << ", "
               << to_string(upper_bounds[0] - lower_bounds[0]) << ") =\n"
               << "    " << emit_vector(args[1]) << ";\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (!strided && arg0_rank == 2)
    {
        writer << "{   // " << n->get_name() << " 3\n";
        writer.indent++;
        writer << emit_matrix(out[0]) << " =\n"
               << "    " << emit_matrix(args[0]) << ";\n"
               << emit_matrix(out[0]) << ".block(\n"
               << "        " << to_string(lower_bounds[0]) << ",\n"
               << "        " << to_string(lower_bounds[1]) << ",\n"
               << "        " << to_string(upper_bounds[0] - lower_bounds[0]) << ",\n"
               << "        " << to_string(upper_bounds[1] - lower_bounds[1]) << ") =\n"
               << "    " << emit_matrix(args[1]) << ";\n";
        writer.indent--;
        writer << "}\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        writer << "kernel::replace_slice<" << out[0].get_type() << ">(" << args[0].get_name()
               << ",\n";
        writer << "                         " << args[1].get_name() << ",\n";
        writer << "                         " << out[0].get_name() << ",\n";
        writer << "                         {" << join(args[1].get_shape()) << "},\n";
        writer << "                         {" << join(replace_slice->get_lower_bounds()) << "},\n";
        writer << "                         {" << join(replace_slice->get_upper_bounds()) << "},\n";
        writer << "                         {" << join(replace_slice->get_strides()) << "},\n";
        writer << "                         {" << join(out[0].get_shape()) << "});\n";
    }
#else
    kernel::emit_replace_slice(writer,
                               args[0].get_element_type().c_type_string(),
                               args[0].get_name(),
                               args[1].get_name(),
                               out[0].get_name(),
                               args[1].get_shape(),
                               out[0].get_shape(),
                               replace_slice->get_lower_bounds(),
                               replace_slice->get_upper_bounds(),
                               replace_slice->get_strides());
#endif
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitOneHot(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto oh = static_cast<const op::OneHot*>(n);

    auto arg_rank = args[0].get_shape().size();

    size_t bounds = out[0].get_shape()[oh->get_one_hot_axis()];

    if (arg_rank == 0)
    {
        writer << "{   // " << n->get_name() << " 1\n";
        writer.indent++;

        writer << emit_vector(out[0], "out_vector") << ";\n";

        writer << "out_vector.setZero();\n"
               << ""
               << "auto pos_raw = " << emit_vector(args[0]) << "(0, 0);\n"
               << "if (floor(pos_raw) != pos_raw)\n"
               << "{\n";
        writer.indent++;
        writer << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
        writer.indent--;
        writer << "}\n";

        writer << "size_t pos = pos_raw;\n"
               << "if (pos >= " << bounds << ")\n";

        writer << "{\n";
        writer.indent++;
        writer << "throw(std::range_error(\"One-hot: value is out of category range\"));\n";
        writer.indent--;
        writer << "}\n";

        writer << "out_vector(pos, 0) = 1;\n";

        writer.indent--;
        writer << "}\n";
    }
    else if (arg_rank == 1)
    {
        writer << "{   // " << n->get_name() << " 1\n";
        writer.indent++;

        writer << emit_vector(args[0], "arg_vector") << ";\n";

        writer << emit_matrix(out[0], "out_vector") << ";\n";
        writer << "out_vector.setZero();\n";

        writer << "for (size_t i = 0; i < " << args[0].get_shape()[0] << "; i++)\n"
               << "{\n";
        writer.indent++;

        writer << "auto pos_raw = arg_vector(i, 0);\n";

        writer << "if (floor(pos_raw) != pos_raw)\n"
               << "{\n";
        writer.indent++;
        writer << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
        writer.indent--;
        writer << "}\n";

        writer << "size_t pos = pos_raw;\n";
        writer << "bool found = false;\n";

        writer << "if (pos >= " << bounds << ")\n"
               << "{\n";
        writer.indent++;
        writer << "throw(std::range_error(\"One-hot: value is out of category range\"));\n";
        writer.indent--;
        writer << "}\n";

        writer << "out_vector" << (oh->get_one_hot_axis() == 0 ? "(pos, i)" : "(i, pos)")
               << " = 1;\n";

        writer.indent--;
        writer << "}\n";

        writer.indent--;
        writer << "}\n";
    }
    // Other cases are not handled yet.
    else
    {
        writer << "kernel::one_hot<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        writer << "                   " << out[0].get_name() << ",\n";
        writer << "                   {" << join(args[0].get_shape()) << "},\n";
        writer << "                   {" << join(out[0].get_shape()) << "},\n";
        writer << "                   " << oh->get_one_hot_axis() << ");\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitCeiling(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
    size_t element_count = out[0].get_size();
#if PREFER_EIGEN == 0
    writer << "#pragma omp parallel for\n";
#endif
    writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = ceil(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitFloor(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
    size_t element_count = out[0].get_size();
#if PREFER_EIGEN == 0
    writer << "#pragma omp parallel for\n";
#endif
    writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = floor(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSqrt(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
    size_t element_count = out[0].get_size();
#if PREFER_EIGEN == 0
    writer << "#pragma omp parallel for\n";
#endif
    writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    writer << "{\n";
    writer << "    " << out[0].get_name() << "[i] = sqrt(" << args[0].get_name() << "[i]);\n";
    writer << "}\n";
    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitConvolution(codegen::CodeWriter& writer,
                                                const ngraph::Node* n,
                                                const vector<runtime::cpu::TensorViewWrapper>& args,
                                                const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto convolution = static_cast<const op::Convolution*>(n);

    auto arg0_shape = args[0].get_shape();
    auto arg1_shape = args[1].get_shape();
    auto result_shape = out[0].get_shape();
    auto arg0_rank = arg0_shape.size();
    auto arg1_rank = arg1_shape.size();

    bool filter_dilated = false;
    for (size_t s : convolution->get_window_dilation_strides())
    {
        filter_dilated = filter_dilated || (s != 1);
    }

    bool images_dilated = false;
    for (size_t s : convolution->get_image_dilation_strides())
    {
        images_dilated = images_dilated || (s != 1);
    }

    if (!writer.emitted_mkldnn_preamble)
    {
        EmitMKLDNNPreamble(writer);
    }

    // TODO: MKLDNN streams should be static so we need to either implement
    // codegen for statics or move primitive and stream construction out
    // of the generated function and only generate code to run/rerun the stream

    if (!filter_dilated && !images_dilated && arg0_rank == 4 && arg1_rank == 4 &&
        args[0].get_element_type() == element::f32)
    {
        const string& et = get_mkldnn_data_type(args[0].get_element_type().c_type_string());

        writer << "{\n";
        writer.indent++;

        writer << "auto input_data_desc = memory::desc({" << join(arg0_shape) << "}, " << et
               << ", memory::format::nchw);\n";
        writer << "auto weights_desc = memory::desc({" << join(arg1_shape) << "}, " << et
               << ", memory::format::oihw);\n";
        writer << "auto result_desc = memory::desc({" << join(result_shape) << "}, " << et
               << ", memory::format::nchw);\n";

        writer << "auto input_data = memory({input_data_desc, cpu_engine}, " << args[0].get_name()
               << ");\n";
        writer << "auto weights = memory({weights_desc, cpu_engine}, " << args[1].get_name()
               << ");\n";
        writer << "auto result = memory({result_desc, cpu_engine}, " << out[0].get_name() << ");\n";
        writer << "auto conv = convolution_forward({"
               << "{prop_kind::forward, algorithm::convolution_direct, input_data_desc, "
                  "weights_desc, result_desc, {"
               << join(convolution->get_window_movement_strides()) << "}, {"
               << join(convolution->get_padding_below()) << "}, {"
               << join(convolution->get_padding_above()) << "}, padding_kind::zero}, cpu_engine}, "
               << "input_data, weights, result);\n";

        writer << "auto s = stream(stream::kind::eager);\n"
               << "s.submit({conv}).wait();\n";
        writer.indent--;
        writer << "}\n";
    }
    else if (filter_dilated && !images_dilated && arg0_rank == 4 && arg1_rank == 4 &&
             args[0].get_element_type() == element::f32)
    {
        // For dilation, MKLDNN wants to know how many elements to insert between, not how far
        // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
        Strides window_dilation_strides_adjusted;

        for (size_t s : convolution->get_window_dilation_strides())
        {
            window_dilation_strides_adjusted.push_back(s - 1);
        }

        const string& et = get_mkldnn_data_type(args[0].get_element_type().c_type_string());

        writer << "{\n";
        writer.indent++;

        writer << "auto input_data_desc = memory::desc({" << join(arg0_shape) << "}, " << et
               << ", memory::format::nchw);\n";
        writer << "auto weights_desc = memory::desc({" << join(arg1_shape) << "}, " << et
               << ", memory::format::oihw);\n";
        writer << "auto result_desc = memory::desc({" << join(result_shape) << "}, " << et
               << ", memory::format::nchw);\n";

        writer << "auto input_data = memory({input_data_desc, cpu_engine}, " << args[0].get_name()
               << ");\n";
        writer << "auto weights = memory({weights_desc, cpu_engine}, " << args[1].get_name()
               << ");\n";
        writer << "auto result = memory({result_desc, cpu_engine}, " << out[0].get_name() << ");\n";
        writer << "auto conv = convolution_forward({"
               << "{prop_kind::forward, algorithm::convolution_direct, input_data_desc, "
                  "weights_desc, result_desc, {"
               << join(convolution->get_window_movement_strides()) << "}, {"
               << join(window_dilation_strides_adjusted) << "}, {"
               << join(convolution->get_padding_below()) << "}, {"
               << join(convolution->get_padding_above()) << "}, padding_kind::zero}, cpu_engine}, "
               << "input_data, weights, result);\n";

        writer << "auto s = stream(stream::kind::eager);\n"
               << "s.submit({conv}).wait();\n";
        writer.indent--;
        writer << "}\n";
    }
    else
    {
        writer << "kernel::convolution<" << out[0].get_type() << ">(" << args[0].get_name()
               << ",\n";
        writer << "                         " << args[1].get_name() << ",\n";
        writer << "                         " << out[0].get_name() << ",\n";
        writer << "                         {" << join(arg0_shape) << "},\n";
        writer << "                         {" << join(arg1_shape) << "},\n";
        writer << "                         {" << join(result_shape) << "},\n";
        writer << "                         {" << join(convolution->get_window_movement_strides())
               << "},\n";
        writer << "                         {" << join(convolution->get_window_dilation_strides())
               << "},\n";
        writer << "                         {" << join(convolution->get_padding_below()) << "},\n";
        writer << "                         {" << join(convolution->get_padding_above()) << "},\n";
        writer << "                         {" << join(convolution->get_image_dilation_strides())
               << "});\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitNot(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    writer << "kernel::logical_not(" << args[0].get_name() << ",\n"
           << "                    " << out[0].get_name() << ",\n"
           << "                    " << out[0].get_size() << ");\n";
}

void runtime::cpu::CPU_Emitter::EmitMaxPool(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto max_pool = static_cast<const op::MaxPool*>(n);

    auto arg_shape = args[0].get_shape();
    auto arg_rank = arg_shape.size();

    auto result_shape = out[0].get_shape();

    // TODO: Optimize for 1D

    if (!writer.emitted_mkldnn_preamble)
    {
        EmitMKLDNNPreamble(writer);
    }

    // TODO: Remove element type restriction
    if (arg_rank == 4 && max_pool->get_window_shape().size() == 2 &&
        args[0].get_element_type() == element::f32)
    {
        const string& et = get_mkldnn_data_type(args[0].get_element_type().c_type_string());

        writer << "{\n";
        writer.indent++;

        writer << "auto input_data_desc = memory::desc({" << join(arg_shape) << "}, " << et
               << ", memory::format::nchw);\n";
        writer << "auto result_desc = memory::desc({" << join(result_shape) << "}, " << et
               << ", memory::format::nchw);\n";

        writer << "auto input_data = memory({input_data_desc, cpu_engine}, " << args[0].get_name()
               << ");\n";
        writer << "auto result = memory({result_desc, cpu_engine}, " << out[0].get_name() << ");\n";

        // TODO: Use a workspace
        writer << "auto max_pooling = pooling_forward({"
               << "{prop_kind::forward_inference, algorithm::pooling_max, "
               << "input_data_desc, result_desc, {" << join(max_pool->get_window_movement_strides())
               << "}, {" << join(max_pool->get_window_shape()) << "}, {0, 0}, "
               << "{0, 0}, padding_kind::zero}, cpu_engine}, "
               << "input_data, result);\n";

        writer << "auto s = stream(stream::kind::eager);\n"
               << "s.submit({max_pooling}).wait();\n";
        writer.indent--;
        writer << "}\n";
    }
    else
    {
        writer << "kernel::max_pool<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        writer << "                 " << out[0].get_name() << ",\n";
        writer << "                 {" << join(arg_shape) << "},\n";
        writer << "                 {" << join(result_shape) << "},\n";
        writer << "                 {" << join(max_pool->get_window_shape()) << "},\n";
        writer << "                 {" << join(max_pool->get_window_movement_strides()) << "});\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitReverse(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto reverse = static_cast<const op::Reverse*>(n);

    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    writer << "kernel::reverse<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    writer << "                " << out[0].get_name() << ",\n";
    writer << "                {" << join(arg_shape) << "},\n";
    writer << "                {" << join(result_shape) << "},\n";
    writer << "                {" << join(reverse->get_reversed_axes()) << "});\n";
}

void runtime::cpu::CPU_Emitter::EmitReduceWindow(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto reduce_window = static_cast<const op::ReduceWindow*>(n);

    auto arg_reductee_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();
    auto reduction_function = reduce_window->get_functions()[0];
    auto& f_result_element_type = out[0].get_element_type();

    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;

    string type = f_result_element_type.c_type_string();
    writer << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
    writer.indent++;
    writer << "\n";
    writer << type << " result;\n";
    writer << "void* args[] = {&x, &y};\n";
    writer << "void* out[] = {&result};\n";
    writer << reduction_function->get_name() << "(args, out);\n";
    writer << "return result;\n";
    writer.indent--;
    writer << "};\n";

    writer << "kernel::reduce_window<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    writer << "                      " << args[1].get_name() << ",\n";
    writer << "                      " << out[0].get_name() << ",\n";
    writer << "                      {" << join(arg_reductee_shape) << "},\n";
    writer << "                      {" << join(result_shape) << "},\n";
    writer << "                      f,\n";
    writer << "                      {" << join(reduce_window->get_window_shape()) << "},\n";
    writer << "                      {" << join(reduce_window->get_window_movement_strides())
           << "});\n";

    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSelectAndScatter(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto select_and_scatter = static_cast<const op::SelectAndScatter*>(n);
    auto selection_function = select_and_scatter->get_functions()[0];
    auto scatter_function = select_and_scatter->get_functions()[1];

    auto arg0_shape = args[0].get_shape();
    auto arg1_shape = args[1].get_shape();
    auto result_shape = out[0].get_shape();

    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;

    string type = n->get_output_element_type(0).c_type_string();

    writer << "auto f_select = [](" << type << " x, " << type << " y) -> char\n{";
    writer.indent++;
    writer << "\n";
    writer << "char result;\n";
    writer << "void* args[] = {&x, &y};\n";
    writer << "void* out[] = {&result};\n";
    writer << selection_function->get_name() << "(args, out);\n";
    writer << "return result;\n";
    writer.indent--;
    writer << "};\n";

    writer << "auto f_scatter = [](" << type << " x, " << type << " y) -> " << type << "\n{";
    writer.indent++;
    writer << "\n";
    writer << type << " result;\n";
    writer << "void* args[] = {&x, &y};\n";
    writer << "void* out[] = {&result};\n";
    writer << scatter_function->get_name() << "(args, out);\n";
    writer << "return result;\n";
    writer.indent--;
    writer << "};\n";

    writer << "kernel::select_and_scatter<" << out[0].get_type() << ">(" << args[0].get_name()
           << ",\n";
    writer << "                " << args[1].get_name() << ",\n";
    writer << "                " << args[2].get_name() << ",\n";
    writer << "                " << out[0].get_name() << ",\n";
    writer << "                {" << join(arg0_shape) << "},\n";
    writer << "                {" << join(arg1_shape) << "},\n";
    writer << "                {" << join(result_shape) << "},\n";
    writer << "                f_select,\n";
    writer << "                f_scatter,\n";
    writer << "                {" << join(select_and_scatter->get_window_shape()) << "},\n";
    writer << "                {" << join(select_and_scatter->get_window_movement_strides())
           << "});\n";

    writer.indent--;
    writer << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAvgPool(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto avg_pool = static_cast<const op::AvgPool*>(n);

    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    writer << "kernel::avg_pool<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    writer << "                 " << out[0].get_name() << ",\n";
    writer << "                 {" << join(arg_shape) << "},\n";
    writer << "                 {" << join(result_shape) << "},\n";
    writer << "                 {" << join(avg_pool->get_window_shape()) << "},\n";
    writer << "                 {" << join(avg_pool->get_window_movement_strides()) << "},\n";
    writer << "                 {" << join(avg_pool->get_padding_below()) << "},\n";
    writer << "                 {" << join(avg_pool->get_padding_above()) << "});\n";
}

void runtime::cpu::CPU_Emitter::EmitPad(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto pad = static_cast<const op::Pad*>(n);

    auto arg0_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    writer << "kernel::pad<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
    writer << "            " << args[1].get_name() << ",\n";
    writer << "            " << out[0].get_name() << ",\n";
    writer << "            {" << join(arg0_shape) << "},\n";
    writer << "            {" << join(result_shape) << "},\n";
    writer << "            {" << join(pad->get_padding_below()) << "},\n";
    writer << "            {" << join(pad->get_padding_above()) << "},\n";
    writer << "            {" << join(pad->get_padding_interior()) << "});\n";
}

//------------------------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------------------------

static string format_name(const string& name)
{
    string rc;
    if (!name.empty())
    {
        rc = " " + name;
    }
    return rc;
}

string runtime::cpu::CPU_Emitter::emit_vector(const runtime::cpu::TensorViewWrapper& tvi,
                                              const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_element_type();
    ss << "EigenVector<" << et.c_type_string() << ">" << format_name(name) << "(" << tvi.get_name()
       << ", " << eigen_vector_format(tvi) << ")";
    return ss.str();
}

string runtime::cpu::CPU_Emitter::emit_array1d(const runtime::cpu::TensorViewWrapper& tvi,
                                               const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_element_type();
    ss << "EigenArray1d<" << et.c_type_string() << ">" << format_name(name) << "(" << tvi.get_name()
       << ", " << eigen_vector_format(tvi) << ")";
    return ss.str();
}

string runtime::cpu::CPU_Emitter::emit_matrix(const runtime::cpu::TensorViewWrapper& tvi,
                                              const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_element_type();
    ss << "EigenMatrix<" << et.c_type_string() << ">" << format_name(name) << "(" << tvi.get_name()
       << ", " << eigen_matrix_format(tvi.get_shape(), tvi.get_strides()) << ")";
    return ss.str();
}
