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
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_emitters.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

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

void runtime::cpu::CPU_Emitter::EmitNop(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
}

void runtime::cpu::CPU_Emitter::EmitAdd(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    // TODO: Audit all uses of Add and fix this to use
    // the right alignment instead of Eigen::Unaligned
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "Eigen::Map<Eigen::Array<" << out[0].get_element_type().c_type_string() << ", "
          << out[0].get_size() << ", 1>, Eigen::Unaligned> out(" << out[0].get_name() << ");\n";
    m_out << "Eigen::Map<Eigen::Array<" << args[0].get_element_type().c_type_string() << ", "
          << args[0].get_size() << ", 1>, Eigen::Unaligned> arg0(" << args[0].get_name() << ");\n";
    m_out << "Eigen::Map<Eigen::Array<" << args[1].get_element_type().c_type_string() << ", "
          << args[1].get_size() << ", 1>, Eigen::Unaligned> arg1(" << args[1].get_name() << ");\n";
    m_out << "out = arg0 + arg1;\n";

    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitDot(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    const Shape& arg0_shape = args[0].get_shape();
    const Shape& arg1_shape = args[1].get_shape();
    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& first = (arg0_shape.empty() ? args[0] : args[1]);
        auto& second = (arg0_shape.empty() ? args[1] : args[0]);

        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_vector(out[0]) << "\n    = ";
        m_out << first.get_name() << "[0]\n    * " << emit_vector(second) << ";\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1))
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_vector(out[0]) << " << \n"
              << "    " << emit_vector(args[0]) << ".dot(" << emit_vector(args[1]) << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1))
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_vector(out[0]) << " = \n"
              << "    " << emit_matrix(args[0]) << " * " << emit_vector(args[1]) << ";\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2))
    {
        // Emit an MKL SGEMM call if possible
        // clang-format off
        if (args[0].get_element_type() == element::f32)
        {
            m_out << "{   // " << n->get_name() << "\n";
            m_out.indent++;
            m_out << "cblas::cblas_sgemm("
               << "cblas::Layout::RowMajor, "
               << "cblas::Transpose::None, "
               << "cblas::Transpose::None, "
               << arg0_shape[0] << ", " << arg1_shape[1] << ", " << arg0_shape[1] << ",\n" <<
                "        1.0f, " << args[0].get_name() << ", " << max(1UL, arg0_shape[1]) << ", " << args[1].get_name() << ", " << max(1UL, arg1_shape[1]) << ", 0.0f,\n" <<
                "        " << out[0].get_name() << ", " << max(1UL, arg1_shape[1]) << ");\n";
            m_out.indent--;
            m_out << "}\n";
        }
        // clang-format on
        else
        {
            m_out << "{   // " << n->get_name() << "\n";
            m_out.indent++;
            m_out << emit_matrix(out[0]) << " = \n"
                  << "    " << emit_matrix(args[0]) << " * " << emit_matrix(args[1]) << ";\n";
            m_out.indent--;
            m_out << "}\n";
        }
    }
    else
    {
        const ngraph::op::Dot* dot = static_cast<const ngraph::op::Dot*>(n);

        m_out << "kernel::dot(" << args[0].get_name() << ",\n";
        m_out << "            " << args[1].get_name() << ",\n";
        m_out << "            " << out[0].get_name() << ",\n";
        m_out << "            {" << join(args[0].get_shape()) << "},\n";
        m_out << "            {" << join(args[1].get_shape()) << "},\n";
        m_out << "            {" << join(out[0].get_shape()) << "},\n";
        m_out << "            " << dot->get_reduction_axes_count() << ");\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitMultiply(const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "   " << emit_array1d(args[0]) << " *\n"
          << "   " << emit_array1d(args[1]) << ";\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitGetOutputElement(
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto get_tuple_element = static_cast<const op::GetOutputElement*>(n);

    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "memcpy(" << out[0].get_name() << ", " << args[get_tuple_element->get_n()].get_name()
          << ", " << out[0].get_size() * out[0].get_element_type().size() << ");\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitTuple(const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    for (size_t i = 0; i < args.size(); ++i)
    {
        m_out << "memcpy(" << out.at(i).get_name() << ", " << args.at(i).get_name() << ", "
              << out[i].get_size() * out[i].get_element_type().size() << ");\n";
    }
    m_out.indent--;
    m_out += "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAbs(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n";
    m_out << "Eigen::abs(" << emit_array1d(args[0]) << ");\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitConcat(const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto result_shape = out[0].get_shape();

    if (result_shape.size() == 1)
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_vector(out[0], "out_vector") << ";\n";

        size_t concat_pos = 0;
        for (size_t i = 0; i < args.size(); i++)
        {
            m_out << "out_vector.segment(" << concat_pos << ", " << args[i].get_shape().at(0)
                  << ") << " << emit_vector(args[i]) << ";\n";
            concat_pos += args[i].get_shape().at(0);
        }
        m_out.indent--;
        m_out << "}\n";
    }
    else if (result_shape.size() == 2)
    {
        auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_matrix(out[0], "out_matrix") << ";\n";

        size_t concat_pos[2]{0, 0};
        for (size_t i = 0; i < args.size(); i++)
        {
            auto& arg_shape = args[i].get_shape();

            m_out << "out_matrix.block(" << concat_pos[0] << ", " << concat_pos[1] << ", "
                  << arg_shape.at(0) << ", " << arg_shape.at(1) << ") << " << emit_matrix(args[i])
                  << ";\n";

            concat_pos[axis] += arg_shape.at(axis);
        }

        m_out.indent--;
        m_out << "}\n";
    }
    else
    {
        if (m_use_ref_kernels)
        {
            auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

            std::vector<std::string> arg_names;
            std::vector<std::string> arg_shape_strings;

            for (auto arg : args)
            {
                arg_names.push_back(arg.get_name());
                arg_shape_strings.push_back("{" + join(arg.get_shape()) + "}");
            }

            m_out << "kernel::concat<" << out[0].get_type() << ">({" << join(arg_names) << "},\n";
            m_out << "                         " << out[0].get_name() << ",\n";
            m_out << "                         {" << join(arg_shape_strings) << "},\n";
            m_out << "                         {" << join(result_shape) << "},\n";
            m_out << "                         " << axis << ");\n";
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

            kernels::emit_concat(m_out,
                                 args[0].get_element_type().c_type_string(),
                                 arg_names,
                                 out[0].get_name(),
                                 arg_shapes,
                                 result_shape,
                                 axis);
        }
    }
}

void runtime::cpu::CPU_Emitter::EmitDivide(const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
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
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << " /\n"
          << "    " << emit_array1d(args[1]) << ";\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitEqual(const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    (" << emit_array1d(args[0]) << " ==\n"
          << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitGreater(const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << " xxx\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    (" << emit_array1d(args[0]) << " >\n"
          << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitGreaterEq(const ngraph::Node* n,
                                              const vector<runtime::cpu::TensorViewWrapper>& args,
                                              const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    (" << emit_array1d(args[0]) << " >=\n"
          << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitLess(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    (" << emit_array1d(args[0]) << " <\n"
          << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitLessEq(const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    (" << emit_array1d(args[0]) << " <=\n"
          << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitLog(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    Eigen::log(" << emit_array1d(args[0]) << ");\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitMaximum(const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "        " << emit_array1d(args[0]) << ".max(\n"
          << "        " << emit_array1d(args[1]) << ");\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitMinimum(const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".min(\n"
          << "    " << emit_array1d(args[1]) << ");\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitNegative(const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    -" << emit_array1d(args[0]) << ";\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitNotEqual(const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    (" << emit_array1d(args[0]) << " !=\n"
          << "    " << emit_array1d(args[1]) << ").template cast<char>();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSelect(const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "   " << emit_array1d(args[0]) << "\n"
          << "    .select(" << emit_array1d(args[1]) << ",\n"
          << "       " << emit_array1d(args[2]) << ");\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSubtract(const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << " -\n"
          << "    " << emit_array1d(args[1]) << ";\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitBroadcast(const ngraph::Node* n,
                                              const vector<runtime::cpu::TensorViewWrapper>& args,
                                              const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto broadcast = static_cast<const op::Broadcast*>(n);

    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    if (broadcast->get_broadcast_axes().empty())
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
              << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (arg_shape.size() == 0)
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_array1d(out[0]) << " =\n"
              << "    " << emit_array1d(args[0]) << "(0, 0);\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (arg_shape.size() == 1 && result_shape.size() == 2)
    {
        if (broadcast->get_broadcast_axes() == AxisSet{1})
        {
            m_out << "{   // " << n->get_name() << "\n";
            m_out.indent++;
            m_out << emit_matrix(out[0]) << ".colwise() =\n"
                  << "    " << emit_vector(args[0]) << ";\n";
            m_out.indent--;
            m_out << "}\n";
        }
        else if (broadcast->get_broadcast_axes() == AxisSet{0})
        {
            m_out << "{   // " << n->get_name() << "\n";
            m_out.indent++;

            m_out << "Eigen::Map<Eigen::Matrix<" << out[0].get_element_type().c_type_string()
                  << ", " << join(out[0].get_shape())
                  << ", Eigen::RowMajor>, Eigen::Aligned64, Eigen::Stride<"
                  << join(out[0].get_strides()) << ">> out(" << out[0].get_name() << ");\n";
            m_out << "Eigen::Map<Eigen::Matrix<" << args[0].get_element_type().c_type_string()
                  << ", 1, " << args[0].get_size()
                  << ", Eigen::RowMajor>, Eigen::Aligned64, Eigen::Stride<" << args[0].get_size()
                  << ", 1>> arg0(" << args[0].get_name() << ");\n";
            m_out << "out = arg0.replicate<" << out[0].get_shape().at(0) << ", 1>();\n";

            m_out.indent--;
            m_out << "}\n";
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
        m_out << "kernel::broadcast<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        m_out << "                         " << out[0].get_name() << ",\n";
        m_out << "                         {" << join(arg_shape) << "},\n";
        m_out << "                         {" << join(result_shape) << "},\n";
        m_out << "                         {" << join(broadcast->get_broadcast_axes()) << "});\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitConvert(const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto& result_element_type = out[0].get_element_type();

    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << "\n"
          << "    .template cast<" << result_element_type.c_type_string() << ">();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitConstant(const ngraph::Node* n,
                                             const vector<runtime::cpu::TensorViewWrapper>& args,
                                             const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto c = static_cast<const op::Constant*>(n);
    auto c_value_strings = c->get_value_strings();
    auto type = out[0].get_type();

    m_out << "{   // " << n->get_name() << " EmitConstant\n";
    m_out.indent++;
    for (size_t i = 0; i < c_value_strings.size(); i++)
    {
        m_out << out[0].get_name() << "[" << i << "] = static_cast<" << type << ">("
              << c_value_strings[i] << ");\n";
    }
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitReshape(const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto reshape = static_cast<const op::Reshape*>(n);

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
        m_out << "{   // " << n->get_name() << " 1\n";
        m_out.indent++;
        m_out << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
              << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    // If there *is* a layout change in the 2D case, we transpose the input.
    else if (arg_rank == 2)
    {
        // Emit an MKL transpose call if possible
        // clang-format off
        if (result_element_type == ngraph::element::Float32::element_type())
        {
            m_out << "{   // " << n->get_name() << " 2\n";
            m_out.indent++;
            m_out << "mkl::MKL_Somatcopy('R', 'T', " << to_string(arg_shape[0]) << ",\n" <<
                "                   " << to_string(arg_shape[1]) << ", 1.0f,\n" <<
                "                   " << args[0].get_name() << ", "
                << to_string(arg_shape[1]) << ",\n" <<
                "                   " << out[0].get_name()
                << ", " << to_string(arg_shape[0]) << ");\n";
                m_out.indent--;
                m_out << "}\n";
        }
        // clang-format on
        else
        {
            m_out << "{   // " << n->get_name() << " 3\n";
            m_out.indent++;
            m_out << emit_matrix(out[0]) << " =\n"
                  << "        " << emit_matrix(args[0]) << ".transpose();\n";
            m_out.indent--;
            m_out << "}\n";
        }
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error(
            "Axis permutation in reshape is not implemented yet for tensors with rank>2");
    }
}

void runtime::cpu::CPU_Emitter::EmitFunctionCall(
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto function_call = static_cast<const op::FunctionCall*>(n);
    shared_ptr<Function> function = function_call->get_function();

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

void runtime::cpu::CPU_Emitter::EmitReduce(const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto reduce = static_cast<const op::Reduce*>(n);
    auto reduction_function = reduce->get_function();

    auto reductee_shape = args[0].get_shape();

    auto& f_result_element_type = out[0].get_element_type();
    auto result_shape = out[0].get_shape();

    auto& reduction_axes = reduce->get_reduction_axes();

    // Trivial case: no reduction axes (this includes the scalar-reductee case).
    if (reduction_axes.empty())
    {
        m_out << "{   // " << n->get_name() << " 1\n";
        m_out.indent++;
        m_out << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
              << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        m_out.indent--;
        m_out << "}\n";
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
            m_out << "{   // " << n->get_name() << " 2\n";
            m_out.indent++;
            m_out << "memcpy(" << out[0].get_name() << ", " << args[1].get_name() << ", "
                  << out[0].get_size() * out[0].get_element_type().size() << ");\n";
            m_out.indent--;
            m_out << "}\n";
        }
        else
        {
            m_out << "{   // " << n->get_name() << " 3\n";
            m_out.indent++;
            string type = f_result_element_type.c_type_string();
            m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            m_out.indent++;
            m_out << "\n";
            m_out << type << " result;\n";
            m_out << "void* args[] = {&x, &y};\n";
            m_out << "void* out[] = {&result};\n";
            m_out << reduction_function->get_name() << "(args, out);\n";
            m_out << "return result;\n";
            m_out.indent--;
            m_out << "};\n";
            m_out << emit_array1d(out[0]) << " =\n"
                  << "    " << emit_array1d(args[0]) << ".redux(f);\n";
            m_out.indent--;
            m_out << "}\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{1})
    {
        if (reductee_shape.at(1) == 0)
        {
            m_out << "{   // " << n->get_name() << " 4\n";
            m_out.indent++;
            m_out << emit_array1d(out[0]) << " =\n"
                  << "    " << emit_array1d(args[1]) << "(0, 0);\n";
            m_out.indent--;
            m_out << "}\n";
        }
        else
        {
            // shared_ptr<CallFrame> cf =
            //     dynamic_pointer_cast<CallFrame>(external->make_call_frame());
            // ef->get_callees().emplace_back(cf);

            m_out << "{   // " << n->get_name() << " 5\n";
            m_out.indent++;
            string type = f_result_element_type.c_type_string();
            m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            m_out.indent++;
            m_out << "\n";
            m_out << type << " result;\n";
            m_out << "void* args[] = {&x, &y};\n";
            m_out << "void* out[] = {&result};\n";
            m_out << reduction_function->get_name() << "(args, out);\n";
            m_out << "return result;\n";
            m_out.indent--;
            m_out << "};\n";
            m_out << emit_vector(out[0]) << " =\n"
                  << "        " << emit_matrix(args[0]) << ".rowwise().redux(f);\n";
            m_out.indent--;
            m_out << "}\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{0})
    {
        if (reductee_shape.at(0) == 0)
        {
            m_out << "{   // " << n->get_name() << " 6\n";
            m_out.indent++;
            m_out << emit_array1d(out[0]) << " =\n"
                  << "    " << emit_array1d(args[1]) << "(0, 0);\n";
            m_out.indent--;
            m_out << "}\n";
        }
        else
        {
            m_out << "{   // " << n->get_name() << " 7\n";
            m_out.indent++;
            string type = f_result_element_type.c_type_string();
            m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            m_out.indent++;
            m_out << "\n";
            m_out << type << " result;\n";
            m_out << "void* args[] = {&x, &y};\n";
            m_out << "void* out[] = {&result};\n";
            m_out << reduction_function->get_name() << "(args, out);\n";
            m_out << "return result;\n";
            m_out.indent--;
            m_out << "};\n";
            m_out << emit_vector(out[0]) << " =\n"
                  << "    " << emit_matrix(args[0]) << ".colwise().redux(f);\n";
            m_out.indent--;
            m_out << "}\n";
        }
    }
    else
    {
        string type = f_result_element_type.c_type_string();
        m_out << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
        m_out.indent++;
        m_out << "\n";
        m_out << type << " result;\n";
        m_out << "void* args[] = {&x, &y};\n";
        m_out << "void* out[] = {&result};\n";
        m_out << reduction_function->get_name() << "(args, out);\n";
        m_out << "return result;\n";
        m_out.indent--;
        m_out << "};\n";

        m_out << "kernel::reduce<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        m_out << "               " << args[1].get_name() << ",\n";
        m_out << "               " << out[0].get_name() << ",\n";
        m_out << "               {" << join(args[0].get_shape()) << "},\n";
        m_out << "               {" << join(out[0].get_shape()) << "},\n";
        m_out << "               {" << join(reduce->get_reduction_axes()) << "},\n";
        m_out << "               f);\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitSign(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".sign();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSlice(const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    const op::Slice* slice = static_cast<const op::Slice*>(n);

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
        m_out << "{   // " << n->get_name() << " 1\n";
        m_out.indent++;
        m_out << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
              << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (!strided && arg_rank == 1)
    {
        m_out << "{   // " << n->get_name() << " 2\n";
        m_out.indent++;
        m_out << emit_vector(out[0]) << " =\n"
              << "    " << emit_vector(args[0]) << ".segment(\n"
              << "        " << to_string(lower_bounds[0]) << ", "
              << to_string(upper_bounds[0] - lower_bounds[0]) << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (!strided && arg_rank == 2)
    {
        m_out << "{   // " << n->get_name() << " 3\n";
        m_out.indent++;
        m_out << emit_matrix(out[0]) << " = \n"
              << "        " << emit_matrix(args[0]) << ".block(" << to_string(lower_bounds[0])
              << ", " << to_string(lower_bounds[1]) << ",\n"
              << "        " << to_string(upper_bounds[0] - lower_bounds[0]) << ",\n"
              << "        " << to_string(upper_bounds[1] - lower_bounds[1]) << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        m_out << "kernel::slice<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        m_out << "                         " << out[0].get_name() << ",\n";
        m_out << "                         {" << join(args[0].get_shape()) << "},\n";
        m_out << "                         {" << join(slice->get_lower_bounds()) << "},\n";
        m_out << "                         {" << join(slice->get_upper_bounds()) << "},\n";
        m_out << "                         {" << join(slice->get_strides()) << "},\n";
        m_out << "                         {" << join(out[0].get_shape()) << "});\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitSum(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    const op::Sum* sum = static_cast<const op::Sum*>(n);
    const Shape& arg_shape = args[0].get_shape();
    size_t arg_rank = arg_shape.size();
    const AxisSet& reduction_axes = sum->get_reduction_axes();

    // Trivial case: no reduction axes.
    if (reduction_axes.size() == 0)
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
              << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    // Full reduction? Then sum to scalar.
    else if ((arg_rank == 1 && reduction_axes == AxisSet{0}) ||
             (arg_rank == 2 && reduction_axes == AxisSet{0, 1}))
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_array1d(out[0]) << " =\n"
              << "    " << emit_array1d(args[0]) << ".sum();\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{1})
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_vector(out[0]) << " =\n"
              << "    " << emit_matrix(args[0]) << ".rowwise().sum();\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{0})
    {
        m_out << "{   // " << n->get_name() << "\n";
        m_out.indent++;
        m_out << emit_vector(out[0]) << " =\n"
              << "    " << emit_matrix(args[0]) << ".colwise().sum();\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else
    {
        m_out << "kernel::sum<" << out[0].get_type() << ">(" << args[0].get_name() << ",\n";
        m_out << "                         " << out[0].get_name() << ",\n";
        m_out << "                         {" << join(args[0].get_shape()) << "},\n";
        m_out << "                         {" << join(out[0].get_shape()) << "},\n";
        m_out << "                         {" << join(sum->get_reduction_axes()) << "});\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitExp(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".exp();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSin(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".sin();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSinh(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".sinh();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitCos(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".cos();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitCosh(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".cosh();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitTan(const ngraph::Node* n,
                                        const vector<runtime::cpu::TensorViewWrapper>& args,
                                        const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".tan();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitTanh(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    // Eigen's generic_fast_tanh_float<float> is currently miscompiled by Clang/LLVM
    // so we fall-back to tanh
    // TODO: Implement our own internal fast/approximate tanh if this actually gets used
    // by models
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << "for (size_t i=0; i<" << out[0].get_size() << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = tanh(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAsin(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".asin();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAcos(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".acos();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitAtan(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " =\n"
          << "    " << emit_array1d(args[0]) << ".atan();\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitPower(const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    m_out << emit_array1d(out[0]) << " = \n";
    m_out.indent++;
    m_out << emit_array1d(args[0]) << ".pow(\n ";
    m_out << emit_array1d(args[1]) << ");\n";
    m_out.indent -= 2;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitReplaceSlice(
    const ngraph::Node* n,
    const vector<runtime::cpu::TensorViewWrapper>& args,
    const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto replace_slice = static_cast<const op::Slice*>(n);

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
        m_out << "{   // " << n->get_name() << " 1\n";
        m_out.indent++;
        m_out << "memcpy(" << out[0].get_name() << ", " << args[1].get_name() << ", "
              << out[0].get_size() * out[0].get_element_type().size() << ");\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (!strided && arg0_rank == 1)
    {
        m_out << "{   // " << n->get_name() << " 2\n";
        m_out.indent++;
        m_out << emit_vector(out[0]) << " =\n"
              << "    " << emit_vector(args[0]) << ";\n"
              << emit_vector(out[0]) << ".segment(\n"
              << "    " << to_string(lower_bounds[0]) << ", "
              << to_string(upper_bounds[0] - lower_bounds[0]) << ") =\n"
              << "    " << emit_vector(args[1]) << ";\n";
        m_out.indent--;
        m_out << "}\n";
    }
    else if (!strided && arg0_rank == 2)
    {
        m_out << "{   // " << n->get_name() << " 3\n";
        m_out.indent++;
        m_out << emit_matrix(out[0]) << " =\n"
              << "    " << emit_matrix(args[0]) << ";\n"
              << emit_matrix(out[0]) << ".block(\n"
              << "        " << to_string(lower_bounds[0]) << ",\n"
              << "        " << to_string(lower_bounds[1]) << ",\n"
              << "        " << to_string(upper_bounds[0] - lower_bounds[0]) << ",\n"
              << "        " << to_string(upper_bounds[1] - lower_bounds[1]) << ") =\n"
              << "    " << emit_matrix(args[1]) << ";\n";
        m_out.indent--;
        m_out << "}\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        m_out << "kernel::replace_slice<" << out[0].get_type() << ">(" << args[0].get_name()
              << ",\n";
        m_out << "                         " << args[1].get_name() << ",\n";
        m_out << "                         " << out[0].get_name() << ",\n";
        m_out << "                         {" << join(args[1].get_shape()) << "},\n";
        m_out << "                         {" << join(replace_slice->get_lower_bounds()) << "},\n";
        m_out << "                         {" << join(replace_slice->get_upper_bounds()) << "},\n";
        m_out << "                         {" << join(replace_slice->get_strides()) << "},\n";
        m_out << "                         {" << join(out[0].get_shape()) << "});\n";
    }
}

void runtime::cpu::CPU_Emitter::EmitOneHot(const ngraph::Node* n,
                                           const vector<runtime::cpu::TensorViewWrapper>& args,
                                           const vector<runtime::cpu::TensorViewWrapper>& out)
{
    auto oh = static_cast<const op::OneHot*>(n);

    auto arg_rank = args[0].get_shape().size();

    size_t bounds = out[0].get_shape()[oh->get_one_hot_axis()];

    if (arg_rank == 0)
    {
        m_out << "{   // " << n->get_name() << " 1\n";
        m_out.indent++;

        m_out << emit_vector(out[0], "out_vector") << ";\n";

        m_out << "out_vector.setZero();\n"
              << ""
              << "auto pos_raw = " << emit_vector(args[0]) << "(0, 0);\n"
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

        m_out << emit_vector(args[0], "arg_vector") << ";\n";

        m_out << emit_matrix(out[0], "out_vector") << ";\n";
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

void runtime::cpu::CPU_Emitter::EmitCeiling(const ngraph::Node* n,
                                            const vector<runtime::cpu::TensorViewWrapper>& args,
                                            const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    size_t element_count = out[0].get_size();
    m_out << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = ceil(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitFloor(const ngraph::Node* n,
                                          const vector<runtime::cpu::TensorViewWrapper>& args,
                                          const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    size_t element_count = out[0].get_size();
    m_out << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = floor(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitSqrt(const ngraph::Node* n,
                                         const vector<runtime::cpu::TensorViewWrapper>& args,
                                         const vector<runtime::cpu::TensorViewWrapper>& out)
{
    m_out << "{   // " << n->get_name() << "\n";
    m_out.indent++;
    size_t element_count = out[0].get_size();
    m_out << "for (size_t i = 0; i < " << element_count << "; i++)\n";
    m_out << "{\n";
    m_out << "    " << out[0].get_name() << "[i] = sqrt(" << args[0].get_name() << "[i]);\n";
    m_out << "}\n";
    m_out.indent--;
    m_out << "}\n";
}

void runtime::cpu::CPU_Emitter::EmitConvolution(const ngraph::Node* n,
                                                const vector<runtime::cpu::TensorViewWrapper>& args,
                                                const vector<runtime::cpu::TensorViewWrapper>& out)
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
          << "});\n";
}

//------------------------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------------------------

void runtime::cpu::CPU_Emitter::generate_call(const vector<runtime::cpu::TensorViewWrapper>& args,
                                              const vector<runtime::cpu::TensorViewWrapper>& out,
                                              shared_ptr<Function> function)
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
