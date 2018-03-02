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

#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/allreduce.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concat.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/max.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/min.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/ops/pad.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/product.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/reduce_window.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/ops/remainder.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/select_and_scatter.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_emitters.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/ops/convert_layout.hpp"
#include "ngraph/runtime/cpu/ops/matmul_bias.hpp"
#include "ngraph/types/element_type.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include <mpi.h>
#include "ngraph/ops/allreduce.hpp"
#endif

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

void runtime::cpu::CPU_Emitter::emit_mkldnn_preamble(codegen::CodeWriter& writer)
{
    writer << "// MKLDNN Preamble\n";
    writer << "#include <mkldnn.hpp>\n";
    writer << "using namespace mkldnn;\n\n";
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Add)
            {
                // TODO: Audit all uses of Add and fix this to use
                // the right alignment instead of Eigen::Unaligned
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << "Eigen::Map<Eigen::Array<" << out[0].get_element_type().c_type_string()
                       << ", " << out[0].get_size() << ", 1>, Eigen::Unaligned> out("
                       << out[0].get_name() << ");\n";
                writer << "Eigen::Map<Eigen::Array<" << args[0].get_element_type().c_type_string()
                       << ", " << args[0].get_size() << ", 1>, Eigen::Unaligned> arg0("
                       << args[0].get_name() << ");\n";
                writer << "Eigen::Map<Eigen::Array<" << args[1].get_element_type().c_type_string()
                       << ", " << args[1].get_size() << ", 1>, Eigen::Unaligned> arg1("
                       << args[1].get_name() << ");\n";
                writer << "out = arg0 + arg1;\n";
#else

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    std::vector<float> scale_vector(2, 1);
                    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

                    auto input0_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto input1_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 1);
                    auto result_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input0_data_desc =
                        mkldnn_emitter->build_memory_descriptor(args[0], input0_format);
                    auto input1_data_desc =
                        mkldnn_emitter->build_memory_descriptor(args[1], input1_format);
                    auto result_desc =
                        mkldnn_emitter->build_memory_descriptor(out[0], result_format);
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input0_data_desc, runtime::cpu::mkldnn_utils::global_cpu_engine));
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input1_data_desc, runtime::cpu::mkldnn_utils::global_cpu_engine));

                    size_t add_index = 0;
                    add_index = mkldnn_emitter->build_elementwise_add(
                        input0_data_desc, input1_data_desc, result_desc, scale_vector, inputs_pd);
                    auto& deps = mkldnn_emitter->get_primitive_deps(add_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(add_index) << ");\n";
                }
                else
                {
                    writer << "#pragma omp parallel for\n";
                    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                    writer << "{\n";
                    writer << "    " << out[0].get_name() << "[i] = " << args[0].get_name()
                           << "[i] + " << args[1].get_name() << "[i];\n";
                    writer << "}\n";
                }
#endif
                writer.indent--;
                writer << "}\n";
            }

#ifdef NGRAPH_DISTRIBUTED
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AllReduce)
            {
                const element::Type& element_type = args[0].get_element_type();
                auto data_type = "MPI_FLOAT";

                if (element_type == element::f32)
                {
                    data_type = "MPI_FLOAT";
                }
                else if (element_type == element::f64)
                {
                    data_type = "MPI_DOUBLE";
                }

                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
                writer << "MPI_Allreduce(" << args[0].get_name() << ", " << out[0].get_name()
                       << ", " << out[0].get_size() << ", " << data_type
                       << ", MPI_SUM, MPI_COMM_WORLD);\n";
                writer.indent--;
                writer << "}\n";
            }
#endif

            //TODO: This could be further optimized to reduce the impact of memcpy by either
            //a) emitting customized code for initializing output/bias
            //b) emitting two cblas calls (one for gemm on W and x and the second for gemm on Bias and E^T + the result of the first gemm)
            //@jbobba suggests b) is more efficient but we should benchmark both
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MatmulBias)
            {
                const ngraph::op::MatmulBias* cg = static_cast<const ngraph::op::MatmulBias*>(node);

                const Shape& arg0_shape = cg->get_arg0_shape(); //W
                const Shape& arg1_shape = cg->get_arg1_shape(); //x
                const Shape& arg2_shape = args[2].get_shape();  //bias (C)

                static const char* ctranspose = "cblas::Transpose::Transpose, ";
                static const char* cnotranspose = "cblas::Transpose::None, ";

                size_t m = arg0_shape[0];
                size_t n = arg1_shape[1];
                size_t k = arg0_shape[1];
                //
                const char* tranpose_a = cnotranspose;
                const char* tranpose_b = cnotranspose;
                size_t lda = arg0_shape[1];
                size_t ldb = arg1_shape[1];

                if (cg->get_is_arg0_transposed())
                {
                    tranpose_a = ctranspose;
                    m = arg0_shape[1];
                    k = arg0_shape[0];
                }

                if (cg->get_is_arg1_transposed())
                {
                    tranpose_b = ctranspose;
                    n = arg1_shape[0];
                }

                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;

                writer << "memcpy(" << out[0].get_name() << ", " << args[2].get_name() << ", "
                       << out[0].get_size() * out[0].get_element_type().size() << ");\n";

                writer << "cblas::cblas_sgemm("
                       << "cblas::Layout::RowMajor, " << tranpose_a << tranpose_b << m << ", " << n
                       << ", " << k << ",\n"
                       << "        1.0f, " << args[0].get_name() << ", " << max(1UL, lda) << ", "
                       << args[1].get_name() << ", " << max(1UL, ldb) << ", 1.0f,\n"
                       << "        " << out[0].get_name() << ", " << max(1UL, arg2_shape[1])
                       << ");\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNorm)
            {
                const ngraph::op::BatchNorm* batchnorm =
                    static_cast<const ngraph::op::BatchNorm*>(node);

                // get the shape of all the inputs and output to batchnorm
                auto gamma_shape = args[0].get_shape();
                auto beta_shape = args[1].get_shape();
                auto input_shape = args[2].get_shape();
                auto result_shape = out[0].get_shape();
                auto mean_shape = out[1].get_shape();
                auto variance_shape = out[2].get_shape();

                // get input element type
                const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                    args[2].get_element_type());

                writer << "{\n";
                writer.indent++;

                // define weights
                writer << "std::vector<" << args[0].get_element_type().c_type_string()
                       << ">bn_weights(2*" << input_shape[1] << ");\n";
                auto weights_shape = Shape{2, input_shape[1]};

                // push gamma and beta
                writer << "auto gamma = " << args[0].get_name() << ";\n";
                writer << "auto beta = " << args[1].get_name() << ";\n";

                writer << "memcpy(&bn_weights[0], gamma,"
                       << args[1].get_size() * args[0].get_element_type().size() << ");\n";
                writer << "memcpy(&bn_weights[0]+" << args[1].get_size() << ", beta, "
                       << args[1].get_size() * args[1].get_element_type().size() << ");\n";

                // get the eps value from the bn node
                writer << "auto epsilon = " << batchnorm->get_eps_value() << ";\n";

                const string& input_format = runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                    runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 2));
                const string& result_format = runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                    runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0));
                // Bind to CPU engine
                writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                // create memory descriptors
                writer << "memory::desc input_data_desc = memory::desc({" << join(input_shape)
                       << "}, " << et << ", " << input_format << ");\n";
                // TODO define weights by stacking gamma and beta values
                writer << "memory::desc weights_desc = memory::desc({" << join(weights_shape)
                       << "}, " << et << ", memory::format::nc);\n";
                writer << "memory::desc result_desc = memory::desc({" << join(result_shape) << "}, "
                       << et << ", " << result_format << ");\n";
                writer << "memory::desc mean_desc = memory::desc({" << join(mean_shape) << "}, "
                       << et << ", memory::format::x);\n";
                writer << "memory::desc variance_desc = memory::desc({" << join(variance_shape)
                       << "}, " << et << ", memory::format::x);\n";

                // Define memory for the user data
                writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                       << args[2].get_name() << ");\n";
                writer << "memory weights = memory({weights_desc, cpu_engine}, bn_weights.data()"
                       << ");\n";
                writer << "memory result = memory({result_desc, cpu_engine}, " << out[0].get_name()
                       << ");\n";
                writer << "memory mean = memory({mean_desc, cpu_engine}, " << out[1].get_name()
                       << ");\n";
                writer << "memory variance = memory({variance_desc, cpu_engine}, "
                       << out[2].get_name() << ");\n";

                // create batchnorm descriptor
                writer << "batch_normalization_forward::desc bn_fprop_desc = "
                          "batch_normalization_forward::desc(forward_training,"
                       << "input_data_desc, epsilon, use_scale_shift);\n";
                // bn fprop primitive descriptor
                writer
                    << "batch_normalization_forward::primitive_desc bn_fprop_prim_desc = "
                       "batch_normalization_forward::primitive_desc(bn_fprop_desc, cpu_engine);\n";

                // create a batchnorm fprop primitive
                writer << "batch_normalization_forward bn_fprop = "
                          "batch_normalization_forward(bn_fprop_prim_desc, "
                          "primitive::at(input_data),"
                       << "primitive::at(weights), result, mean, variance); \n";

                // create stream and execute
                writer << "stream s = stream(stream::kind::eager);\n"
                       << "s.submit({bn_fprop}).wait();\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormBackprop)
            {
                const ngraph::op::BatchNormBackprop* batchnorm =
                    static_cast<const ngraph::op::BatchNormBackprop*>(node);
                auto gamma_shape = args[0].get_shape();
                auto beta_shape = args[1].get_shape();
                auto input_shape = args[2].get_shape();
                auto mean_shape = args[3].get_shape();
                auto variance_shape = args[4].get_shape();
                auto delta_shape = args[5].get_shape();
                auto result_shape = out[0].get_shape();

                // get input element type
                const string& et =
                    mkldnn_utils::get_mkldnn_data_type_string(args[2].get_element_type());
                writer << "{\n";
                writer.indent++;
                // define weights
                writer << "std::vector<" << args[0].get_element_type().c_type_string()
                       << ">bn_weights(" << input_shape[1] * 2 << ");\n";
                writer << "std::vector<" << args[0].get_element_type().c_type_string()
                       << ">vdiff_weights(" << input_shape[1] * 2 << ");\n";
                auto weights_shape = Shape{2, input_shape[1]};

                // push gamma and beta
                writer << "auto gamma = " << args[0].get_name() << ";\n";
                writer << "auto beta = " << args[1].get_name() << ";\n";

                writer << "memcpy(&bn_weights[0], gamma,"
                       << args[1].get_size() * args[0].get_element_type().size() << ");\n";
                writer << "memcpy(&bn_weights[0]+" << args[1].get_size() << ", beta, "
                       << args[1].get_size() * args[1].get_element_type().size() << ");\n";

                // get the eps value from the bn node
                writer << "auto epsilon = " << batchnorm->get_eps_value() << ";\n";
                // Bind to CPU engine
                writer << "using namespace mkldnn; \n";
                writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                // create memory descriptors
                writer << "memory::desc input_data_desc = memory::desc({" << join(input_shape)
                       << "}, " << et << ", memory::format::nchw);\n";
                // TODO define weights by stacking gamma and beta values
                writer << "memory::desc weights_desc = memory::desc({" << join(weights_shape)
                       << "}, " << et << ", memory::format::nc);\n";
                writer << "memory::desc diff_weights_desc = memory::desc({" << join(weights_shape)
                       << "}, " << et << ", memory::format::nc);\n";
                writer << "memory::desc result_desc = memory::desc({" << join(result_shape) << "}, "
                       << et << ", memory::format::nchw);\n";
                writer << "memory::desc mean_desc = memory::desc({" << join(mean_shape) << "}, "
                       << et << ", memory::format::x);\n";
                writer << "memory::desc variance_desc = memory::desc({" << join(variance_shape)
                       << "}, " << et << ", memory::format::x);\n";
                writer << "memory::desc delta_desc = memory::desc({" << join(input_shape) << "}, "
                       << et << ", memory::format::nchw);\n";

                // Define memory for the user data
                writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                       << args[2].get_name() << ");\n";
                writer << "memory weights = memory({weights_desc, cpu_engine}, bn_weights.data()"
                       << ");\n";
                writer << "memory diff_weights = memory({diff_weights_desc, cpu_engine}, "
                          "vdiff_weights.data()"
                       << ");\n";
                writer << "memory mean = memory({mean_desc, cpu_engine}, " << args[3].get_name()
                       << ");\n";
                writer << "memory variance = memory({variance_desc, cpu_engine}, "
                       << args[4].get_name() << ");\n";
                writer << "memory delta = memory({delta_desc, cpu_engine}, " << args[5].get_name()
                       << ");\n";
                writer << "memory result = memory({result_desc, cpu_engine}, " << out[0].get_name()
                       << ");\n";

                //create fprop batchnorm descriptor
                writer << "batch_normalization_forward::desc bn_fprop_desc = "
                          "batch_normalization_forward::desc(forward_training,"
                       << "input_data_desc, epsilon, use_scale_shift);\n";
                //bn fprop primitive descriptor
                writer
                    << "batch_normalization_forward::primitive_desc bn_fprop_prim_desc = "
                       "batch_normalization_forward::primitive_desc(bn_fprop_desc, cpu_engine);\n";

                //create bprop batchnorm descriptor
                writer << "batch_normalization_backward::desc bn_bprop_desc = "
                          "batch_normalization_backward::desc(backward, delta_desc, "
                          "input_data_desc, epsilon, use_scale_shift);\n";

                //bn bprop primitive descriptor
                writer << "batch_normalization_backward::primitive_desc bn_bprop_prim_desc = "
                          "batch_normalization_backward::primitive_desc(bn_bprop_desc, cpu_engine, "
                          "bn_fprop_prim_desc);\n";

                //create a batchnorm fprop primitive
                writer << " batch_normalization_backward bn_bprop = "
                          "batch_normalization_backward(bn_bprop_prim_desc, input_data, mean, "
                          "variance, delta, weights, result, diff_weights);\n ";

                //create stream and execute
                writer << "stream s = stream(stream::kind::eager);\n"
                       << "s.submit({bn_bprop}).wait();\n";

                writer << "memcpy(" << out[1].get_name() << ",&vdiff_weights[0],"
                       << args[1].get_size() * args[0].get_element_type().size() << ");\n";
                writer << "memcpy(" << out[2].get_name() << ",&vdiff_weights[0] + "
                       << args[1].get_size() << ","
                       << args[1].get_size() * args[1].get_element_type().size() << ");\n";

                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dot)
            {
                const ngraph::op::Dot* dot = static_cast<const ngraph::op::Dot*>(node);

                const Shape& arg0_shape = args[0].get_shape();
                const Shape& arg1_shape = args[1].get_shape();
                if (arg0_shape.empty() || arg1_shape.empty())
                {
                    auto& first = (arg0_shape.empty() ? args[0] : args[1]);
                    auto& second = (arg0_shape.empty() ? args[1] : args[0]);

                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << "\n    = ";
                    writer << first.get_name() << "[0]\n    * " << emit_vector(second) << ";\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1) &&
                         dot->get_reduction_axes_count() == 1)
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " << \n"
                           << "    " << emit_vector(args[0]) << ".dot(" << emit_vector(args[1])
                           << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) &&
                         dot->get_reduction_axes_count() == 1)
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " = \n"
                           << "    " << emit_matrix(args[0]) << " * " << emit_vector(args[1])
                           << ";\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2) &&
                         dot->get_reduction_axes_count() == 1)
                {
                    // Emit an MKL SGEMM call if possible
                    if (args[0].get_element_type() == element::f32)
                    {
                        writer << "{   // " << node->get_name() << "\n";
                        writer.indent++;
                        writer << "cblas::cblas_sgemm("
                               << "cblas::Layout::RowMajor, "
                               << "cblas::Transpose::None, "
                               << "cblas::Transpose::None, " << arg0_shape[0] << ", "
                               << arg1_shape[1] << ", " << arg0_shape[1] << ",\n"
                               << "        1.0f, " << args[0].get_name() << ", "
                               << max(1UL, arg0_shape[1]) << ", " << args[1].get_name() << ", "
                               << max(1UL, arg1_shape[1]) << ", 0.0f,\n"
                               << "        " << out[0].get_name() << ", " << max(1UL, arg1_shape[1])
                               << ");\n";
                        writer.indent--;
                        writer << "}\n";
                    }
                    else
                    {
                        writer << "{   // " << node->get_name() << "\n";
                        writer.indent++;
                        writer << emit_matrix(out[0]) << " = \n"
                               << "    " << emit_matrix(args[0]) << " * " << emit_matrix(args[1])
                               << ";\n";
                        writer.indent--;
                        writer << "}\n";
                    }
                }
                else
                {
                    writer << "kernel::dot(" << args[0].get_name() << ",\n";
                    writer << "            " << args[1].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(args[0].get_shape()) << "},\n";
                    writer << "            {" << join(args[1].get_shape()) << "},\n";
                    writer << "            {" << join(out[0].get_shape()) << "},\n";
                    writer << "            " << dot->get_reduction_axes_count() << ");\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Multiply)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GetOutputElement)
            {
                auto get_tuple_element = static_cast<const ngraph::op::GetOutputElement*>(node);

                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
                writer << "memcpy(" << out[0].get_name() << ", "
                       << args[get_tuple_element->get_n()].get_name() << ", "
                       << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Abs)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n";
                writer << "Eigen::abs(" << emit_array1d(args[0]) << ");\n";
#else
                // Some C++ implementations don't like it when we call std::abs on unsigned types, so we will
                // avoid doing so here.
                auto& result_element_type = out[0].get_element_type();

                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name()
                       << "[i] = " << (result_element_type.is_signed() ? "std::abs" : "") << "("
                       << args[0].get_name() << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Concat)
            {
                auto result_shape = out[0].get_shape();

#if PREFER_EIGEN == 1
                if (result_shape.size() == 1)
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0], "out_vector") << ";\n";

                    size_t concat_pos = 0;
                    for (size_t i = 0; i < args.size(); i++)
                    {
                        writer << "out_vector.segment(" << concat_pos << ", "
                               << args[i].get_shape().at(0) << ") << " << emit_vector(args[i])
                               << ";\n";
                        concat_pos += args[i].get_shape().at(0);
                    }
                    writer.indent--;
                    writer << "}\n";
                }
                else if (result_shape.size() == 2)
                {
                    auto axis =
                        (dynamic_cast<const ngraph::op::Concat*>(node))->get_concatenation_axis();

                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_matrix(out[0], "out_matrix") << ";\n";

                    size_t concat_pos[2]{0, 0};
                    for (size_t i = 0; i < args.size(); i++)
                    {
                        auto& arg_shape = args[i].get_shape();

                        writer << "out_matrix.block(" << concat_pos[0] << ", " << concat_pos[1]
                               << ", " << arg_shape.at(0) << ", " << arg_shape.at(1) << ") << "
                               << emit_matrix(args[i]) << ";\n";

                        concat_pos[axis] += arg_shape.at(axis);
                    }

                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    if (s_use_ref_kernels)
                    {
                        auto axis = (dynamic_cast<const ngraph::op::Concat*>(node))
                                        ->get_concatenation_axis();

                        std::vector<std::string> arg_names;
                        std::vector<std::string> arg_shape_strings;

                        for (auto arg : args)
                        {
                            arg_names.push_back(arg.get_name());
                            arg_shape_strings.push_back("{" + join(arg.get_shape()) + "}");
                        }

                        writer << "kernel::concat<" << out[0].get_type() << ">({" << join(arg_names)
                               << "},\n";
                        writer << "                         " << out[0].get_name() << ",\n";
                        writer << "                         {" << join(arg_shape_strings) << "},\n";
                        writer << "                         {" << join(result_shape) << "},\n";
                        writer << "                         " << axis << ");\n";
                    }
                    else
                    {
                        auto axis = (dynamic_cast<const ngraph::op::Concat*>(node))
                                        ->get_concatenation_axis();

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
                auto axis =
                    (dynamic_cast<const ngraph::op::Concat*>(node))->get_concatenation_axis();

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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Divide)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
                if (node->get_element_type().is_real() == false)
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Equal)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Greater)
            {
                writer << "{   // " << node->get_name() << " xxx\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GreaterEq)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Less)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::LessEq)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Log)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    Eigen::log(" << emit_array1d(args[0]) << ");\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = log(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Maximum)
            {
                writer << "{   // " << node->get_name() << "\n";
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
                       << args[1].get_name() << "[i] ? " << args[0].get_name()
                       << "[i] : " << args[1].get_name() << "[i] ;\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Minimum)
            {
                writer << "{   // " << node->get_name() << "\n";
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
                       << args[1].get_name() << "[i] ? " << args[0].get_name()
                       << "[i] : " << args[1].get_name() << "[i] ;\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Negative)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    -" << emit_array1d(args[0]) << ";\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = -" << args[0].get_name()
                       << "[i];\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::NotEqual)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Select)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Subtract)
            {
                writer << "{   // " << node->get_name() << "\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Broadcast)
            {
                auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);

                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                if (broadcast->get_broadcast_axes().empty())
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
                           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (arg_shape.size() == 0)
                {
                    writer << "{   // " << node->get_name() << "\n";
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
                        writer << "{   // " << node->get_name() << "\n";
                        writer.indent++;
                        writer << emit_matrix(out[0]) << ".colwise() =\n"
                               << "    " << emit_vector(args[0]) << ";\n";
                        writer.indent--;
                        writer << "}\n";
                    }
                    else if (broadcast->get_broadcast_axes() == AxisSet{0})
                    {
                        writer << "{   // " << node->get_name() << "\n";
                        writer.indent++;

                        writer << "Eigen::Map<Eigen::Matrix<"
                               << out[0].get_element_type().c_type_string() << ", "
                               << join(out[0].get_shape())
                               << ", Eigen::RowMajor>, Eigen::Aligned64, Eigen::Stride<"
                               << join(out[0].get_strides()) << ">> out(" << out[0].get_name()
                               << ");\n";
                        writer << "Eigen::Map<Eigen::Matrix<"
                               << args[0].get_element_type().c_type_string() << ", 1, "
                               << args[0].get_size()
                               << ", Eigen::RowMajor>, Eigen::Aligned64, Eigen::Stride<"
                               << args[0].get_size() << ", 1>> arg0(" << args[0].get_name()
                               << ");\n";
                        writer << "out = arg0.replicate<" << out[0].get_shape().at(0)
                               << ", 1>();\n";

                        writer.indent--;
                        writer << "}\n";
                    }
                    else
                    {
                        throw ngraph_error(
                            "Internal error: axis set for vector-matrix broadcast is neither {0} "
                            "nor "
                            "{1}");
                    }
                }
                else
                {
                    writer << "kernel::broadcast<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(arg_shape) << "},\n";
                    writer << "                         {" << join(result_shape) << "},\n";
                    writer << "                         {" << join(broadcast->get_broadcast_axes())
                           << "});\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Convert)
            {
                auto& result_element_type = out[0].get_element_type();

                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << "\n"
                       << "    .template cast<" << result_element_type.c_type_string() << ">();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = ("
                       << result_element_type.c_type_string() << ")(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Constant)
            {
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reshape)
            {
                auto reshape = static_cast<const ngraph::op::Reshape*>(node);
                writer << "{   // " << node->get_name() << "\n";
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
                    writer << "{   // " << node->get_name() << " 1\n";
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
                    if (result_element_type == ngraph::element::f32)
                    {
                        writer << "{   // " << node->get_name() << " 2\n";
                        writer.indent++;
                        writer << "mkl::MKL_Somatcopy('R', 'T', " << to_string(arg_shape[0])
                               << ",\n"
                               << "                   " << to_string(arg_shape[1]) << ", 1.0f,\n"
                               << "                   " << args[0].get_name() << ", "
                               << to_string(arg_shape[1]) << ",\n"
                               << "                   " << out[0].get_name() << ", "
                               << to_string(arg_shape[0]) << ");\n";
                        writer.indent--;
                        writer << "}\n";
                    }
                    else
                    {
                        writer << "{   // " << node->get_name() << " 3\n";
                        writer.indent++;
                        writer << emit_matrix(out[0]) << " =\n"
                               << "        " << emit_matrix(args[0]) << ".transpose();\n";
                        writer.indent--;
                        writer << "}\n";
                    }
                }
                // Other cases
                else
                {
                    writer << "kernel::reshape<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                " << out[0].get_name() << ",\n";
                    writer << "               {" << join(args[0].get_shape()) << "},\n";
                    writer << "               {" << join(reshape->get_input_order()) << "},\n";
                    writer << "               {" << join(out[0].get_shape()) << "}\n";
                    writer << "               );\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::FunctionCall)
            {
                auto function_call = static_cast<const ngraph::op::FunctionCall*>(node);
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
                    writer << function->get_name() << "(args, out, ctx);\n";
                }
                writer.indent--;
                writer << "}\n";
            }

            // TODO: This and other ops include comments/notes that
            // we don't want to just copy-paste here. Figure out a better way
            // or just point to ngvm/external_function.cpp with a note that
            // the compiled version of these ops is intended to have semantics identical
            // to what's seen there (for now atleast)

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reduce)
            {
                auto reduce = static_cast<const ngraph::op::Reduce*>(node);
                auto reduction_function = reduce->get_functions()[0];

                auto reductee_shape = args[0].get_shape();

                auto& f_result_element_type = out[0].get_element_type();
                auto result_shape = out[0].get_shape();

#if PREFER_EIGEN == 1
                auto& reduction_axes = reduce->get_reduction_axes();
                // Trivial case: no reduction axes (this includes the scalar-reductee case).
                if (reduction_axes.empty())
                {
                    writer << "{   // " << node->get_name() << " 1\n";
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
                    if (reductee_shape.at(0) == 0 ||
                        (reductee_shape.size() == 2 && reductee_shape.at(1) == 0))
                    {
                        writer << "{   // " << node->get_name() << " 2\n";
                        writer.indent++;
                        writer << "memcpy(" << out[0].get_name() << ", " << args[1].get_name()
                               << ", " << out[0].get_size() * out[0].get_element_type().size()
                               << ");\n";
                        writer.indent--;
                        writer << "}\n";
                    }
                    else
                    {
                        writer << "{   // " << node->get_name() << " 3\n";
                        writer.indent++;
                        string type = f_result_element_type.c_type_string();
                        writer << "auto f = [&](" << type << " x, " << type << " y) -> " << type
                               << "\n{";
                        writer.indent++;
                        writer << "\n";
                        writer << type << " result;\n";
                        writer << "void* args[] = {&x, &y};\n";
                        writer << "void* out[] = {&result};\n";
                        writer << reduction_function->get_name() << "(args, out, ctx);\n";
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
                        writer << "{   // " << node->get_name() << " 4\n";
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

                        writer << "{   // " << node->get_name() << " 5\n";
                        writer.indent++;
                        string type = f_result_element_type.c_type_string();
                        writer << "auto f = [&](" << type << " x, " << type << " y) -> " << type
                               << "\n{";
                        writer.indent++;
                        writer << "\n";
                        writer << type << " result;\n";
                        writer << "void* args[] = {&x, &y};\n";
                        writer << "void* out[] = {&result};\n";
                        writer << reduction_function->get_name() << "(args, out, ctx);\n";
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
                        writer << "{   // " << node->get_name() << " 6\n";
                        writer.indent++;
                        writer << emit_array1d(out[0]) << " =\n"
                               << "    " << emit_array1d(args[1]) << "(0, 0);\n";
                        writer.indent--;
                        writer << "}\n";
                    }
                    else
                    {
                        writer << "{   // " << node->get_name() << " 7\n";
                        writer.indent++;
                        string type = f_result_element_type.c_type_string();
                        writer << "auto f = [&](" << type << " x, " << type << " y) -> " << type
                               << "\n{";
                        writer.indent++;
                        writer << "\n";
                        writer << type << " result;\n";
                        writer << "void* args[] = {&x, &y};\n";
                        writer << "void* out[] = {&result};\n";
                        writer << reduction_function->get_name() << "(args, out, ctx);\n";
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
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;

                    string type = f_result_element_type.c_type_string();
                    writer << "auto f = [&](" << type << " x, " << type << " y) -> " << type
                           << "\n{";
                    writer.indent++;
                    writer << "\n";
                    writer << type << " result;\n";
                    writer << "void* args[] = {&x, &y};\n";
                    writer << "void* out[] = {&result};\n";
                    writer << reduction_function->get_name() << "(args, out, ctx);\n";
                    writer << "return result;\n";
                    writer.indent--;
                    writer << "};\n";

                    writer << "kernel::reduce<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
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
                writer << "{   // " << node->get_name() << " 1\n";
                writer.indent++;

                string type = f_result_element_type.c_type_string();

                writer << "auto f = [&](" << type << " x, " << type << " y) -> " << type << "\n{";
                writer.indent++;
                writer << "\n";
                writer << type << " result;\n";
                writer << "void* args[] = {&x, &y};\n";
                writer << "void* out[] = {&result};\n";
                writer << reduction_function->get_name() << "(args, out, ctx);\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sign)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".sign();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = (0 < " << args[0].get_name()
                       << "[i]) - (" << args[0].get_name() << "[i] < 0);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Slice)
            {
                const ngraph::op::Slice* slice = static_cast<const ngraph::op::Slice*>(node);

                writer << "{   // " << node->get_name() << "\n";
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
                    writer << "{   // " << node->get_name() << " 1\n";
                    writer.indent++;
                    writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
                           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (!strided && arg_rank == 1)
                {
                    writer << "{   // " << node->get_name() << " 2\n";
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
                    writer << "{   // " << node->get_name() << " 3\n";
                    writer.indent++;
                    writer << emit_matrix(out[0]) << " = \n"
                           << "        " << emit_matrix(args[0]) << ".block("
                           << to_string(lower_bounds[0]) << ", " << to_string(lower_bounds[1])
                           << ",\n"
                           << "        " << to_string(upper_bounds[0] - lower_bounds[0]) << ",\n"
                           << "        " << to_string(upper_bounds[1] - lower_bounds[1]) << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
                else
                {
                    writer << "kernel::slice<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(args[0].get_shape()) << "},\n";
                    writer << "                         {" << join(slice->get_lower_bounds())
                           << "},\n";
                    writer << "                         {" << join(slice->get_upper_bounds())
                           << "},\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sum)
            {
                const ngraph::op::Sum* sum = static_cast<const ngraph::op::Sum*>(node);
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                const Shape& arg_shape = args[0].get_shape();
                size_t arg_rank = arg_shape.size();
                const AxisSet& reduction_axes = sum->get_reduction_axes();

                // Trivial case: no reduction axes.
                if (reduction_axes.size() == 0)
                {
                    writer << "{   // " << node->get_name() << "\n";
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
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_array1d(out[0]) << " =\n"
                           << "    " << emit_array1d(args[0]) << ".sum();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (arg_rank == 2 && reduction_axes == AxisSet{1})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".rowwise().sum();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (arg_rank == 2 && reduction_axes == AxisSet{0})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".colwise().sum();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::sum<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(args[0].get_shape()) << "},\n";
                    writer << "                         {" << join(out[0].get_shape()) << "},\n";
                    writer << "                         {" << join(sum->get_reduction_axes())
                           << "});\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Exp)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".exp();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = exp(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sin)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".sin();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = sin(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sinh)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".sinh();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = sinh(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Cos)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".cos();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = cos(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Cosh)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".cosh();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = cosh(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tan)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".tan();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = tan(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tanh)
            {
                // Eigen's generic_fast_tanh_float<float> is currently miscompiled by Clang/LLVM
                // so we fall-back to tanh
                // TODO: Implement our own internal fast/approximate tanh if this actually gets used
                // by models
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 0
                writer << "#pragma omp parallel for\n";
#endif
                writer << "for (size_t i=0; i<" << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = tanh(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Asin)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".asin();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = asin(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Acos)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".acos();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = acos(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Atan)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                writer << emit_array1d(out[0]) << " =\n"
                       << "    " << emit_array1d(args[0]) << ".atan();\n";
#else
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = atan(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Power)
            {
                writer << "{   // " << node->get_name() << "\n";
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
                writer << "    " << out[0].get_name() << "[i] = pow(" << args[0].get_name()
                       << "[i], " << args[1].get_name() << "[i]);\n";
                writer << "}\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReplaceSlice)
            {
                auto replace_slice = static_cast<const ngraph::op::Slice*>(node);
                writer << "{   // " << node->get_name() << "\n";
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
                    writer << "{   // " << node->get_name() << " 1\n";
                    writer.indent++;
                    writer << "memcpy(" << out[0].get_name() << ", " << args[1].get_name() << ", "
                           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (!strided && arg0_rank == 1)
                {
                    writer << "{   // " << node->get_name() << " 2\n";
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
                    writer << "{   // " << node->get_name() << " 3\n";
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
                    writer << "kernel::replace_slice<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                         " << args[1].get_name() << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(args[1].get_shape()) << "},\n";
                    writer << "                         {"
                           << join(replace_slice->get_lower_bounds()) << "},\n";
                    writer << "                         {"
                           << join(replace_slice->get_upper_bounds()) << "},\n";
                    writer << "                         {" << join(replace_slice->get_strides())
                           << "},\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::OneHot)
            {
                auto oh = static_cast<const ngraph::op::OneHot*>(node);

                auto arg_rank = args[0].get_shape().size();

                size_t bounds = out[0].get_shape()[oh->get_one_hot_axis()];

                if (arg_rank == 0)
                {
                    writer << "{   // " << node->get_name() << " 1\n";
                    writer.indent++;

                    writer << emit_vector(out[0], "out_vector") << ";\n";

                    writer << "out_vector.setZero();\n"
                           << ""
                           << "auto pos_raw = " << emit_vector(args[0]) << "(0, 0);\n"
                           << "if (floor(pos_raw) != pos_raw)\n"
                           << "{\n";
                    writer.indent++;
                    writer
                        << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
                    writer.indent--;
                    writer << "}\n";

                    writer << "size_t pos = pos_raw;\n"
                           << "if (pos >= " << bounds << ")\n";

                    writer << "{\n";
                    writer.indent++;
                    writer << "throw(std::range_error(\"One-hot: value is out of category "
                              "range\"));\n";
                    writer.indent--;
                    writer << "}\n";

                    writer << "out_vector(pos, 0) = 1;\n";

                    writer.indent--;
                    writer << "}\n";
                }
                else if (arg_rank == 1)
                {
                    writer << "{   // " << node->get_name() << " 1\n";
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
                    writer
                        << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
                    writer.indent--;
                    writer << "}\n";

                    writer << "size_t pos = pos_raw;\n";
                    writer << "bool found = false;\n";

                    writer << "if (pos >= " << bounds << ")\n"
                           << "{\n";
                    writer.indent++;
                    writer << "throw(std::range_error(\"One-hot: value is out of category "
                              "range\"));\n";
                    writer.indent--;
                    writer << "}\n";

                    writer << "out_vector"
                           << (oh->get_one_hot_axis() == 0 ? "(pos, i)" : "(i, pos)") << " = 1;\n";

                    writer.indent--;
                    writer << "}\n";

                    writer.indent--;
                    writer << "}\n";
                }
                // Other cases are not handled yet.
                else
                {
                    writer << "kernel::one_hot<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   {" << join(args[0].get_shape()) << "},\n";
                    writer << "                   {" << join(out[0].get_shape()) << "},\n";
                    writer << "                   " << oh->get_one_hot_axis() << ");\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Ceiling)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
                size_t element_count = out[0].get_size();
#if PREFER_EIGEN == 0
                writer << "#pragma omp parallel for\n";
#endif
                writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = ceil(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Floor)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
                size_t element_count = out[0].get_size();
#if PREFER_EIGEN == 0
                writer << "#pragma omp parallel for\n";
#endif
                writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = floor(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sqrt)
            {
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
                size_t element_count = out[0].get_size();
#if PREFER_EIGEN == 0
                writer << "#pragma omp parallel for\n";
#endif
                writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
                writer << "{\n";
                writer << "    " << out[0].get_name() << "[i] = sqrt(" << args[0].get_name()
                       << "[i]);\n";
                writer << "}\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Convolution)
            {
                auto convolution = static_cast<const ngraph::op::Convolution*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto input_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto weights_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 1);
                    auto output_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_data_desc =
                        mkldnn_emitter->build_memory_descriptor(args[0], input_format);
                    auto weights_desc =
                        mkldnn_emitter->build_memory_descriptor(args[1], weights_format);
                    auto result_desc =
                        mkldnn_emitter->build_memory_descriptor(out[0], output_format);
                    size_t conv_index = 0;

                    conv_index = mkldnn_emitter->build_convolution_forward(
                        input_data_desc,
                        weights_desc,
                        result_desc,
                        convolution->get_window_movement_strides(),
                        window_dilation_strides_adjusted,
                        convolution->get_padding_below(),
                        convolution->get_padding_above());

                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(conv_index) << ");\n";
                }
                else
                {
                    writer << "kernel::convolution<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                         " << args[1].get_name() << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(arg0_shape) << "},\n";
                    writer << "                         {" << join(arg1_shape) << "},\n";
                    writer << "                         {" << join(result_shape) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_movement_strides()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_dilation_strides()) << "},\n";
                    writer << "                         {" << join(convolution->get_padding_below())
                           << "},\n";
                    writer << "                         {" << join(convolution->get_padding_above())
                           << "},\n";
                    writer << "                         {"
                           << join(convolution->get_data_dilation_strides()) << "},\n";
                    writer << "                         0, 1, 1, 0, 0, 1, false);\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropFilters)
            {
                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropFilters*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const string& elem_type =
                        runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                            args[0].get_element_type());
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto data_format = runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto delta_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 1);
                    auto result_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);

                    auto emit_memory_desc = [&writer](const std::string& var,
                                                      const std::string& shape,
                                                      const std::string& type,
                                                      const std::string& layout) {
                        writer << "memory::desc " << var << " = memory::desc({" << shape << "}, "
                               << type << ", " << layout << ");\n";
                    };

                    auto emit_memory = [&writer](
                        const std::string& var, const std::string& desc, const std::string& data) {
                        writer << "memory " << var << " = memory({" << desc << ", cpu_engine}, "
                               << data << ");\n";
                    };

                    auto emit_memory_dims = [&writer](const std::string& var,
                                                      const std::string& dims) {
                        writer << "memory::dims " << var << "{" << dims << "};\n";
                    };

                    writer.block_begin();
                    writer << "try\n";
                    writer.block_begin();
                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    emit_memory_desc(
                        "data_desc",
                        join(arg0_shape),
                        elem_type,
                        runtime::cpu::mkldnn_utils::get_mkldnn_format_string(data_format));
                    emit_memory_desc(
                        "delta_desc",
                        join(arg1_shape),
                        elem_type,
                        runtime::cpu::mkldnn_utils::get_mkldnn_format_string(delta_format));
                    emit_memory_desc(
                        "result_desc",
                        join(result_shape),
                        elem_type,
                        runtime::cpu::mkldnn_utils::get_mkldnn_format_string(result_format));
                    emit_memory("data", "data_desc", args[0].get_name());
                    emit_memory("delta", "delta_desc", args[1].get_name());
                    emit_memory("result", "result_desc", out[0].get_name());
                    emit_memory_dims("dilates", join(window_dilation_strides_adjusted));
                    emit_memory_dims("strides",
                                     join(convolution->get_window_movement_strides_forward()));
                    emit_memory_dims("padding_l", join(convolution->get_padding_below_forward()));
                    emit_memory_dims("padding_r", join(convolution->get_padding_above_forward()));

                    writer
                        << "convolution_backward_weights::desc bwd_weights_desc("
                           "algorithm::convolution_direct, "
                           "data_desc, result_desc, delta_desc, strides, dilates,"
                           "padding_l, padding_r, padding_kind::zero);\n"
                           "convolution_forward::primitive_desc fwd_pd({prop_kind::forward, "
                           "algorithm::convolution_direct, data_desc, "
                           "result_desc, delta_desc, strides, dilates, padding_l, padding_r, "
                           "padding_kind::zero}, cpu_engine);\n"
                           "convolution_backward_weights::primitive_desc "
                           "bwd_weights_pd(bwd_weights_desc, "
                           "cpu_engine, fwd_pd);\n"
                           "convolution_backward_weights bwd_weights(bwd_weights_pd, data, delta, "
                           "result);\n"
                           "stream s = stream(stream::kind::eager);\n"
                           "s.submit({bwd_weights}).wait();\n";
                    writer.block_end();
                    writer << "catch (const mkldnn::error& e)\n";
                    writer.block_begin();
                    writer << "throw ngraph::ngraph_error(\"MKLDNN ERROR (\" + std::to_string("
                              "e.status) + \"): \" + e.message);\n";
                    writer.block_end();
                    writer.block_end();
                }
                else
                {
                    writer << "kernel::convolution<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                         " << args[1].get_name() << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(arg0_shape) << "},\n";
                    writer << "                         {" << join(arg1_shape) << "},\n";
                    writer << "                         {" << join(result_shape) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_movement_strides_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_dilation_strides_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_padding_below_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_padding_above_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_data_dilation_strides_backward()) << "},\n";
                    writer << "                         1, 0, 0, 1, 1, 0, false);\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropData)
            {
                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropData*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const string& elem_type =
                        runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                            args[0].get_element_type());
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weight_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto delta_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 1);
                    auto result_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);

                    auto emit_memory_desc = [&writer](const std::string& var,
                                                      const std::string& shape,
                                                      const std::string& type,
                                                      const std::string& layout) {
                        writer << "memory::desc " << var << " = memory::desc({" << shape << "}, "
                               << type << ", " << layout << ");\n";
                    };

                    auto emit_memory = [&writer](
                        const std::string& var, const std::string& desc, const std::string& data) {
                        writer << "memory " << var << " = memory({" << desc << ", cpu_engine}, "
                               << data << ");\n";
                    };

                    auto emit_memory_dims = [&writer](const std::string& var,
                                                      const std::string& dims) {
                        writer << "memory::dims " << var << "{" << dims << "};\n";
                    };

                    writer.block_begin();
                    writer << "try\n";
                    writer.block_begin();
                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    emit_memory_desc(
                        "weight_desc",
                        join(arg0_shape),
                        elem_type,
                        runtime::cpu::mkldnn_utils::get_mkldnn_format_string(weight_format));
                    emit_memory_desc(
                        "delta_desc",
                        join(arg1_shape),
                        elem_type,
                        runtime::cpu::mkldnn_utils::get_mkldnn_format_string(delta_format));
                    emit_memory_desc(
                        "result_desc",
                        join(result_shape),
                        elem_type,
                        runtime::cpu::mkldnn_utils::get_mkldnn_format_string(result_format));
                    emit_memory("weight", "weight_desc", args[0].get_name());
                    emit_memory("delta", "delta_desc", args[1].get_name());
                    emit_memory("result", "result_desc", out[0].get_name());
                    emit_memory_dims("dilates", join(window_dilation_strides_adjusted));
                    emit_memory_dims("strides",
                                     join(convolution->get_window_movement_strides_forward()));
                    emit_memory_dims("padding_l", join(convolution->get_padding_below_forward()));
                    emit_memory_dims("padding_r", join(convolution->get_padding_above_forward()));

                    writer
                        << "convolution_backward_data::desc "
                           "bwd_data_desc(algorithm::convolution_direct, "
                           "result_desc, weight_desc, delta_desc, strides, dilates, "
                           "padding_l, padding_r, padding_kind::zero);\n"
                           "convolution_forward::primitive_desc fwd_pd({prop_kind::forward, "
                           "algorithm::convolution_direct, result_desc, weight_desc, delta_desc, "
                           "strides, dilates, padding_l, padding_r, padding_kind::zero}, "
                           "cpu_engine);\n"
                           "convolution_backward_data::primitive_desc bwd_data_pd(bwd_data_desc, "
                           "cpu_engine, fwd_pd);\n"
                           "convolution_backward_data bwd_data(bwd_data_pd, delta, weight, "
                           "result);\n"
                           "stream s = stream(stream::kind::eager);\n"
                           "s.submit({bwd_data}).wait();\n";
                    writer.block_end();
                    writer << "catch (const mkldnn::error& e)\n";
                    writer.block_begin();
                    writer << "throw ngraph::ngraph_error(\"MKLDNN ERROR (\" + std::to_string("
                              "e.status) + \"): \" + e.message);\n";
                    writer.block_end();
                    writer.block_end();
                }
                else
                {
                    // Note that args[1] and args[0] are switched here from the usual order.
                    writer << "kernel::convolution<" << out[0].get_type() << ">("
                           << args[1].get_name() << ",\n";
                    writer << "                         " << args[0].get_name() << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(arg1_shape) << "},\n";
                    writer << "                         {" << join(arg0_shape) << "},\n";
                    writer << "                         {" << join(result_shape) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_movement_strides_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_dilation_strides_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_padding_below_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_padding_above_backward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_data_dilation_strides_backward()) << "},\n";
                    writer << "                         0, 1, 0, 1, 0, 1, true);\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Not)
            {
                writer << "kernel::logical_not(" << args[0].get_name() << ",\n"
                       << "                    " << out[0].get_name() << ",\n"
                       << "                    " << out[0].get_size() << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPool)
            {
                auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);

                auto arg_shape = args[0].get_shape();
                auto arg_rank = arg_shape.size();

                auto result_shape = out[0].get_shape();

                // TODO(jmenon): Optimize for 1D

                // TODO(jmenon): Remove element type restriction
                if (arg_rank == 4 && max_pool->get_window_shape().size() == 2 &&
                    args[0].get_element_type() == element::f32)
                {
                    const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                        args[0].get_element_type());

                    writer << "{\n";
                    writer.indent++;

                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    writer << "memory::desc input_data_desc = memory::desc({" << join(arg_shape)
                           << "}, " << et << ", memory::format::nchw);\n";
                    writer << "memory::desc result_desc = memory::desc({" << join(result_shape)
                           << "}, " << et << ", memory::format::nchw);\n";

                    writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                           << args[0].get_name() << ");\n";
                    writer << "memory result = memory({result_desc, cpu_engine}, "
                           << out[0].get_name() << ");\n";

                    // TODO(jmenon): Use a workspace
                    writer << "pooling_forward max_pooling = pooling_forward({"
                           << "{prop_kind::forward_inference, algorithm::pooling_max, "
                           << "input_data_desc, result_desc, {"
                           << join(max_pool->get_window_movement_strides()) << "}, {"
                           << join(max_pool->get_window_shape()) << "}, {"
                           << join(max_pool->get_padding_below()) << "}, "
                           << "{" << join(max_pool->get_padding_above())
                           << "}, padding_kind::zero}, cpu_engine}, "
                           << "input_data, result);\n";

                    writer << "stream s = stream(stream::kind::eager);\n"
                           << "s.submit({max_pooling}).wait();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::max_pool<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                 " << out[0].get_name() << ",\n";
                    writer << "                 {" << join(arg_shape) << "},\n";
                    writer << "                 {" << join(result_shape) << "},\n";
                    writer << "                 {" << join(max_pool->get_window_shape()) << "},\n";
                    writer << "                 {" << join(max_pool->get_window_movement_strides())
                           << "},\n";
                    writer << "                 {" << join(max_pool->get_padding_below()) << "},\n";
                    writer << "                 {" << join(max_pool->get_padding_above())
                           << "});\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reverse)
            {
                auto reverse = static_cast<const ngraph::op::Reverse*>(node);

                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                writer << "kernel::reverse<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                " << out[0].get_name() << ",\n";
                writer << "                {" << join(arg_shape) << "},\n";
                writer << "                {" << join(result_shape) << "},\n";
                writer << "                {" << join(reverse->get_reversed_axes()) << "});\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReduceWindow)
            {
                auto reduce_window = static_cast<const ngraph::op::ReduceWindow*>(node);

                auto arg_reductee_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();
                auto reduction_function = reduce_window->get_functions()[0];
                auto& f_result_element_type = out[0].get_element_type();

                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;

                string type = f_result_element_type.c_type_string();
                writer << "auto f = [&](" << type << " x, " << type << " y) -> " << type << "\n{";
                writer.indent++;
                writer << "\n";
                writer << type << " result;\n";
                writer << "void* args[] = {&x, &y};\n";
                writer << "void* out[] = {&result};\n";
                writer << reduction_function->get_name() << "(args, out, ctx);\n";
                writer << "return result;\n";
                writer.indent--;
                writer << "};\n";

                writer << "kernel::reduce_window<" << out[0].get_type() << ">("
                       << args[0].get_name() << ",\n";
                writer << "                      " << args[1].get_name() << ",\n";
                writer << "                      " << out[0].get_name() << ",\n";
                writer << "                      {" << join(arg_reductee_shape) << "},\n";
                writer << "                      {" << join(result_shape) << "},\n";
                writer << "                      f,\n";
                writer << "                      {" << join(reduce_window->get_window_shape())
                       << "},\n";
                writer << "                      {"
                       << join(reduce_window->get_window_movement_strides()) << "});\n";

                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SelectAndScatter)
            {
                auto select_and_scatter = static_cast<const ngraph::op::SelectAndScatter*>(node);
                auto selection_function = select_and_scatter->get_functions()[0];
                auto scatter_function = select_and_scatter->get_functions()[1];

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;

                string type = node->get_output_element_type(0).c_type_string();

                writer << "auto f_select = [&](" << type << " x, " << type << " y) -> char\n{";
                writer.indent++;
                writer << "\n";
                writer << "char result;\n";
                writer << "void* args[] = {&x, &y};\n";
                writer << "void* out[] = {&result};\n";
                writer << selection_function->get_name() << "(args, out, ctx);\n";
                writer << "return result;\n";
                writer.indent--;
                writer << "};\n";

                writer << "auto f_scatter = [&](" << type << " x, " << type << " y) -> " << type
                       << "\n{";
                writer.indent++;
                writer << "\n";
                writer << type << " result;\n";
                writer << "void* args[] = {&x, &y};\n";
                writer << "void* out[] = {&result};\n";
                writer << scatter_function->get_name() << "(args, out, ctx);\n";
                writer << "return result;\n";
                writer.indent--;
                writer << "};\n";

                writer << "kernel::select_and_scatter<" << out[0].get_type() << ">("
                       << args[0].get_name() << ",\n";
                writer << "                " << args[1].get_name() << ",\n";
                writer << "                " << args[2].get_name() << ",\n";
                writer << "                " << out[0].get_name() << ",\n";
                writer << "                {" << join(arg0_shape) << "},\n";
                writer << "                {" << join(arg1_shape) << "},\n";
                writer << "                {" << join(result_shape) << "},\n";
                writer << "                f_select,\n";
                writer << "                f_scatter,\n";
                writer << "                {" << join(select_and_scatter->get_window_shape())
                       << "},\n";
                writer << "                {"
                       << join(select_and_scatter->get_window_movement_strides()) << "});\n";

                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AvgPool)
            {
                auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);

                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                // TODO(jmenon): Refactor into an MKLDNN Pooling emitter that handles
                // all pooling variants

                // TODO(jmenon): Optimize for 1D

                // TODO(jmenon): Remove element type restriction
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                        args[0].get_element_type());

                    const char* algorithm_enumerator =
                        avg_pool->get_include_padding_in_avg_computation()
                            ? "algorithm::pooling_avg_include_padding"
                            : "algorithm::pooling_avg_exclude_padding";

                    auto input_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto result_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);

                    writer << "{\n";
                    writer.indent++;

                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    writer << "memory::desc input_data_desc = memory::desc({" << join(arg_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(input_format)
                           << ");\n";
                    writer << "memory::desc result_desc = memory::desc({" << join(result_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(result_format)
                           << ");\n";
                    writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                           << args[0].get_name() << ");\n";
                    writer << "memory result = memory({result_desc, cpu_engine}, "
                           << out[0].get_name() << ");\n";

                    // TODO(jmenon): Use a workspace
                    writer << "pooling_forward avg_pooling = pooling_forward({"
                           << "{prop_kind::forward_inference, " << algorithm_enumerator << ", "
                           << "input_data_desc, result_desc, {"
                           << join(avg_pool->get_window_movement_strides()) << "}, {"
                           << join(avg_pool->get_window_shape()) << "}, "
                           << "{" << join(avg_pool->get_padding_below()) << "}, "
                           << "{" << join(avg_pool->get_padding_above()) << "}, "
                           << "padding_kind::zero}, cpu_engine}, "
                           << "input_data, result);\n";

                    writer << "stream s = stream(stream::kind::eager);\n"
                           << "s.submit({avg_pooling}).wait();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::avg_pool<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                 " << out[0].get_name() << ",\n";
                    writer << "                 {" << join(arg_shape) << "},\n";
                    writer << "                 {" << join(result_shape) << "},\n";
                    writer << "                 {" << join(avg_pool->get_window_shape()) << "},\n";
                    writer << "                 {" << join(avg_pool->get_window_movement_strides())
                           << "},\n";
                    writer << "                 {" << join(avg_pool->get_padding_below()) << "},\n";
                    writer << "                 {" << join(avg_pool->get_padding_above()) << "},\n";
                    writer << "                 "
                           << ngraph::to_cplusplus_sourcecode_literal(
                                  avg_pool->get_include_padding_in_avg_computation())
                           << "\n";
                    writer << "                  );\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Pad)
            {
                auto pad = static_cast<const ngraph::op::Pad*>(node);

                auto arg0_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                writer << "kernel::pad<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "            " << args[1].get_name() << ",\n";
                writer << "            " << out[0].get_name() << ",\n";
                writer << "            {" << join(arg0_shape) << "},\n";
                writer << "            {" << join(result_shape) << "},\n";
                writer << "            {" << join(pad->get_padding_below()) << "},\n";
                writer << "            {" << join(pad->get_padding_above()) << "},\n";
                writer << "            {" << join(pad->get_padding_interior()) << "});\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AvgPoolBackprop)
            {
                auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);

                auto delta_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                        args[0].get_element_type());

                    auto input_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto result_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);

                    writer << "{\n";
                    writer.indent++;

                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    writer << "memory::desc input_data_desc = memory::desc({" << join(delta_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(input_format)
                           << ");\n";
                    writer << "memory::desc result_desc = memory::desc({" << join(out_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(result_format)
                           << ");\n";
                    writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                           << args[0].get_name() << ");\n";
                    writer << "memory result = memory({result_desc, cpu_engine}, "
                           << out[0].get_name() << ");\n";
                    // Dummy forward primitive descriptor to keep MKLDNN happy
                    const char* algorithm_enumerator =
                        apb->get_include_padding_in_avg_computation()
                            ? "algorithm::pooling_avg_include_padding"
                            : "algorithm::pooling_avg_exclude_padding";

                    writer << "pooling_forward::primitive_desc fwd_pd = "
                              "pooling_forward::primitive_desc("
                           << "{prop_kind::forward, " << algorithm_enumerator << ", "
                           << "result_desc, input_data_desc, {"
                           << join(apb->get_window_movement_strides()) << "}, {"
                           << join(apb->get_window_shape()) << "}, "
                           << "{" << join(apb->get_padding_below()) << "}, "
                           << "{" << join(apb->get_padding_above()) << "}, "
                           << "padding_kind::zero}, cpu_engine);\n";
                    writer
                        << "auto avg_pooling = pooling_backward(pooling_backward::primitive_desc("
                        << "pooling_backward::desc(" << algorithm_enumerator << ", "
                        << "result_desc, input_data_desc, {"
                        << join(apb->get_window_movement_strides()) << "}, {"
                        << join(apb->get_window_shape()) << "}, "
                        << "{" << join(apb->get_padding_below()) << "}, "
                        << "{" << join(apb->get_padding_above()) << "}, "
                        << "padding_kind::zero), cpu_engine, fwd_pd), "
                        << "input_data, result);\n";
                    writer << "auto s = stream(stream::kind::eager);\n"
                           << "s.submit({avg_pooling}).wait();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::avg_pool_backprop<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                 " << out[0].get_name() << ",\n";
                    writer << "                 {" << join(delta_shape) << "},\n";
                    writer << "                 {" << join(out_shape) << "},\n";
                    writer << "                 {" << join(apb->get_window_shape()) << "},\n";
                    writer << "                 {" << join(apb->get_window_movement_strides())
                           << "},\n";
                    writer << "                 {" << join(apb->get_padding_below()) << "},\n";
                    writer << "                 {" << join(apb->get_padding_above()) << "},\n";
                    writer << "                 "
                           << ngraph::to_cplusplus_sourcecode_literal(
                                  apb->get_include_padding_in_avg_computation())
                           << "\n";
                    writer << "                 );\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolBackprop)
            {
                auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);

                auto delta_shape = args[1].get_shape();
                auto delta_rank = delta_shape.size();
                auto out_shape = out[0].get_shape();

                if (delta_rank == 4 && mpb->get_window_shape().size() == 2 &&
                    args[0].get_element_type() == element::f32)
                {
                    const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                        args[1].get_element_type());

                    writer << "{\n";
                    writer.indent++;
                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    writer << "memory::desc input_data_desc = memory::desc({" << join(delta_shape)
                           << "}, " << et << ", memory::format::nchw);\n";
                    writer << "memory::desc result_desc = memory::desc({" << join(out_shape)
                           << "}, " << et << ", memory::format::nchw);\n";
                    writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                           << args[1].get_name() << ");\n";
                    writer << "memory result = memory({result_desc, cpu_engine}, "
                           << out[0].get_name() << ");\n";

                    //----------------------------------------------------------------------------------------------
                    // create a forward primitive_desc, use this to query the workspace
                    // TODO: (pruthvi) this is a workaround, till we maintain a global context to refer to the corrosponding
                    //        MKLDNN fprop kernel. this impacts performance
                    writer << "memory::desc max_pool_input_desc = memory::desc({"
                           << join(args[0].get_shape()) << "}, " << et
                           << ", memory::format::nchw);\n";
                    writer << "memory::desc max_pool_result_desc = memory::desc({"
                           << join(args[1].get_shape()) << "}, " << et
                           << ", memory::format::nchw);\n";
                    writer
                        << "memory maxpool_input_data = memory({max_pool_input_desc, cpu_engine}, "
                        << args[0].get_name() << ");\n";
                    writer << "memory maxpool_result = memory({max_pool_result_desc, cpu_engine}, "
                           << out[0].get_name() << ");\n";
                    writer << "pooling_forward::primitive_desc pool_fwd_pd = "
                              "pooling_forward::primitive_desc("
                           << "{prop_kind::forward, algorithm::pooling_max, "
                           << "max_pool_input_desc, max_pool_result_desc, {"
                           << join(mpb->get_window_movement_strides()) << "}, {"
                           << join(mpb->get_window_shape()) << "}, "
                           << "{" << join(mpb->get_padding_below()) << "}, "
                           << "{" << join(mpb->get_padding_above()) << "}, "
                           << "padding_kind::zero}, cpu_engine);\n";

                    // query the workspace from the forward primitive desc and allocates memory
                    writer << "auto max_pool_workspace_memory = "
                              "memory(pool_fwd_pd.workspace_primitive_desc());\n";
                    //run fprop with this workspace attached
                    writer << "pooling_forward max_pooling_fwd = pooling_forward("
                           << "pool_fwd_pd, maxpool_input_data, maxpool_result, "
                              "max_pool_workspace_memory);\n";

                    writer << "stream s_fprop = stream(stream::kind::eager);\n"
                           << "s_fprop.submit({max_pooling_fwd}).wait();\n";

                    //---------------------------------------------------------------------------------------------
                    writer << "auto max_pooling_bwd = "
                              "pooling_backward(pooling_backward::primitive_desc("
                           << "pooling_backward::desc(algorithm::pooling_max, "
                           << "result_desc, input_data_desc, {"
                           << join(mpb->get_window_movement_strides()) << "}, {"
                           << join(mpb->get_window_shape()) << "}, "
                           << "{" << join(mpb->get_padding_below()) << "}, "
                           << "{" << join(mpb->get_padding_above()) << "}, "
                           << "padding_kind::zero), cpu_engine, pool_fwd_pd), "
                           << "input_data, max_pool_workspace_memory, result);\n";
                    writer << "auto s_bwd = stream(stream::kind::eager);\n"
                           << "s_bwd.submit({max_pooling_bwd}).wait();\n";

                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::max_pool_backprop<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                 " << args[1].get_name() << ",\n";
                    writer << "                 " << out[0].get_name() << ",\n";
                    writer << "                 {" << join(delta_shape) << "},\n";
                    writer << "                 {" << join(out_shape) << "},\n";
                    writer << "                 {" << join(mpb->get_window_shape()) << "},\n";
                    writer << "                 {" << join(mpb->get_window_movement_strides())
                           << "},\n";
                    writer << "                 {" << join(mpb->get_padding_below()) << "},\n";
                    writer << "                 {" << join(mpb->get_padding_above()) << "}\n";
                    writer << "                 );\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Product)
            {
                const ngraph::op::Product* product = static_cast<const ngraph::op::Product*>(node);
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                const Shape& arg_shape = args[0].get_shape();
                size_t arg_rank = arg_shape.size();
                const AxisSet& reduction_axes = product->get_reduction_axes();

                // Trivial case: no reduction axes.
                if (reduction_axes.size() == 0)
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
                           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                // Full reduction? Then reduce to scalar.
                else if ((arg_rank == 1 && reduction_axes == AxisSet{0}) ||
                         (arg_rank == 2 && reduction_axes == AxisSet{0, 1}))
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_array1d(out[0]) << " =\n"
                           << "    " << emit_array1d(args[0]) << ".prod();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (arg_rank == 2 && reduction_axes == AxisSet{1})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".rowwise().prod();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (arg_rank == 2 && reduction_axes == AxisSet{0})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".colwise().prod();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::product<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(args[0].get_shape()) << "},\n";
                    writer << "                         {" << join(out[0].get_shape()) << "},\n";
                    writer << "                         {" << join(product->get_reduction_axes())
                           << "});\n";
                }
#else
                // TODO: add an emitter akin to the emit_sum
                writer << "kernel::product<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                         " << out[0].get_name() << ",\n";
                writer << "                         {" << join(args[0].get_shape()) << "},\n";
                writer << "                         {" << join(out[0].get_shape()) << "},\n";
                writer << "                         {" << join(product->get_reduction_axes())
                       << "});\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Max)
            {
                const ngraph::op::Max* max = static_cast<const ngraph::op::Max*>(node);
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                const Shape& arg_shape = args[0].get_shape();
                size_t arg_rank = arg_shape.size();
                const AxisSet& reduction_axes = max->get_reduction_axes();

                bool zero_sized = false;
                for (size_t s : arg_shape)
                {
                    zero_sized |= (s == 0);
                }

                // Trivial case: no reduction axes.
                if (!zero_sized && reduction_axes.size() == 0)
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
                           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                // Full reduction? Then reduce to scalar.
                else if (!zero_sized && ((arg_rank == 1 && reduction_axes == AxisSet{0}) ||
                                         (arg_rank == 2 && reduction_axes == AxisSet{0, 1})))
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_array1d(out[0]) << " =\n"
                           << "    " << emit_array1d(args[0]) << ".maxCoeff();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (!zero_sized && arg_rank == 2 && reduction_axes == AxisSet{1})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".rowwise().maxCoeff();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (!zero_sized && arg_rank == 2 && reduction_axes == AxisSet{0})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".colwise().maxCoeff();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::max<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(args[0].get_shape()) << "},\n";
                    writer << "                         {" << join(out[0].get_shape()) << "},\n";
                    writer << "                         {" << join(max->get_reduction_axes())
                           << "});\n";
                }
#else
                // TODO: add an emitter akin to the emit_sum
                writer << "kernel::max<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                         " << out[0].get_name() << ",\n";
                writer << "                         {" << join(args[0].get_shape()) << "},\n";
                writer << "                         {" << join(out[0].get_shape()) << "},\n";
                writer << "                         {" << join(max->get_reduction_axes())
                       << "});\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Min)
            {
                const ngraph::op::Min* min = static_cast<const ngraph::op::Min*>(node);
                writer << "{   // " << node->get_name() << "\n";
                writer.indent++;
#if PREFER_EIGEN == 1
                const Shape& arg_shape = args[0].get_shape();
                size_t arg_rank = arg_shape.size();
                const AxisSet& reduction_axes = min->get_reduction_axes();

                bool zero_sized = false;
                for (size_t s : arg_shape)
                {
                    zero_sized |= (s == 0);
                }

                // Trivial case: no reduction axes.
                if (!zero_sized && reduction_axes.size() == 0)
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
                           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                    writer.indent--;
                    writer << "}\n";
                }
                // Full reduction? Then reduce to scalar.
                else if (!zero_sized && ((arg_rank == 1 && reduction_axes == AxisSet{0}) ||
                                         (arg_rank == 2 && reduction_axes == AxisSet{0, 1})))
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_array1d(out[0]) << " =\n"
                           << "    " << emit_array1d(args[0]) << ".minCoeff();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (!zero_sized && arg_rank == 2 && reduction_axes == AxisSet{1})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".rowwise().minCoeff();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else if (!zero_sized && arg_rank == 2 && reduction_axes == AxisSet{0})
                {
                    writer << "{   // " << node->get_name() << "\n";
                    writer.indent++;
                    writer << emit_vector(out[0]) << " =\n"
                           << "    " << emit_matrix(args[0]) << ".colwise().minCoeff();\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::min<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(args[0].get_shape()) << "},\n";
                    writer << "                         {" << join(out[0].get_shape()) << "},\n";
                    writer << "                         {" << join(min->get_reduction_axes())
                           << "});\n";
                }
#else
                // TODO: add an emitter akin to the emit_sum
                writer << "kernel::min<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                         " << out[0].get_name() << ",\n";
                writer << "                         {" << join(args[0].get_shape()) << "},\n";
                writer << "                         {" << join(out[0].get_shape()) << "},\n";
                writer << "                         {" << join(min->get_reduction_axes())
                       << "});\n";
#endif
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::runtime::cpu::op::ConvertLayout)
            {
                auto input_tvl =
                    node->get_inputs()[0].get_output().get_tensor_view()->get_tensor_view_layout();
                auto output_tvl = node->get_output_tensor_view(0)->get_tensor_view_layout();
                auto input_format =
                    dynamic_cast<runtime::cpu::LayoutDescriptor&>(*input_tvl).get_mkldnn_format();
                auto output_format =
                    dynamic_cast<runtime::cpu::LayoutDescriptor&>(*output_tvl).get_mkldnn_format();
                const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                    args[0].get_element_type());

                writer << "{\n";
                writer.indent++;

                writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                writer << "memory::desc input_desc = memory::desc({" << join(args[0].get_shape())
                       << "}, " << et << ", "
                       << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(input_format)
                       << ");\n";
                writer << "memory::desc output_desc = memory::desc({" << join(out[0].get_shape())
                       << "}, " << et << ", "
                       << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(output_format)
                       << ");\n";
                writer << "memory input = memory({input_desc, cpu_engine}, " << args[0].get_name()
                       << ");\n";
                writer << "memory output = memory({output_desc, cpu_engine}, " << out[0].get_name()
                       << ");\n";
                writer << "reorder prim = reorder(input, output);\n";

                writer << "stream s = stream(stream::kind::eager);\n"
                       << "s.submit({prim}).wait();\n";
                writer.indent--;
                writer << "}\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReluBackprop)
            {
                const auto& arg_shape = args[0].get_shape();
                const auto& result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                        args[0].get_element_type());

                    auto input_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto delta_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 1);
                    if (!runtime::cpu::mkldnn_utils::compare_mkldnn_formats(input_format,
                                                                            delta_format))
                    {
                        throw ngraph_error(
                            "mkldnn emitter: Relu backprop fprop input and delta layouts should be "
                            "the same");
                    }
                    auto result_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);

                    writer << "{\n";
                    writer.indent++;

                    writer << "try {\n";
                    writer.indent++;
                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    writer << "memory::desc input_data_desc = memory::desc({" << join(arg_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(input_format)
                           << ");\n";
                    writer << "memory::desc delta_data_desc = memory::desc({"
                           << join(args[1].get_shape()) << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(delta_format)
                           << ");\n";
                    writer << "memory::desc result_desc = memory::desc({" << join(result_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(result_format)
                           << ");\n";

                    writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                           << args[0].get_name() << ");\n";
                    writer << "memory delta_data = memory({delta_data_desc, cpu_engine}, "
                           << args[1].get_name() << ");\n";
                    writer << "memory result = memory({result_desc, cpu_engine}, "
                           << out[0].get_name() << ");\n";
                    writer << "relu_forward::desc relu_fwd_desc = "
                              "relu_forward::desc(prop_kind::forward, "
                              "algorithm::eltwise_relu, input_data_desc, 0, 0);\n";
                    writer << "relu_forward::primitive_desc relu_fwd_prim_desc = "
                              "relu_forward::primitive_desc(relu_fwd_desc, cpu_engine);\n";
                    writer << "relu_backward::desc relu_bwd_desc = "
                              "relu_backward::desc(algorithm::eltwise_relu, "
                              "delta_data_desc, input_data_desc, 0, 0);\n";
                    writer << "relu_backward::primitive_desc relu_bdw_prim_desc = "
                              "relu_backward::primitive_desc(relu_bwd_desc, cpu_engine, "
                              "relu_fwd_prim_desc);\n";
                    writer
                        << "relu_backward relu_bwd= relu_backward(relu_bdw_prim_desc, input_data, "
                           "delta_data, result);\n";
                    writer << "stream s = stream(stream::kind::eager);\n"
                              "s.submit({relu_bwd}).wait();\n";
                    writer.indent--;
                    writer << "} catch (const mkldnn::error& e) {\n";
                    writer.indent++;
                    writer << "throw ngraph::ngraph_error(\"MKLDNN ERROR (\" + std::to_string("
                              "e.status) + \"): \" + e.message);\n";
                    writer.indent--;
                    writer << "}\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::relu_backprop<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                      " << args[1].get_name() << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   " << out[0].get_size() << ");\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Relu)
            {
                const auto& arg_shape = args[0].get_shape();
                const auto& result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const string& et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(
                        args[0].get_element_type());

                    auto input_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto result_format =
                        runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0);

                    writer << "{\n";
                    writer.indent++;

                    writer << "try {\n";
                    writer.indent++;
                    writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    writer << "memory::desc input_data_desc = memory::desc({" << join(arg_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(input_format)
                           << ");\n";
                    writer << "memory::desc result_desc = memory::desc({" << join(result_shape)
                           << "}, " << et << ", "
                           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(result_format)
                           << ");\n";

                    writer << "memory input_data = memory({input_data_desc, cpu_engine}, "
                           << args[0].get_name() << ");\n";
                    writer << "memory result = memory({result_desc, cpu_engine}, "
                           << out[0].get_name() << ");\n";
                    writer << "relu_forward::desc relu_fwd_desc = "
                              "relu_forward::desc(prop_kind::forward_training, "
                              "algorithm::eltwise_relu, input_data_desc, 0, 0);\n";
                    writer << "relu_forward::primitive_desc relu_prim_desc = "
                              "relu_forward::primitive_desc(relu_fwd_desc, cpu_engine);\n";
                    writer << "relu_forward relu_fwd= relu_forward(relu_prim_desc, input_data, "
                              "result);\n";
                    writer << "stream s = stream(stream::kind::eager);\n"
                              "s.submit({relu_fwd}).wait();\n";
                    writer.indent--;
                    writer << "} catch (const mkldnn::error& e) {\n";
                    writer.indent++;
                    writer << "throw ngraph::ngraph_error(\"MKLDNN ERROR (\" + std::to_string("
                              "e.status) + \"): \" + e.message);\n";
                    writer.indent--;
                    writer << "}\n";
                    writer.indent--;
                    writer << "}\n";
                }
                else
                {
                    writer << "kernel::relu<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   " << out[0].get_size() << ");\n";
                }
            }
        }
    }
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
