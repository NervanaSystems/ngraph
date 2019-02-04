//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_emitters.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_dot.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/loop_kernel.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include <mlsl.hpp>

#include "ngraph/op/allreduce.hpp"
#endif

using namespace std;
using namespace ngraph;

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

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Add)
            {
                writer.block_begin();
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    std::vector<float> scale_vector(2, 1);
                    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input0_data_desc, runtime::cpu::executor::global_cpu_engine));
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input1_data_desc, runtime::cpu::executor::global_cpu_engine));

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
                    writer.block_begin();
                    writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] + "
                           << args[1].get_name() << "[i];\n";
                    writer.block_end();
                }
                writer.block_end();
            }

#ifdef NGRAPH_DISTRIBUTED
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AllReduce)
            {
                const element::Type& element_type = args[0].get_element_type();
                auto data_type = "MLSL::DT_FLOAT";

                if (element_type == element::f32)
                {
                    data_type = "MLSL::DT_FLOAT";
                }
                else if (element_type == element::f64)
                {
                    data_type = "MLSL::DT_DOUBLE";
                }

                writer.block_begin();
                writer << "MLSL::CommReq* req = ctx->mlsl_dist->AllReduce(" << args[0].get_name()
                       << ", " << out[0].get_name() << ", " << out[0].get_size() << ", "
                       << data_type << ", MLSL::RT_SUM, MLSL::GT_DATA);\n";
                writer << "ctx->mlsl_env->Wait(req);\n";
                writer.block_end();
            }
#endif

            static void emitCblasSgemmBatch(codegen::CodeWriter& writer,
                                            const Shape& shape_a,
                                            const Shape& shape_b,
                                            const Shape& shape_c,
                                            bool transpose_a,
                                            bool transpose_b,
                                            const std::string& data_a,
                                            const std::string& data_b,
                                            const std::string& data_c,
                                            const std::string& alpha,
                                            const std::string& beta,
                                            size_t group_size)
            {
                static const char* cblas_transpose = "cblas::Transpose::Transpose";
                static const char* cblas_no_transpose = "cblas::Transpose::None";

                size_t m = shape_a[1];
                size_t k = shape_a[2];
                size_t n = shape_b[2];
                size_t lda = std::max(1UL, k);
                size_t ldb = std::max(1UL, n);
                const char* ctranspose_a = cblas_no_transpose;
                const char* ctranspose_b = cblas_no_transpose;
                if (transpose_a)
                {
                    ctranspose_a = cblas_transpose;
                    m = shape_a[2];
                    k = shape_a[1];
                    lda = std::max(1UL, m);
                }
                if (transpose_b)
                {
                    ctranspose_b = cblas_transpose;
                    n = shape_b[1];
                    ldb = std::max(1UL, k);
                }
                size_t ldc = std::max(1UL, n);

                const size_t offset_a = (shape_a.at(0) > 1) ? m * k : 0;
                const size_t offset_b = (shape_b.at(0) > 1) ? k * n : 0;
                const size_t offset_c = (shape_c.at(0) > 1) ? m * n : 0;

                writer.block_begin();

                const size_t group_count = 1;
                auto populate_array =
                    [&writer](const std::string& var, size_t size, size_t offset) {
                        for (size_t i = 0; i < size; ++i)
                        {
                            writer << var << "+" << i * offset << ((i < size - 1) ? ", " : "");
                        }
                    };
                writer << "cblas::Transpose transa_array[] = {" << ctranspose_a << "};\n";
                writer << "cblas::Transpose transb_array[] = {" << ctranspose_b << "};\n";
                writer << "int64_t m_array[] = {" << m << "};\n";
                writer << "int64_t n_array[] = {" << n << "};\n";
                writer << "int64_t k_array[] = {" << k << "};\n";
                writer << "std::vector<const float*> a{";
                populate_array(data_a, group_size, offset_a);
                writer << "};\n";
                writer << "const float** a_array = &a[0];\n";
                writer << "int64_t lda_array[] = {" << lda << "};\n";
                writer << "std::vector<const float*> b{";
                populate_array(data_b, group_size, offset_b);
                writer << "};\n";
                writer << "const float** b_array = &b[0];\n";
                writer << "int64_t ldb_array[] = {" << ldb << "};\n";
                writer << "std::vector<float*> c{";
                populate_array(data_c, group_size, offset_c);
                writer << "};\n";
                writer << "float** c_array = &c[0];\n";
                writer << "int64_t ldc_array[] = {" << ldc << "};\n";
                writer << "int64_t group_size[] = {" << group_size << "};\n";

                writer << "cblas_sgemm_batch(cblas::Layout::RowMajor, ";
                writer << "transa_array, transb_array, m_array, n_array, k_array, \n";
                writer << alpha << ", a_array, lda_array, b_array, ldb_array, " << beta << ", \n";
                writer << "c_array, ldc_array, " << group_count << ", group_size);\n";
                writer.block_end();
            }

            template <typename T>
            static void emitBatchDot(const ngraph::Node* node,
                                     const Shape& shape_a,
                                     const Shape& shape_b,
                                     const Shape& shape_c,
                                     const std::vector<TensorViewWrapper>& args,
                                     const std::vector<TensorViewWrapper>& out,
                                     codegen::CodeWriter& writer)
            {
                writer.block_begin();

                const T* batch_dot = static_cast<const T*>(node);

                auto mat_a = args[0];
                auto mat_b = args[1];
                auto mat_c = out[0];

                writer << "float alpha_array[] = {1.0f};\n";
                writer << "float beta_array[] = {0.0f};\n";

                const size_t group_size = shape_a[0];
                emitCblasSgemmBatch(writer,
                                    shape_a,
                                    shape_b,
                                    shape_c,
                                    batch_dot->get_is_a_transposed(),
                                    batch_dot->get_is_b_transposed(),
                                    mat_a.get_name(),
                                    mat_b.get_name(),
                                    mat_c.get_name(),
                                    "alpha_array",
                                    "beta_array",
                                    group_size);

                writer.block_end();
            }

            static Shape pad_with(Shape v, size_t val, size_t length)
            {
                if (length <= v.size())
                {
                    return v;
                }

                Shape tv(length - v.size(), val);
                v.insert(v.begin(), tv.begin(), tv.end());
                return v;
            }

            static std::string emit_constant_array(const std::string& type,
                                                   const std::string& name,
                                                   const string& val,
                                                   size_t size)
            {
                std::stringstream writer;
                writer << "static " << type << " " << name << "[" << size << "]"
                       << " = { " << val;
                for (size_t i = 1; i < size; ++i)
                {
                    writer << ", " << val;
                }
                writer << "};\n";
                return writer.str();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MatmulBias)
            {
                const ngraph::op::MatmulBias* cg = static_cast<const ngraph::op::MatmulBias*>(node);

                const Shape& arg0_shape = pad_with(cg->get_a_shape(), 1, 3); // A
                const Shape& arg1_shape = pad_with(cg->get_b_shape(), 1, 3); // B
                const Shape& arg2_shape = node->get_shape();                 // bias (C)
                const Shape& padded_result_shape = pad_with(node->get_shape(), 1, 3);
                // Step 1: dot(A,B)
                emitBatchDot<ngraph::op::MatmulBias>(
                    node, arg0_shape, arg1_shape, padded_result_shape, args, out, writer);

                // Step 2: add bias
                if (args.size() < 3)
                {
                    // no bias
                    return;
                }
                auto mat_c = args[2];

                // the bias argument of add(dot(A,B), broadcast(C)) is typically C
                // In order to broadcast C to the same shape as dot(A,B)
                // we use cblas_gemm_batch(ones, C) or cblas_gemm_batch(C, ones)
                // where ones is a tensor of appropriate shape
                // consisting of the identity element

                // Consider an example of broadcasing a tensor of Shape{1,3}
                // to Shape {4,3}
                //
                // [1    [1 2 3]  [1 2 3
                //  1             1 2 3
                //  1   *         1 2 3
                //  1]            1 2 3]

                // The next example is broadcasting a tensor of Shape{3,1} to Shape {3,4}
                //
                // [1  [1 1 1 1]  [1 1 1 1
                // 2  *           2 2 2 2
                // 3]             3 3 3 3]

                writer << "float alpha_beta_array[] = {1.0f};\n";

                const size_t group_size = 1;
                auto axes = cg->get_broadcast_axes();
                if (axes.size() == 1)
                {
                    auto second_broadcast_axis = *axes.begin();

                    if (second_broadcast_axis == 0)
                    {
                        writer << emit_constant_array(out[0].get_element_type().c_type_string(),
                                                      "ones",
                                                      "1.0f",
                                                      arg2_shape.at(0));
                        ;
                        emitCblasSgemmBatch(writer,
                                            Shape{1, arg2_shape.at(0), 1}, // ones shape
                                            Shape{1, 1, arg2_shape.at(1)}, // C shape
                                            node->get_shape(),
                                            false,
                                            false,
                                            "ones",            // ones
                                            mat_c.get_name(),  // C
                                            out[0].get_name(), // dot(A,B)
                                            "alpha_beta_array",
                                            "alpha_beta_array",
                                            group_size);
                    }
                    else
                    {
                        writer << emit_constant_array(out[0].get_element_type().c_type_string(),
                                                      "ones",
                                                      "1.0f",
                                                      arg2_shape.at(1));
                        emitCblasSgemmBatch(writer,
                                            Shape{1, arg2_shape.at(0), 1}, // C shape
                                            Shape{1, 1, arg2_shape.at(1)}, // ones shape
                                            node->get_shape(),
                                            false, // C transpose
                                            false, // C shape
                                            mat_c.get_name(),
                                            "ones",
                                            out[0].get_name(), // dot(A,B)
                                            "alpha_beta_array",
                                            "alpha_beta_array",
                                            group_size);
                    }
                }
                else
                {
                    if (axes.size() != 2)
                    {
                        throw ngraph_error("unexpected broadcast rank");
                    }

                    writer << emit_constant_array(out[0].get_element_type().c_type_string(),
                                                  "ones",
                                                  "1.0f",
                                                  arg2_shape.at(1));
                    auto bias_scalar = args[2].get_name() + "[0]";
                    writer << emit_constant_array(out[0].get_element_type().c_type_string(),
                                                  "bias_vector",
                                                  bias_scalar,
                                                  arg2_shape.at(0));

                    emitCblasSgemmBatch(writer,
                                        Shape{1, arg2_shape.at(0), 1}, // bias_vector shape
                                        Shape{1, 1, arg2_shape.at(1)}, // ones shape
                                        node->get_shape(),
                                        false, // bias_vector tranpose
                                        false, // ones tranpose
                                        "bias_vector",
                                        "ones",
                                        out[0].get_name(), // dot(A,B)
                                        "alpha_beta_array",
                                        "alpha_beta_array",
                                        group_size);
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchDot)
            {
                const auto* cg = static_cast<const ngraph::op::BatchDot*>(node);
                emitBatchDot<ngraph::op::BatchDot>(node,
                                                   cg->get_a_shape(),
                                                   cg->get_b_shape(),
                                                   out[0].get_shape(),
                                                   args,
                                                   out,
                                                   writer);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Lstm)
            {
                if (args.size() != 5)
                {
                    throw ngraph_error(
                        "Lstm op doesnt have the required number of inputs to emit MKLDNN kernel");
                }
                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto lstm_index = mkldnn_emitter->build_rnn<ngraph::op::Lstm>(node, args, out);
                auto& deps = mkldnn_emitter->get_primitive_deps(lstm_index);

                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0]) << ", "
                       << args[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1]) << ", "
                       << args[1].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2]) << ", "
                       << args[2].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3]) << ", "
                       << args[3].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[4]) << ", "
                       << args[4].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[5]) << ", "
                       << out[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[6]) << ", "
                       << out[1].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[7])
                       << ", ctx->mkldnn_workspaces[" << deps[8] << "]);\n";

                writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                       << to_string(lstm_index) << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Rnn)
            {
                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto rnn_index = mkldnn_emitter->build_rnn<ngraph::op::Rnn>(node, args, out);
                auto& deps = mkldnn_emitter->get_primitive_deps(rnn_index);

                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0]) << ", "
                       << args[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1]) << ", "
                       << args[1].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2]) << ", "
                       << args[2].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3]) << ", "
                       << args[3].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[4]) << ", "
                       << args[4].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[5]) << ", "
                       << out[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[6]) << ", "
                       << out[1].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[7])
                       << ", ctx->mkldnn_workspaces[" << deps[8] << "]);\n";
                writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, " << to_string(rnn_index)
                       << ");\n";
            }

            template <typename T>
            void CPU_Emitter::emitBatchNorm(CPU_ExternalFunction* external_function,
                                            codegen::CodeWriter& writer,
                                            const ngraph::Node* node,
                                            const std::vector<TensorViewWrapper>& args,
                                            const std::vector<TensorViewWrapper>& out,
                                            bool append_relu,
                                            bool training)
            {
                const T* batchnorm = static_cast<const T*>(node);

                writer.block_begin();
                // define weights
                writer << "std::vector<" << args[0].get_element_type().c_type_string()
                       << ">bn_weights(2*" << args[0].get_size() << ");\n";
                writer << "memcpy(&bn_weights[0], " << args[0].get_name() << ", "
                       << args[0].get_size() * args[0].get_element_type().size() << ");\n";
                writer << "memcpy(&bn_weights[0]+" << args[0].get_size() << ", "
                       << args[1].get_name() << ", "
                       << args[1].get_size() * args[1].get_element_type().size() << ");\n";

                const float ops_scale = 1.f;
                const float ops_alpha = -0.f; // relu negative slope
                const float ops_beta = 0.f;

                mkldnn::post_ops ops;
                if (append_relu)
                {
                    ops.append_eltwise(
                        ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                }

                if (training && args.size() == 3)
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto results_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mean_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);
                    auto variance_desc = mkldnn_utils::get_output_mkldnn_md(node, 2);

                    auto batchnorm_index =
                        mkldnn_emitter->build_batchnorm_forward(input_desc,
                                                                weights_desc,
                                                                results_desc,
                                                                mean_desc,
                                                                variance_desc,
                                                                batchnorm->get_eps_value(),
                                                                false,
                                                                training,
                                                                ops);

                    auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", bn_weights.data());\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[4])
                           << ", " << out[2].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(batchnorm_index) << ");\n";
                }
                else
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto mean_desc = mkldnn_utils::get_input_mkldnn_md(node, 3);
                    auto variance_desc = mkldnn_utils::get_input_mkldnn_md(node, 4);
                    auto results_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto batchnorm_index =
                        mkldnn_emitter->build_batchnorm_forward(input_desc,
                                                                weights_desc,
                                                                results_desc,
                                                                mean_desc,
                                                                variance_desc,
                                                                batchnorm->get_eps_value(),
                                                                true,
                                                                training,
                                                                ops);

                    auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[3].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << args[4].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", bn_weights.data());\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[4])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(batchnorm_index) << ");\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormTraining)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const ngraph::op::BatchNormTraining* batchnorm =
                        static_cast<const ngraph::op::BatchNormTraining*>(node);

                    if (args.size() == 3)
                    {
                        writer << "reference::batch_norm_training(" << batchnorm->get_eps_value()
                               << ",\n";
                        writer << "            " << args[0].get_name() << ",\n";
                        writer << "            " << args[1].get_name() << ",\n";
                        writer << "            " << args[2].get_name() << ",\n";
                        writer << "            " << out[0].get_name() << ",\n";
                        writer << "            " << out[1].get_name() << ",\n";
                        writer << "            " << out[2].get_name() << ",\n";
                        writer << "            {" << join(args[2].get_shape()) << "});\n";
                    }
                    else
                    {
                        writer << "reference::batch_norm_inference(" << batchnorm->get_eps_value()
                               << ",\n";
                        writer << "            " << args[0].get_name() << ",\n";
                        writer << "            " << args[1].get_name() << ",\n";
                        writer << "            " << args[2].get_name() << ",\n";
                        writer << "            " << args[3].get_name() << ",\n";
                        writer << "            " << args[4].get_name() << ",\n";
                        writer << "            " << out[0].get_name() << ",\n";
                        writer << "            {" << join(args[2].get_shape()) << "});\n";
                    }
                }
                else
                {
                    emitBatchNorm<ngraph::op::BatchNormTraining>(
                        external_function, writer, node, args, out, false, true);
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormInference)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const ngraph::op::BatchNormInference* batchnorm =
                        static_cast<const ngraph::op::BatchNormInference*>(node);

                    writer << "reference::batch_norm_inference(" << batchnorm->get_eps_value()
                           << ",\n";
                    writer << "            " << args[0].get_name() << ",\n";
                    writer << "            " << args[1].get_name() << ",\n";
                    writer << "            " << args[2].get_name() << ",\n";
                    writer << "            " << args[3].get_name() << ",\n";
                    writer << "            " << args[4].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(args[2].get_shape()) << "});\n";
                }
                else
                {
                    emitBatchNorm<ngraph::op::BatchNormInference>(
                        external_function, writer, node, args, out, false, false);
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormTrainingRelu)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error("BatchNormRelu is only supported with 4-D MKLDNN kernel.");
                }
                emitBatchNorm<ngraph::op::BatchNormTrainingRelu>(
                    external_function, writer, node, args, out, true, true);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormInferenceRelu)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error("BatchNormRelu is only supported with 4-D MKLDNN kernel.");
                }
                emitBatchNorm<ngraph::op::BatchNormTrainingRelu>(
                    external_function, writer, node, args, out, true, false);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormTrainingBackprop)
            {
                const ngraph::op::BatchNormTrainingBackprop* batchnorm =
                    static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);

                writer.block_begin();
                // define weights
                writer << "std::vector<" << args[0].get_element_type().c_type_string()
                       << ">bn_weights(2*" << args[0].get_size() << ");\n";
                writer << "std::vector<" << args[0].get_element_type().c_type_string()
                       << ">bn_dweights(2*" << args[0].get_size() << ");\n";

                writer << "memcpy(&bn_weights[0], " << args[0].get_name() << ", "
                       << args[0].get_size() * args[0].get_element_type().size() << ");\n";
                writer << "memcpy(&bn_weights[0]+" << args[0].get_size() << ", "
                       << args[1].get_name() << ", "
                       << args[1].get_size() * args[1].get_element_type().size() << ");\n";

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto weights_shape = Shape{2, args[0].get_size()};
                auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                    weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                auto mean_desc = mkldnn_utils::get_input_mkldnn_md(node, 3);
                auto variance_desc = mkldnn_utils::get_input_mkldnn_md(node, 4);
                auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 5);
                auto dinput_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                auto dweights_desc = mkldnn_emitter->build_memory_descriptor(
                    weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);

                auto batchnorm_index =
                    mkldnn_emitter->build_batchnorm_backward(weights_desc,
                                                             input_desc,
                                                             mean_desc,
                                                             variance_desc,
                                                             delta_desc,
                                                             dinput_desc,
                                                             dweights_desc,
                                                             batchnorm->get_eps_value());

                auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                       << ", bn_weights.data());\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1]) << ", "
                       << args[2].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2]) << ", "
                       << args[3].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3]) << ", "
                       << args[4].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[4]) << ", "
                       << args[5].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[5]) << ", "
                       << out[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[6])
                       << ", bn_dweights.data());\n";

                writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                       << to_string(batchnorm_index) << ");\n";

                writer << "memcpy(" << out[1].get_name() << ", &bn_dweights[0], "
                       << args[0].get_size() * args[0].get_element_type().size() << ");\n";
                writer << "memcpy(" << out[2].get_name() << ", &bn_dweights[0]+"
                       << args[0].get_size() << ", "
                       << args[1].get_size() * args[1].get_element_type().size() << ");\n";
                writer.block_end();
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

                    writer.block_begin();
                    writer << emit_vector(out[0]) << "\n    = ";
                    writer << first.get_name() << "[0]\n    * " << emit_vector(second) << ";\n";
                    writer.block_end();
                }
                else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1) &&
                         dot->get_reduction_axes_count() == 1)
                {
                    writer.block_begin();
                    writer << emit_vector(out[0]) << " << \n"
                           << "    " << emit_vector(args[0]) << ".dot(" << emit_vector(args[1])
                           << ");\n";
                    writer.block_end();
                }
                else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) &&
                         dot->get_reduction_axes_count() == 1)
                {
                    writer.block_begin();
                    writer << emit_vector(out[0]) << " = \n"
                           << "    " << emit_matrix(args[0]) << " * " << emit_vector(args[1])
                           << ";\n";
                    writer.block_end();
                }
                else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2) &&
                         dot->get_reduction_axes_count() == 1)
                {
                    // Emit an MKL SGEMM call if possible
                    if (args[0].get_element_type() == element::f32)
                    {
                        writer.block_begin();
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
                        writer.block_end();
                    }
                    else
                    {
                        writer.block_begin();
                        writer << emit_matrix(out[0]) << " = \n"
                               << "    " << emit_matrix(args[0]) << " * " << emit_matrix(args[1])
                               << ";\n";
                        writer.block_end();
                    }
                }
                // Specialized handling of rank 3 tensor multiply rank 2 tensor where
                // each of the
                else if ((arg0_shape.size() == 3) && (arg1_shape.size() == 2) &&
                         dot->get_reduction_axes_count() == 1 &&
                         args[0].get_element_type() == element::f32)
                {
                    auto mat_a = args[0];
                    auto mat_b = args[1];
                    auto mat_c = out[0];
                    const Shape& shape_a = mat_a.get_shape();
                    const Shape& shape_b = mat_b.get_shape();

                    const size_t m = shape_a[1];
                    const size_t k = shape_a[2];
                    const size_t n = shape_b[1];

                    // this also works when mat_a is shape (1, m, k)
                    const size_t offset_a = m * k;
                    // we do not offset mat_b
                    const size_t offset_b = 0;
                    const size_t offset_c = m * n;

                    const size_t group_count = 1;
                    const size_t group_size = shape_a[0];
                    auto populate_array =
                        [&writer](const std::string& var, size_t size, size_t offset) {
                            for (size_t i = 0; i < size; ++i)
                            {
                                writer << var << "+" << i * offset << ((i < size - 1) ? ", " : "");
                            }
                        };

                    writer.block_begin();
                    writer << "cblas::Transpose transa_array[] = {cblas::Transpose::None};\n";
                    writer << "cblas::Transpose transb_array[] = {cblas::Transpose::None};\n";
                    writer << "int64_t m_array[] = {" << m << "};\n";
                    writer << "int64_t n_array[] = {" << n << "};\n";
                    writer << "int64_t k_array[] = {" << k << "};\n";
                    writer << "float alpha_array[] = {1.0f};\n";
                    writer << "std::vector<const float*> a{";
                    populate_array(mat_a.get_name(), group_size, offset_a);
                    writer << "};\n";
                    writer << "const float** a_array = &a[0];\n";
                    writer << "int64_t lda_array[] = {" << std::max(1UL, k) << "};\n";
                    writer << "std::vector<const float*> b{";
                    populate_array(mat_b.get_name(), group_size, offset_b);
                    writer << "};\n";
                    writer << "const float** b_array = &b[0];\n";
                    writer << "int64_t ldb_array[] = {" << std::max(1UL, n) << "};\n";
                    writer << "float beta_array[] = {0.0f};\n";
                    writer << "std::vector<float*> c{";
                    populate_array(mat_c.get_name(), group_size, offset_c);
                    writer << "};\n";
                    writer << "float** c_array = &c[0];\n";
                    writer << "int64_t ldc_array[] = {" << std::max(1UL, n) << "};\n";
                    writer << "int64_t group_size[] = {" << group_size << "};\n";

                    writer << "cblas_sgemm_batch(cblas::Layout::RowMajor, ";
                    writer << "transa_array, transb_array, m_array, n_array, k_array, \n";
                    writer << "alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, \n";
                    writer << "c_array, ldc_array, " << group_count << ", group_size);\n";
                    writer.block_end();
                }
                else
                {
                    writer << "reference::dot(" << args[0].get_name() << ",\n";
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
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] * "
                       << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GetOutputElement)
            {
                auto get_tuple_element = static_cast<const ngraph::op::GetOutputElement*>(node);

                writer.block_begin();
                writer << "memcpy(" << out[0].get_name() << ", "
                       << args[get_tuple_element->get_n()].get_name() << ", "
                       << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Abs)
            {
                writer.block_begin();
                // Some C++ implementations don't like it when we call std::abs on unsigned types, so we will
                // avoid doing so here.
                auto& result_element_type = out[0].get_element_type();

                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name()
                       << "[i] = " << (result_element_type.is_signed() ? "std::abs" : "") << "("
                       << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Concat)
            {
                auto concat = static_cast<const ngraph::op::Concat*>(node);
                if (auto op_annotations = concat->get_op_annotations())
                {
                    auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                    if (in_place_oi_pairs.size() > 0)
                    {
                        auto offset = 0;
                        for (auto i = 0; i < args.size(); i++)
                        {
                            writer << "if (" << args[i].get_name() << " < " << out[0].get_name()
                                   << " || " << args[i].get_name() << " >= " << out[0].get_name()
                                   << " + " << out[0].get_size() << ")\n";
                            writer.block_begin();
                            writer << "memcpy(" << out[0].get_name() << " + " << offset << ", "
                                   << args[i].get_name() << ", "
                                   << args[i].get_size() * out[0].get_element_type().size()
                                   << ");\n";
                            writer.block_end();
                            offset += args[i].get_size();
                        }
                        return;
                    }
                }
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0; i < args.size(); i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_index = 0;
                    size_t concat_dim =
                        (dynamic_cast<const ngraph::op::Concat*>(node))->get_concatenation_axis();
                    concat_index =
                        mkldnn_emitter->build_concat(inputs_data_desc, result_desc, concat_dim);
                    auto& deps = mkldnn_emitter->get_primitive_deps(concat_index);
                    size_t i;
                    for (i = 0; i < args.size(); i++)
                    {
                        writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[i])
                               << ", " << args[i].get_name() << ");\n";
                    }
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[i])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(concat_index) << ");\n";
                }
                else
                {
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
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Divide)
            {
                writer.block_begin();
                if (node->get_element_type().is_real() == false)
                {
                    // Check for divide by zero for integer types only
                    size_t element_count = args[1].get_size();
                    writer << "for (size_t i=0; i<" << element_count << "; i++)\n";
                    writer.block_begin();
                    writer << "if (" << args.at(1).get_name()
                           << "[i] == 0) throw std::runtime_error(\"integer divide by zero\");\n";
                    writer.block_end();
                }
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] / "
                       << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Equal)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name()
                       << "[i] == " << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Greater)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > "
                       << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GreaterEq)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name()
                       << "[i] >= " << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Less)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] < "
                       << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::LessEq)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name()
                       << "[i] <= " << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Any)
            {
                const ngraph::op::Any* any = static_cast<const ngraph::op::Any*>(node);
                writer.block_begin();
                {
                    writer << "reference::any(";
                    writer << "            " << args[0].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(args[0].get_shape()) << "},\n";
                    writer << "            {" << join(out[0].get_shape()) << "},\n";
                    writer << "            {" << join(any->get_reduction_axes()) << "});\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::All)
            {
                const ngraph::op::All* all = static_cast<const ngraph::op::All*>(node);
                writer.block_begin();
                {
                    writer << "reference::all(";
                    writer << "            " << args[0].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(args[0].get_shape()) << "},\n";
                    writer << "            {" << join(out[0].get_shape()) << "},\n";
                    writer << "            {" << join(all->get_reduction_axes()) << "});\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::LRN)
            {
                const ngraph::op::LRN* lrn = static_cast<const ngraph::op::LRN*>(node);

                writer.block_begin();
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_data_desc = runtime::cpu::mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = runtime::cpu::mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto lrn_index =
                        mkldnn_emitter->build_lrn_forward(input_data_desc,
                                                          result_desc,
                                                          static_cast<float>(lrn->get_alpha()),
                                                          static_cast<float>(lrn->get_beta()),
                                                          static_cast<float>(lrn->get_bias()),
                                                          static_cast<int>(lrn->get_nsize()));

                    auto& deps = mkldnn_emitter->get_primitive_deps(lrn_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(lrn_index) << ");\n";
                }
                else
                {
                    writer << "reference::lrn<" << lrn->get_element_type().c_type_string() << ">(";
                    writer << "            " << args[0].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(args[0].get_shape()) << "},\n";
                    writer << "            " << lrn->get_alpha() << ",\n";
                    writer << "            " << lrn->get_beta() << ",\n";
                    writer << "            " << lrn->get_bias() << ",\n";
                    writer << "            " << lrn->get_nsize() << ");\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Log)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = log(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Maximum)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > "
                       << args[1].get_name() << "[i] ? " << args[0].get_name()
                       << "[i] : " << args[1].get_name() << "[i] ;\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Minimum)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] < "
                       << args[1].get_name() << "[i] ? " << args[0].get_name()
                       << "[i] : " << args[1].get_name() << "[i] ;\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Negative)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = -" << args[0].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::NotEqual)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name()
                       << "[i] != " << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Select)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] ? "
                       << args[1].get_name() << "[i] : " << args[2].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Subtract)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] - "
                       << args[1].get_name() << "[i];\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Broadcast)
            {
                auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);

                writer.block_begin();
                kernel::emit_broadcast(writer,
                                       args[0].get_element_type().c_type_string(),
                                       args[0].get_name(),
                                       out[0].get_name(),
                                       args[0].get_shape(),
                                       out[0].get_shape(),
                                       broadcast->get_broadcast_axes());
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Convert)
            {
                auto& result_element_type = out[0].get_element_type();

                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = (" << result_element_type.c_type_string()
                       << ")(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Constant)
            {
                // If an output is a constant then copy it
                size_t output_index = 0;
                for (shared_ptr<Node> result : external_function->get_function()->get_results())
                {
                    if (result.get() == node)
                    {
                        const descriptor::Tensor& tensor = node->get_output_tensor(0);
                        writer << "memcpy(outputs[" << output_index << "], " << tensor.get_name()
                               << ", " << tensor.size() << ");\n";
                    }
                    output_index++;
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reshape)
            {
                auto reshape = static_cast<const ngraph::op::Reshape*>(node);
                auto can_skip_reshape = [&]() {
                    if (!reshape->get_is_transpose())
                    {
                        return true;
                    }
                    auto annotation = reshape->get_op_annotations();
                    if (annotation && annotation->get_in_place_oi_pairs().size() > 0)
                    {
                        return true;
                    }
                    return false;
                };

                if (can_skip_reshape())
                {
                    writer.block_begin();
                    writer << "// Reshape eliminated but copy if needed.\n";
                    writer << "if (" << out[0].get_name() << " != " << args[0].get_name()
                           << ") {\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
                           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "}\n";
                    writer.block_end();
                    return;
                }

                writer.block_begin();
                if (args[0].get_element_type() == element::f32 && args[0].get_shape().size() == 3 &&
                    out[0].get_shape().size() == 3)
                {
                    writer << "cpu::kernel::reshape_3d_3d_float32(" << args[0].get_name() << ", "
                           << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(reshape->get_input_order()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}"
                           << ",  0);\n";
                }
                else if (args[0].get_element_type() == element::f32 &&
                         args[0].get_shape().size() == 4 && out[0].get_shape().size() == 4)
                {
                    writer << "cpu::kernel::reshape_4d_4d_float32(" << args[0].get_name() << ", "
                           << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(reshape->get_input_order()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}"
                           << ", 0);\n";
                }
                else
                {
                    kernel::emit_reshape(writer,
                                         args[0].get_element_type().c_type_string(),
                                         args[0].get_name(),
                                         out[0].get_name(),
                                         args[0].get_shape(),
                                         out[0].get_shape(),
                                         reshape->get_input_order());
                }

                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sign)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = (0 < " << args[0].get_name() << "[i]) - ("
                       << args[0].get_name() << "[i] < 0);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Slice)
            {
                const ngraph::op::Slice* slice = static_cast<const ngraph::op::Slice*>(node);

                if (auto op_annotations = slice->get_op_annotations())
                {
                    auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                    if (in_place_oi_pairs.size() > 0)
                    {
                        auto arg_shape = args[0].get_shape();
                        auto lower_bounds = slice->get_lower_bounds();
                        auto start = 0, accumulated = 1;
                        for (int i = arg_shape.size() - 1; i >= 0; i--)
                        {
                            start += lower_bounds[i] * accumulated;
                            accumulated *= arg_shape[i];
                        }
                        writer << "if (" << out[0].get_name() << " < " << args[0].get_name()
                               << " || " << out[0].get_name() << " >= " << args[0].get_name()
                               << " + " << args[0].get_size() << ")\n";
                        writer.block_begin();
                        writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name()
                               << " + " << start << ", "
                               << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                        writer.block_end();
                        return;
                    }
                }

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto out_shape = out[0].get_shape();

                    auto lower_bounds = slice->get_lower_bounds();

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto slice_index = mkldnn_emitter->build_slice(
                        input_desc, result_desc, lower_bounds, out_shape);
                    auto& deps = mkldnn_emitter->get_primitive_deps(slice_index);

                    writer.block_begin();
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(slice_index) << ");\n";
                    writer.block_end();
                    return;
                }

                writer.block_begin();
                kernel::emit_slice(writer,
                                   args[0].get_element_type().c_type_string(),
                                   args[0].get_name(),
                                   out[0].get_name(),
                                   args[0].get_shape(),
                                   out[0].get_shape(),
                                   slice->get_lower_bounds(),
                                   slice->get_upper_bounds(),
                                   slice->get_strides());
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sum)
            {
                const ngraph::op::Sum* sum = static_cast<const ngraph::op::Sum*>(node);
                writer.block_begin();
                if (args[0].get_element_type() == element::f32 && args[0].get_shape().size() == 1 &&
                    sum->get_reduction_axes().size() == 1)
                {
                    writer << "cpu::kernel::reduce_sum_all_1d_float32(" << args[0].get_name()
                           << ", " << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}"
                           << ", 0);\n";
                }
                else if (args[0].get_element_type() == element::f32 &&
                         args[0].get_shape().size() == 2 && sum->get_reduction_axes().size() == 2)
                {
                    writer << "cpu::kernel::reduce_sum_all_2d_float32(" << args[0].get_name()
                           << ", " << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}"
                           << ", 0);\n";
                }
                else if (args[0].get_element_type() == element::f32 &&
                         args[0].get_shape().size() == 2 && sum->get_reduction_axes().size() == 1)
                {
                    writer << "cpu::kernel::reduce_sum_2d_1rd_float32(" << args[0].get_name()
                           << ", " << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}, "
                           << "{" << join(sum->get_reduction_axes()) << "}"
                           << ", 0);\n";
                }
                else if (args[0].get_element_type() == element::f32 &&
                         args[0].get_shape().size() == 4 && sum->get_reduction_axes().size() == 2)
                {
                    writer << "cpu::kernel::reduce_sum_4d_2rd_float32(" << args[0].get_name()
                           << ", " << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}, "
                           << "{" << join(sum->get_reduction_axes()) << "}"
                           << ", 0);\n";
                }
                else if (args[0].get_element_type() == element::f32 &&
                         args[0].get_shape().size() == 4 && sum->get_reduction_axes().size() == 4)
                {
                    writer << "cpu::kernel::reduce_sum_all_4d_float32(" << args[0].get_name()
                           << ", " << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}"
                           << ");\n";
                }
                else
                {
                    kernel::emit_sum(writer,
                                     args[0].get_element_type().c_type_string(),
                                     args[0].get_name(),
                                     out[0].get_name(),
                                     args[0].get_shape(),
                                     out[0].get_shape(),
                                     sum->get_reduction_axes());
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Exp)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = exp(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::EmbeddingLookup)
            {
                writer.block_begin();
                const ngraph::op::EmbeddingLookup* embed =
                    static_cast<const ngraph::op::EmbeddingLookup*>(node);
                auto index_type_name = embed->get_argument(0)->get_element_type().c_type_string();
                auto type_name = embed->get_element_type().c_type_string();
                auto element_count = shape_size(embed->get_argument(0)->get_shape());
                writer << "reference::embedding<" << type_name << "," << index_type_name << ">(";
                writer << "            " << args[0].get_name() << ",\n";
                writer << "            " << args[1].get_name() << ",\n";
                writer << "            " << out[0].get_name() << ",\n";
                writer << "            " << element_count << ",\n";
                writer << "            {" << join(args[1].get_shape()) << "});\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sin)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = sin(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sinh)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = sinh(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Cos)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = cos(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Cosh)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = cosh(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tan)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = tan(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tanh)
            {
                // Eigen's generic_fast_tanh_float<float> is currently miscompiled by Clang/LLVM
                // so we fall-back to tanh
                // TODO: Implement our own internal fast/approximate tanh if this actually gets used
                // by models
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i=0; i<" << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = tanh(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Asin)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = asin(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Acos)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = acos(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Atan)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = atan(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            static void emitArgMinArgMax(const std::vector<TensorViewWrapper>& args,
                                         const std::vector<TensorViewWrapper>& out,
                                         size_t reduction_axis,
                                         const char* kernel_name,
                                         codegen::CodeWriter& writer)
            {
                if (out[0].get_element_type() != element::i64 &&
                    out[0].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }

                writer.block_begin();
                writer << "reference::" << kernel_name << "<" << args[0].get_type() << ", "
                       << out[0].get_element_type().c_type_string() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                   " << out[0].get_name() << ",\n";
                writer << "                   {" << join(args[0].get_shape()) << "},\n";
                writer << "                   {" << join(out[0].get_shape()) << "},\n";
                writer << "                   " << reduction_axis << ");\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ArgMin)
            {
                auto argmin = static_cast<const ngraph::op::ArgMin*>(node);
                emitArgMinArgMax(args, out, argmin->get_reduction_axis(), "argmin", writer);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ArgMax)
            {
                auto argmax = static_cast<const ngraph::op::ArgMax*>(node);
                emitArgMinArgMax(args, out, argmax->get_reduction_axis(), "argmax", writer);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::TopK)
            {
                auto topk = static_cast<const ngraph::op::TopK*>(node);
                if (out[0].get_element_type() != element::i64 &&
                    out[0].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }

                writer.block_begin();
                writer << "reference::topk<" << args[0].get_type() << ", "
                       << out[0].get_element_type().c_type_string() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                   " << out[0].get_name() << ",\n";
                writer << "                   " << out[1].get_name() << ",\n";
                writer << "                   {" << join(args[0].get_shape()) << "},\n";
                writer << "                   {" << join(out[0].get_shape()) << "},\n";
                writer << "                   " << topk->get_top_k_axis() << ",\n";
                writer << "                   " << topk->get_k() << ",\n";
                writer << "                   " << topk->get_compute_max() << ");\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Power)
            {
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = pow(" << args[0].get_name() << "[i], "
                       << args[1].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::UpdateSlice)
            {
                auto update_slice = static_cast<const ngraph::op::UpdateSlice*>(node);
                const Shape& arg0_shape = args[0].get_shape();
                const Shape& arg1_shape = args[1].get_shape();
                auto strides = update_slice->get_strides();
                writer.block_begin();
                if (!ngraph::is_strided(strides))
                {
                    writer << "cpu::kernel::update_slice<"
                           << args[0].get_element_type().c_type_string() << ", "
                           << arg0_shape.size() << ">(\n"
                           << "                                " << args[0].get_name() << ",\n"
                           << "                                " << args[1].get_name() << ",\n"
                           << "                                " << out[0].get_name() << ",\n"
                           << "                               {" << join(arg0_shape) << "},\n"
                           << "                               {" << join(arg1_shape) << "},\n"
                           << "                               {"
                           << join(update_slice->get_lower_bounds()) << "},\n"
                           << "0);\n";
                }
                else
                {
                    writer << "cpu::kernel::strided_update_slice<"
                           << args[0].get_element_type().c_type_string() << ", "
                           << arg0_shape.size() << ">(\n"
                           << "                                " << args[0].get_name() << ",\n"
                           << "                                " << args[1].get_name() << ",\n"
                           << "                                " << out[0].get_name() << ",\n"
                           << "                               {" << join(arg0_shape) << "},\n"
                           << "                               {" << join(arg1_shape) << "},\n"
                           << "                               {"
                           << join(update_slice->get_lower_bounds()) << "},\n"
                           << "                               {"
                           << join(update_slice->get_upper_bounds()) << "},\n"
                           << "                               {"
                           << join(update_slice->get_strides()) << "},\n"
                           << "0);\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReplaceSlice)
            {
                auto replace_slice = static_cast<const ngraph::op::ReplaceSlice*>(node);
                writer.block_begin();
                if (args[0].get_name() != out[0].get_name())
                {
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
                }
                else
                {
                    kernel::emit_replace_slice_inplace(writer,
                                                       args[0].get_element_type().c_type_string(),
                                                       args[0].get_name(),
                                                       args[1].get_name(),
                                                       args[1].get_shape(),
                                                       args[0].get_shape(),
                                                       replace_slice->get_lower_bounds(),
                                                       replace_slice->get_upper_bounds(),
                                                       replace_slice->get_strides());
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::OneHot)
            {
                auto oh = static_cast<const ngraph::op::OneHot*>(node);

                auto arg_rank = args[0].get_shape().size();

                size_t bounds = out[0].get_shape()[oh->get_one_hot_axis()];

                if (arg_rank == 0)
                {
                    writer.block_begin();

                    writer << emit_vector(out[0], "out_vector") << ";\n";

                    writer << "out_vector.setZero();\n"
                           << ""
                           << "auto pos_raw = " << emit_vector(args[0]) << "(0, 0);\n"
                           << "if (floor(pos_raw) != pos_raw)\n";
                    writer.block_begin();
                    writer
                        << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
                    writer.block_end();

                    writer << "size_t pos = pos_raw;\n"
                           << "if (pos >= " << bounds << ")\n";

                    writer.block_begin();
                    writer << "throw(std::range_error(\"One-hot: value is out of category "
                              "range\"));\n";
                    writer.block_end();

                    writer << "out_vector(pos, 0) = 1;\n";

                    writer.block_end();
                }
                else if (arg_rank == 1)
                {
                    writer.block_begin();

                    writer << emit_vector(args[0], "arg_vector") << ";\n";

                    writer << emit_matrix(out[0], "out_vector") << ";\n";
                    writer << "out_vector.setZero();\n";

                    writer << "for (size_t i = 0; i < " << args[0].get_shape()[0] << "; i++)\n";
                    writer.block_begin();

                    writer << "auto pos_raw = arg_vector(i, 0);\n";

                    writer << "if (floor(pos_raw) != pos_raw)\n";
                    writer.block_begin();
                    writer
                        << "throw(std::range_error(\"One-hot: non-integral value in input\"));\n";
                    writer.block_end();

                    writer << "size_t pos = pos_raw;\n";
                    writer << "bool found = false;\n";

                    writer << "if (pos >= " << bounds << ")\n";
                    writer.block_begin();
                    writer << "throw(std::range_error(\"One-hot: value is out of category "
                              "range\"));\n";
                    writer.block_end();

                    writer << "out_vector"
                           << (oh->get_one_hot_axis() == 0 ? "(pos, i)" : "(i, pos)") << " = 1;\n";

                    writer.block_end();

                    writer.block_end();
                }
                // Other cases are not handled yet.
                else
                {
                    writer << "reference::one_hot<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   {" << join(args[0].get_shape()) << "},\n";
                    writer << "                   {" << join(out[0].get_shape()) << "},\n";
                    writer << "                   " << oh->get_one_hot_axis() << ");\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Ceiling)
            {
                writer.block_begin();
                size_t element_count = out[0].get_size();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = ceil(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Floor)
            {
                writer.block_begin();
                size_t element_count = out[0].get_size();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = floor(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sqrt)
            {
                writer.block_begin();
                size_t element_count = out[0].get_size();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << element_count << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = sqrt(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionRelu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::ConvolutionRelu>(
                            node, args, out);
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
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionRelu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolutionRelu>(
                            node, args, out);
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
                    throw ngraph_error("unsupported parameters for QuantizedConvolutionRelu");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolution)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolution>(
                            node, args, out);
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
                    throw ngraph_error("unsupported parameters for QuantizedConvolution");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GroupConvolution)
            {
                auto convolution = static_cast<const ngraph::op::GroupConvolution*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    Strides window_dilation_strides_adjusted;
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    size_t conv_index =
                        mkldnn_emitter->build_convolution_forward(input_data_desc,
                                                                  weights_desc,
                                                                  result_desc,
                                                                  filter_strides,
                                                                  window_dilation_strides_adjusted,
                                                                  padding_below,
                                                                  padding_above);

                    // invoke group convolution
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
                    throw ngraph_error("unsupported parameters for GroupConvolution");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GroupConvolutionBias)
            {
                auto convolution = static_cast<const ngraph::op::GroupConvolutionBias*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto arg2_shape = args[2].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    Strides window_dilation_strides_adjusted;
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    const float ops_scale = 1.f;
                    const float ops_alpha = -0.f; // relu negative slope
                    const float ops_beta = 0.f;

                    mkldnn::post_ops ops;
                    if (convolution->with_relu())
                    {
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    size_t conv_index =
                        mkldnn_emitter->build_convolution_forward(input_data_desc,
                                                                  weights_desc,
                                                                  bias_desc,
                                                                  result_desc,
                                                                  filter_strides,
                                                                  window_dilation_strides_adjusted,
                                                                  padding_below,
                                                                  padding_above,
                                                                  ops);

                    // invoke group convolution bias
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(conv_index) << ");\n";
                }
                else
                {
                    throw ngraph_error("Unsupported parameters for GroupConvolutionBias");
                }
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
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::Convolution>(node, args, out);
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
                    writer << "reference::convolution<" << out[0].get_type() << ">("
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
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter
                            ->build_convolution_backward<ngraph::op::ConvolutionBackpropFilters>(
                                node, args, out);
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
                    writer << "reference::convolution<" << out[0].get_type() << ">("
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
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter
                            ->build_convolution_backward<ngraph::op::ConvolutionBackpropData>(
                                node, args, out);
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
                    // Note that args[1] and args[0] are switched here from the usual order.
                    writer << "reference::convolution<" << out[0].get_type() << ">("
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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto qconv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolutionBias>(
                            node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(qconv_index);

                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(qconv_index) << ");\n";
                }
                else
                {
                    throw ngraph_error(
                        "QuantizedConvolutionBias is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBiasAdd)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto qconv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolutionBiasAdd>(
                            node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(qconv_index);

                    writer << "if (" << out[0].get_name() << " != " << args[3].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[3].get_name() << ", "
                           << args[3].get_size() * args[3].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(qconv_index) << ");\n";
                }
                else
                {
                    throw ngraph_error(
                        "QuantizedConvolutionBiasAdd is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBiasSignedAdd)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto qconv_index =
                        mkldnn_emitter
                            ->build_convolution<ngraph::op::QuantizedConvolutionBiasSignedAdd>(
                                node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(qconv_index);

                    writer << "if (" << out[0].get_name() << " != " << args[3].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[3].get_name() << ", "
                           << args[3].get_size() * args[3].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(qconv_index) << ");\n";
                }
                else
                {
                    throw ngraph_error(
                        "QuantizedConvolutionBiasSignedAdd is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::ConvolutionBias>(
                            node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(conv_index) << ");\n";
                }
                else
                {
                    throw ngraph_error("ConvolutionBias is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBiasAdd)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::ConvolutionBiasAdd>(
                            node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    writer << "if (" << out[0].get_name() << " != " << args[3].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[3].get_name() << ", "
                           << args[3].get_size() * args[3].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(conv_index) << ");\n";
                }
                else
                {
                    throw ngraph_error("ConvolutionBiasAdd is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionAdd)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index = mkldnn_emitter->build_convolution<ngraph::op::ConvolutionAdd>(
                        node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    writer << "if (" << out[0].get_name() << " != " << args[2].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[2].get_name() << ", "
                           << args[2].get_size() * args[2].get_element_type().size() << ");\n";
                    writer.block_end();
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
                    throw ngraph_error("ConvolutionAdd is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias)
            {
                if (mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index = mkldnn_emitter->build_convolution_backward<
                        ngraph::op::ConvolutionBiasBackpropFiltersBias>(node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[3])
                           << ", " << out[1].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(conv_index) << ");\n";
                }
                else
                {
                    throw ngraph_error(
                        "ConvolutionBiasBackpropFiltersBias is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Not)
            {
                writer << "reference::logical_not(" << args[0].get_name() << ",\n"
                       << "                    " << out[0].get_name() << ",\n"
                       << "                    " << out[0].get_size() << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPool)
            {
                auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);

                auto arg_shape = args[0].get_shape();

                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t max_pool_index = mkldnn_emitter->build_pooling_forward(
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        max_pool->get_window_movement_strides(),
                        max_pool->get_window_shape(),
                        max_pool->get_padding_below(),
                        max_pool->get_padding_above());

                    auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(max_pool_index) << ");\n";
                }
                else
                {
                    writer << "reference::max_pool<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedMaxPool)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    size_t qmax_pool_index = mkldnn_emitter->build_quantized_max_pool(node);
                    auto& deps = mkldnn_emitter->get_primitive_deps(qmax_pool_index);

                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(qmax_pool_index) << ");\n";
                }
                else
                {
                    throw ngraph_error("unsupported parameters for QuantizedMaxPool");
                }
            }
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedAvgPool)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    size_t qavg_pool_index = mkldnn_emitter->build_quantized_avg_pool(node);
                    auto& deps = mkldnn_emitter->get_primitive_deps(qavg_pool_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(qavg_pool_index) << ");\n";
                }
                else
                {
                    throw ngraph_error("unsupported parameters for QuantizedAvgPool");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolWithIndices)
            {
                auto max_pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t max_pool_index = mkldnn_emitter->build_max_pooling_with_indices_forward(
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        max_pool->get_window_movement_strides(),
                        max_pool->get_window_shape(),
                        max_pool->get_padding_below(),
                        max_pool->get_padding_above());

                    auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << out[1].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(max_pool_index) << ");\n";
                }
                else
                {
                    throw ngraph_error("MaxPoolWithIndices isn't supported");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reverse)
            {
                auto reverse = static_cast<const ngraph::op::Reverse*>(node);

                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                writer << "reference::reverse<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                " << out[0].get_name() << ",\n";
                writer << "                {" << join(arg_shape) << "},\n";
                writer << "                {" << join(result_shape) << "},\n";
                writer << "                {" << join(reverse->get_reversed_axes()) << "});\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReverseSequence)
            {
                auto rs = static_cast<const ngraph::op::ReverseSequence*>(node);

                string iv_prefix{"i"};
                size_t ibi = rs->get_batch_axis();
                string bi = iv_prefix + std::to_string(ibi);
                string si = iv_prefix + std::to_string(rs->get_sequence_axis());
                auto arg_shape = args[0].get_shape();

                // iterate over seq_lengths make sure indices aren't out of bounds and normalize
                writer << "std::vector<size_t> norm_seq_lengths (" << arg_shape.at(ibi) << ");\n";
                writer << emit_for_lt(iv_prefix, ibi, arg_shape.at(ibi));
                writer.block_begin();
                writer << "auto seq_len = static_cast<size_t>(" << args[1].get_name() << "[" << bi
                       << "]);\n";
                writer << "if (seq_len > " << arg_shape.at(rs->get_sequence_axis()) << ")\n";
                writer.block_begin();
                writer << "throw \"One of the elements of sequence lengths is greater than "
                          "sequence axis\";\n";
                writer.block_end();

                writer << "if (seq_len == 0)\n";
                writer.block_begin();
                writer << "norm_seq_lengths[" << bi << "] = 1;\n";
                writer.block_end();
                writer << " else \n";
                writer.block_begin();
                writer << "norm_seq_lengths[" << bi << "] = seq_len;\n";
                writer.block_end();

                writer.block_end();

                std::vector<std::string> sdims;
                for (auto d : arg_shape)
                {
                    sdims.push_back(std::to_string(d));
                }

                // convert input and output into multidimensional arrays
                auto isdims = emit_indices(sdims);
                writer << args[0].get_type() << "(&src)" << isdims << " = *reinterpret_cast<"
                       << args[0].get_type() << " (*)" << isdims << ">(" << args[0].get_name()
                       << ");\n";

                writer << args[0].get_type() << "(&dst)" << isdims << " = *reinterpret_cast<"
                       << args[0].get_type() << " (*)" << isdims << ">(" << out[0].get_name()
                       << ");\n";

                // reverse sequence
                std::vector<std::string> source_indices;
                for (size_t i = 0; i < arg_shape.size(); i++)
                {
                    writer << emit_for_lt(iv_prefix, i, arg_shape.at(i));
                    writer.block_begin();
                    source_indices.push_back(iv_prefix + std::to_string(i));
                }

                writer << "auto seq_len = norm_seq_lengths[" << bi << "];\n";
                writer << "size_t sequence_index = " << si << " < seq_len "
                       << "? seq_len - " << si << " - 1"
                       << ": " << si << ";\n";

                std::vector<std::string> output_indices(source_indices);
                output_indices.at(rs->get_sequence_axis()) = "sequence_index";

                writer << "dst" << emit_indices(output_indices) << " = "
                       << "src" << emit_indices(source_indices) << ";\n";

                for (size_t i = 0; i < arg_shape.size(); i++)
                {
                    writer.block_end();
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AvgPool)
            {
                auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);

                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t avg_pool_index = mkldnn_emitter->build_pooling_forward(
                        (avg_pool->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        input_desc,
                        result_desc,
                        avg_pool->get_window_movement_strides(),
                        avg_pool->get_window_shape(),
                        avg_pool->get_padding_below(),
                        avg_pool->get_padding_above());

                    auto& deps = mkldnn_emitter->get_primitive_deps(avg_pool_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(avg_pool_index) << ");\n";
                }
                else
                {
                    writer << "reference::avg_pool<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
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

                if (arg0_shape.size() == 4 && args[0].get_element_type() == element::f32 &&
                    pad->get_padding_interior() == Shape(arg0_shape.size()))
                {
                    writer << "cpu::kernel::pad_4d_float32(" << args[0].get_name() << ",\n"
                           << "                            " << out[0].get_name() << ",\n"
                           << "                            " << args[1].get_name() << ",\n"
                           << "                            {" << join(arg0_shape) << "},\n"
                           << "                            {" << join(result_shape) << "},\n"
                           << "                            {" << join(pad->get_padding_below())
                           << "},\n"
                           << "                            {" << join(pad->get_padding_above())
                           << "}, 0);\n";
                }
                else
                {
                    writer << "reference::pad<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "            " << args[1].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(arg0_shape) << "},\n";
                    writer << "            {" << join(result_shape) << "},\n";
                    writer << "            {" << join(pad->get_padding_below()) << "},\n";
                    writer << "            {" << join(pad->get_padding_above()) << "},\n";
                    writer << "            {" << join(pad->get_padding_interior()) << "});\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AvgPoolBackprop)
            {
                auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);

                auto delta_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t avg_pool_index = mkldnn_emitter->build_pooling_backward(
                        (apb->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        diff_dst_desc,
                        diff_src_desc,
                        apb->get_window_movement_strides(),
                        apb->get_window_shape(),
                        apb->get_padding_below(),
                        apb->get_padding_above());

                    auto& deps = mkldnn_emitter->get_primitive_deps(avg_pool_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(avg_pool_index) << ");\n";
                }
                else
                {
                    writer << "reference::avg_pool_backprop<" << out[0].get_type() << ">("
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
                auto out_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto fprop_src_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t max_pool_index = mkldnn_emitter->build_max_pooling_backward(
                        mkldnn::algorithm::pooling_max,
                        fprop_src_desc,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above());

                    auto& fdeps = mkldnn_emitter->get_primitive_deps(max_pool_index - 1);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(fdeps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(fdeps[1])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(fdeps[2])
                           << ", ctx->mkldnn_workspaces[" << fdeps[3] << "]);\n";
                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(max_pool_index - 1) << ");\n";

                    auto& bdeps = mkldnn_emitter->get_primitive_deps(max_pool_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(bdeps[0])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(bdeps[1])
                           << ", ctx->mkldnn_workspaces[" << bdeps[3] << "]);\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(bdeps[2])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(max_pool_index) << ");\n";
                }
                else
                {
                    writer << "reference::max_pool_backprop<" << out[0].get_type() << ">("
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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolWithIndicesBackprop)
            {
                auto mpb = static_cast<const ngraph::op::MaxPoolWithIndicesBackprop*>(node);

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t max_pool_index = mkldnn_emitter->build_max_pooling_with_indices_backward(
                        mkldnn::algorithm::pooling_max,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above());

                    auto& bdeps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(bdeps[0])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(bdeps[1])
                           << ", " << args[2].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(bdeps[2])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(max_pool_index) << ");\n";
                }
                else
                {
                    throw ngraph_error("MaxPoolWithIndicesBackprop isn't supported");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Product)
            {
                const ngraph::op::Product* product = static_cast<const ngraph::op::Product*>(node);
                writer.block_begin();
                // TODO: add an emitter akin to the emit_sum
                writer << "reference::product<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                         " << out[0].get_name() << ",\n";
                writer << "                         {" << join(args[0].get_shape()) << "},\n";
                writer << "                         {" << join(out[0].get_shape()) << "},\n";
                writer << "                         {" << join(product->get_reduction_axes())
                       << "});\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Max)
            {
                const ngraph::op::Max* max = static_cast<const ngraph::op::Max*>(node);
                writer.block_begin();
                if (args[0].get_element_type() == element::f32 && args[0].get_shape().size() == 2 &&
                    max->get_reduction_axes().size() == 1)
                {
                    writer << "cpu::kernel::reduce_max_2d_1rd_float32(" << args[0].get_name()
                           << ", " << out[0].get_name() << ", "
                           << "{" << join(args[0].get_shape()) << "}, "
                           << "{" << join(out[0].get_shape()) << "}, "
                           << "{" << join(max->get_reduction_axes()) << "}"
                           << ", 0);\n";
                }
                else
                {
                    writer << "reference::max<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(args[0].get_shape()) << "},\n";
                    writer << "                         {" << join(out[0].get_shape()) << "},\n";
                    writer << "                         {" << join(max->get_reduction_axes())
                           << "});\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Min)
            {
                const ngraph::op::Min* min = static_cast<const ngraph::op::Min*>(node);
                writer.block_begin();
                // TODO: add an emitter akin to the emit_sum
                writer << "reference::min<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                         " << out[0].get_name() << ",\n";
                writer << "                         {" << join(args[0].get_shape()) << "},\n";
                writer << "                         {" << join(out[0].get_shape()) << "},\n";
                writer << "                         {" << join(min->get_reduction_axes())
                       << "});\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::runtime::cpu::op::ConvertLayout)
            {
                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                //this is a special case to handle nchw(oihw) to goihw/Goihw16g/Goihw8g for GroupConvolution's weights
                if (input_desc.data.format == mkldnn_nchw &&
                    result_desc.data.format == mkldnn_goihw)
                {
                    input_desc = result_desc;
                }
                else if (input_desc.data.format == mkldnn_nchw &&
                         input_desc.data.ndims == 4 /*nchw*/ &&
                         result_desc.data.ndims == 5 /*Goihw16g/Goihw8g/etc*/ &&
                         node->get_users().size() == 1)
                {
                    Shape weights_shape_groups;
                    if (auto gconv = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(
                            node->get_users()[0]))
                    {
                        weights_shape_groups = gconv->get_weights_dimensions();
                    }
                    else if (auto gconvb =
                                 std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(
                                     node->get_users()[0]))
                    {
                        weights_shape_groups = gconvb->get_weights_dimensions();
                    }
                    else
                    {
                        throw ngraph_error("Incompatible input/output shape in ConvertLayout op");
                    }
                    input_desc = mkldnn::memory::desc(
                        mkldnn::memory::dims(weights_shape_groups.begin(),
                                             weights_shape_groups.end()),
                        mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                        mkldnn::memory::format::goihw);
                }

                size_t reorder_index = mkldnn_emitter->build_reorder(input_desc, result_desc);

                auto& deps = mkldnn_emitter->get_primitive_deps(reorder_index);
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0]) << ", "
                       << args[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1]) << ", "
                       << out[0].get_name() << ");\n";

                writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                       << to_string(reorder_index) << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReluBackprop)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t relu_index =
                        mkldnn_emitter->build_relu_backward(input_desc, delta_desc, result_desc);

                    auto& deps = mkldnn_emitter->get_primitive_deps(relu_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << args[1].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(relu_index) << ");\n";
                }
                else
                {
                    writer << "#pragma omp parallel for\n";
                    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                    writer.block_begin();
                    writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > 0 ? "
                           << args[1].get_name() << "[i] : 0;\n";
                    writer.block_end();
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Relu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t relu_index = mkldnn_emitter->build_relu_forward(input_desc, result_desc);

                    auto& deps = mkldnn_emitter->get_primitive_deps(relu_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(relu_index) << ");\n";
                }
                else
                {
                    writer << "#pragma omp parallel for\n";
                    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                    writer.block_begin();
                    writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > 0 ? "
                           << args[0].get_name() << "[i] : 0;\n";
                    writer.block_end();
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::LeakyRelu)
            {
                auto leaky_relu_node = static_cast<const ngraph::op::LeakyRelu*>(node);
                float alpha = leaky_relu_node->get_alpha();
                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto leaky_relu_index =
                        mkldnn_emitter->build_leaky_relu(input_desc, result_desc, alpha);
                    auto& deps = mkldnn_emitter->get_primitive_deps(leaky_relu_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(leaky_relu_index) << ");\n";
                }
                else
                {
                    writer << "#pragma omp parallel for\n";
                    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                    writer.block_begin();
                    writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] > 0 ? "
                           << args[0].get_name() << "[i] : (" << alpha << " * "
                           << args[0].get_name() << "[i]);\n";
                    writer.block_end();
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BoundedRelu)
            {
                auto bounded_relu_node = static_cast<const ngraph::op::BoundedRelu*>(node);
                float alpha = bounded_relu_node->get_alpha();
                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto bounded_relu_index =
                        mkldnn_emitter->build_bounded_relu(input_desc, result_desc, alpha);
                    auto& deps = mkldnn_emitter->get_primitive_deps(bounded_relu_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(bounded_relu_index) << ");\n";
                }
                else
                {
                    writer << "#pragma omp parallel for\n";
                    writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                    writer.block_begin();
                    writer << args[0].get_name() << "[i] = " << args[0].get_name() << "[i] > 0 ? "
                           << args[0].get_name() << "[i] : 0;\n";
                    writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] < "
                           << alpha << " ? " << args[0].get_name() << "[i] : " << alpha << ";\n";
                    writer.block_end();
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sigmoid)
            {
                auto input_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                size_t sigmoid_index =
                    mkldnn_emitter->build_sigmoid_forward(input_desc, result_desc);

                auto& deps = mkldnn_emitter->get_primitive_deps(sigmoid_index);
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0]) << ", "
                       << args[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1]) << ", "
                       << out[0].get_name() << ");\n";

                writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                       << to_string(sigmoid_index) << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidBackprop)
            {
                auto input_shape = args[0].get_shape();
                auto delta_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                size_t sigmoid_index =
                    mkldnn_emitter->build_sigmoid_backward(input_desc, delta_desc, result_desc);

                auto& deps = mkldnn_emitter->get_primitive_deps(sigmoid_index);
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0]) << ", "
                       << args[0].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1]) << ", "
                       << args[1].get_name() << ");\n";
                writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[2]) << ", "
                       << out[0].get_name() << ");\n";

                writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                       << to_string(sigmoid_index) << ");\n";
            }

            std::string
                generate_sigmoid_mul_func(const ngraph::op::SigmoidMultiply::FunctionType type,
                                          const std::string& input,
                                          const std::string& out_numer,
                                          const std::string& out_denom,
                                          bool derivative)
            {
                std::string func_block;
                switch (type)
                {
                case ngraph::op::SigmoidMultiply::FunctionType::Logistic:
                    func_block = "auto e_x = exp(" + input + ");\n";
                    func_block += out_numer + " = e_x;\n";
                    func_block += out_denom + " = e_x+1;\n";
                    if (derivative)
                    {
                        func_block += "d_" + out_numer + " = " + out_numer + ";\n";
                        func_block +=
                            "d_" + out_denom + " = " + out_denom + " * " + out_denom + ";\n";
                    }
                    break;
                case ngraph::op::SigmoidMultiply::FunctionType::Tanh:
                    func_block = "auto e_2x = exp(2.0*" + input + ");\n";
                    func_block += out_numer + " = e_2x-1;\n";
                    func_block += out_denom + " = e_2x+1;\n";
                    if (derivative)
                    {
                        func_block += "d_" + out_numer + " = 4.0*e_2x;\n";
                        func_block +=
                            "d_" + out_denom + " = " + out_denom + " * " + out_denom + ";\n";
                    }
                    break;
                case ngraph::op::SigmoidMultiply::FunctionType::Identity:
                    func_block = out_numer + " = " + input + ";\n";
                    func_block += out_denom + " = 1;\n";
                    if (derivative)
                    {
                        func_block += "d_" + out_numer + " = 1;\n";
                        func_block += "d_" + out_denom + " = 1;\n";
                    }
                    break;
                }
                if (func_block.empty())
                {
                    throw ngraph_error(
                        "generate_sigmoid_mul_func input function type not supported");
                }
                return func_block;
            }
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidMultiply)
            {
                auto sigmoid_mul = static_cast<const ngraph::op::SigmoidMultiply*>(node);
                std::string numer_0 = "numer_0";
                std::string denom_0 = "denom_0";
                std::string numer_1 = "numer_1";
                std::string denom_1 = "denom_1";
                std::string input_0_func_string =
                    generate_sigmoid_mul_func(sigmoid_mul->get_input_func_type(0),
                                              args[0].get_name() + "[i]",
                                              numer_0,
                                              denom_0,
                                              false);
                std::string input_1_func_string =
                    generate_sigmoid_mul_func(sigmoid_mul->get_input_func_type(1),
                                              args[1].get_name() + "[i]",
                                              numer_1,
                                              denom_1,
                                              false);

                writer.block_begin();
                writer << "#pragma omp parallel for simd\n";
                writer << "for (size_t i=0; i<" << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << "float " << numer_0 << ";\n";
                writer << "float " << denom_0 << ";\n";
                writer.block_begin();
                writer << input_0_func_string;
                writer.block_end();
                writer << "float " << numer_1 << ";\n";
                writer << "float " << denom_1 << ";\n";
                writer.block_begin();
                writer << input_1_func_string;
                writer.block_end();
                writer << out[0].get_name()
                       << "[i] = (" + numer_0 + " * " + numer_1 + ") / (" + denom_0 + " * " +
                              denom_1 + ");\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidMultiplyBackprop)
            {
                // math: we have sigmoid functions f(x) and g(y) multiplied, z = f(x) * g(y)
                // dz/dx = dz/df * df/dx = g(y) * f'(x)
                // dz/dy = dz/dg * dg/dy = f(x) * g'(y)
                auto sigmoid_mul_backprop =
                    static_cast<const ngraph::op::SigmoidMultiplyBackprop*>(node);
                const TensorViewWrapper& data_0 = args[0];
                const TensorViewWrapper& data_1 = args[1];
                const TensorViewWrapper& delta = args[2];
                const TensorViewWrapper& input_0_delta = out[0];
                const TensorViewWrapper& input_1_delta = out[1];
                std::string numer_0 = "numer_0";
                std::string denom_0 = "denom_0";
                std::string numer_1 = "numer_1";
                std::string denom_1 = "denom_1";
                std::string d_numer_0 = "d_numer_0";
                std::string d_denom_0 = "d_denom_0";
                std::string d_numer_1 = "d_numer_1";
                std::string d_denom_1 = "d_denom_1";
                std::string input_0_func_string =
                    generate_sigmoid_mul_func(sigmoid_mul_backprop->get_input_func_type(0),
                                              data_0.get_name() + "[i]",
                                              numer_0,
                                              denom_0,
                                              true);
                std::string input_1_func_string =
                    generate_sigmoid_mul_func(sigmoid_mul_backprop->get_input_func_type(1),
                                              data_1.get_name() + "[i]",
                                              numer_1,
                                              denom_1,
                                              true);
                writer.block_begin();
                writer << "#pragma omp parallel for simd\n";
                writer << "for (size_t i=0; i<" << input_0_delta.get_size() << "; i++)\n";
                writer.block_begin();
                writer << "float " << numer_0 << ";\n";
                writer << "float " << denom_0 << ";\n";
                writer << "float " << d_numer_0 << ";\n";
                writer << "float " << d_denom_0 << ";\n";
                writer.block_begin();
                writer << input_0_func_string;
                writer.block_end();
                writer << "float " << numer_1 << ";\n";
                writer << "float " << denom_1 << ";\n";
                writer << "float " << d_numer_1 << ";\n";
                writer << "float " << d_denom_1 << ";\n";
                writer.block_begin();
                writer << input_1_func_string;
                writer.block_end();
                writer << input_0_delta.get_name()
                       << "[i] = " + delta.get_name() + "[i]*(" + numer_1 + "*" + d_numer_0 +
                              ")/(" + denom_1 + "*" + d_denom_0 + ");\n";
                writer << input_1_delta.get_name()
                       << "[i] = " + delta.get_name() + "[i]*(" + numer_0 + "*" + d_numer_1 +
                              ")/(" + denom_0 + "*" + d_denom_1 + ");\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Softmax)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto softmax = static_cast<const ngraph::op::Softmax*>(node);

                    if (softmax->get_axes().size() != 1)
                    {
                        throw ngraph_error("MKLDNN supports softmax only across single axis");
                    }
                    int softmax_axis = static_cast<int>(*(softmax->get_axes().begin()));
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t softmax_index = mkldnn_emitter->build_softmax_forward(
                        input_desc, result_desc, softmax_axis);

                    auto& deps = mkldnn_emitter->get_primitive_deps(softmax_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";

                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(softmax_index) << ");\n";
                }
                else
                {
                    writer.block_begin();
                    const ngraph::op::Softmax* softmax =
                        static_cast<const ngraph::op::Softmax*>(node);
                    auto type = out[0].get_type();
                    auto shape = out[0].get_shape();
                    auto dims = out[0].get_shape().size();
                    auto axes = softmax->get_axes();

                    // create arg/out if 1d
                    if (dims < 1)
                    {
                        writer << type << "* arg = " << args[0].get_name() << "\n";
                        writer << type << "* out = " << out[0].get_name() << "\n";
                    }
                    // else cast arg/out to an Nd array
                    else
                    {
                        std::string shape1toN;
                        for (size_t d = 1; d < dims; ++d)
                        {
                            shape1toN += "[";
                            shape1toN += std::to_string(shape[d]);
                            shape1toN += "]";
                        }

                        writer << type << " (*arg)" << shape1toN << " = (" << type << " (*)"
                               << shape1toN << ") " << args[0].get_name() << ";\n";
                        writer << type << " (*out)" << shape1toN << " = (" << type << " (*)"
                               << shape1toN << ") " << out[0].get_name() << ";\n";
                    }

                    // build arg/out index
                    std::string index;
                    for (size_t d = 0; d < dims; ++d)
                    {
                        index += "[i";
                        index += std::to_string(d);
                        index += "]";
                    }

                    // calculate e ^ (arg - max)
                    // outer loop(s) - for axis not in axes
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) == axes.end())
                        {
                            writer << "#pragma omp parallel for\n";
                            writer << "for (size_t i" << d << " = 0; i" << d << " < " << shape[d]
                                   << "; ++i" << d << ")\n";
                            writer.block_begin();
                        }
                    }

                    // max inner loop(s)
                    writer << type << " m = 0;\n"; // TODO: needs to be minval for the type

                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer << "for (size_t i" << d << " = 0; i" << d << " < " << shape[d]
                                   << "; ++i" << d << ")\n";
                            writer.block_begin();
                        }
                    }

                    writer << "if (arg" << index << " > m)\n";
                    writer.block_begin();
                    writer << "m = arg" << index << ";\n";
                    writer.block_end();

                    // end max inner loop(s)
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer.block_end();
                        }
                    }

                    // e ^ (arg - max) inner loop
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer << "for (size_t i" << d << " = 0; i" << d << " < " << shape[d]
                                   << "; ++i" << d << ")\n";
                            writer.block_begin();
                        }
                    }

                    writer << "out" << index << " = exp(arg" << index << " - m);\n";

                    // end e ^ (arg - max) inner loop
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer.block_end();
                        }
                    }

                    // end e ^ (arg - max) outer loop(s)
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) == axes.end())
                        {
                            writer.block_end();
                        }
                    }

                    // calculate softmax = e ^ (arg - max) / sum (e ^ (arg - max))
                    // outer loop(s) - for axis not in axes
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) == axes.end())
                        {
                            writer << "#pragma omp parallel for\n";
                            writer << "for (size_t i" << d << " = 0; i" << d << " < " << shape[d]
                                   << "; ++i" << d << ")\n";
                            writer.block_begin();
                        }
                    }

                    // sum (e ^ (arg - max) inner loop(s)
                    writer << type << " d = 0;\n";

                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer << "for (size_t i" << d << " = 0; i" << d << " < " << shape[d]
                                   << "; ++i" << d << ")\n";
                            writer.block_begin();
                        }
                    }

                    writer << "d += out" << index << ";\n";

                    // end sum (e ^ (arg - max) inner loop(s)
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer.block_end();
                        }
                    }

                    writer << "d = 1 / d;\n";

                    // softmax inner loop(s)
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer << "for (size_t i" << d << " = 0; i" << d << " < " << shape[d]
                                   << "; ++i" << d << ")\n";
                            writer.block_begin();
                        }
                    }

                    writer << "out" << index << " *= d;\n";

                    // end softmax inner loop(s)
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) != axes.end())
                        {
                            writer.block_end();
                        }
                    }

                    // end softmax outer loop(s)
                    for (size_t d = 0; d < dims; ++d)
                    {
                        if (axes.find(d) == axes.end())
                        {
                            writer.block_end();
                        }
                    }
                    writer.block_end();
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Result)
            {
                if (args[0].get_name() == out[0].get_name())
                {
                    writer << "// Skipping generation for " << node->get_name() << "\n";
                    return;
                }

                writer << "reference::result<" << out[0].get_type() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "               " << out[0].get_name() << ",\n";
                writer << "               " << shape_size(node->get_shape()) << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::And)
            {
                writer << "reference::logical_and(" << args[0].get_name() << ",\n"
                       << "                       " << args[1].get_name() << ",\n"
                       << "                       " << out[0].get_name() << ",\n"
                       << "                       " << out[0].get_size() << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Or)
            {
                writer << "reference::logical_or(" << args[0].get_name() << ",\n"
                       << "                      " << args[1].get_name() << ",\n"
                       << "                      " << out[0].get_name() << ",\n"
                       << "                      " << out[0].get_size() << ");\n";
            }

#define TI(x) std::type_index(typeid(x))

            static std::string emit_infix_operator(const std::string& opname,
                                                   const std::vector<std::string>& args)
            {
                if (args.size() != 2)
                {
                    throw ngraph_error("args must be equal to 2");
                }
                return args.at(0) + " " + opname + " " + args.at(1);
            }

            static std::string emit_prefix_operator(const std::string& opname,
                                                    const std::vector<std::string>& args)
            {
                if (args.size() != 1)
                {
                    throw ngraph_error("args must be equal to 2");
                }
                return opname + args.at(0);
            }

            static std::string emit_function_call(const std::string& opname,
                                                  const std::vector<std::string>& args)
            {
                return opname + "(" + join(args) + ")";
            }

            static std::unordered_map<std::type_index,
                                      std::function<std::string(const std::vector<std::string>&)>>
                initialize_inline_emitters()
            {
                auto abse =
                    std::bind(emit_function_call, std::string("std::abs"), std::placeholders::_1);
                auto mine =
                    std::bind(emit_function_call, std::string("std::min"), std::placeholders::_1);
                auto maxe =
                    std::bind(emit_function_call, std::string("std::max"), std::placeholders::_1);
                auto adde = std::bind(emit_infix_operator, std::string("+"), std::placeholders::_1);
                auto nege =
                    std::bind(emit_prefix_operator, std::string("-"), std::placeholders::_1);
                auto sube = std::bind(emit_infix_operator, std::string("-"), std::placeholders::_1);

                return std::unordered_map<
                    std::type_index,
                    std::function<std::string(const std::vector<std::string>&)>>{
                    {TI(ngraph::op::Abs), abse},
                    {TI(ngraph::op::Minimum), mine},
                    {TI(ngraph::op::Relu), maxe},
                    {TI(ngraph::op::Maximum), maxe},
                    {TI(ngraph::op::Add), adde},
                    {TI(ngraph::op::Negative), nege},
                    {TI(ngraph::op::Subtract), sube},
                };
            }

            static std::unordered_map<std::type_index,
                                      std::function<std::string(const std::vector<std::string>&)>>
                inline_emitters = initialize_inline_emitters();

            // GOEE doesn't see GOEs in subgraphs that are hidden inside LoopKernels
            // we have to manually propagate the source output
            static const ngraph::descriptor::Output*
                get_goe_input_output(ngraph::descriptor::Output* output)
            {
                auto it = output;
                while (auto goe =
                           std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(it->get_node()))
                {
                    it = &goe->get_inputs().at(goe->get_n()).get_output();
                }
                return it;
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::runtime::cpu::op::LoopKernel)
            {
                std::unordered_map<const ngraph::descriptor::Output*, std::string>
                    loop_symbol_table;
                // pre-fill symbol table with inputs

                const ngraph::runtime::cpu::op::LoopKernel* clk =
                    static_cast<const ngraph::runtime::cpu::op::LoopKernel*>(node);

                NodeVector output_nodes = clk->get_kernel_outputs();
                NodeVector node_list = clk->get_node_list();

                for (size_t i = 0; i < args.size(); i++)
                {
                    std::string sname = std::string(args[i].get_name()) + "[i]";
                    auto entry = std::make_pair(&clk->get_inputs().at(i).get_output(), sname);
                    loop_symbol_table.insert(entry);
                }

                // add outputs so we write output values directly into their
                // corresponding tensors
                for (size_t i = 0; i < out.size(); i++)
                {
                    std::string sname = std::string(out[i].get_name()) + "[i]";
                    // TODO: no support for multiple-output ops in loop kernel
                    auto entry = std::make_pair(&output_nodes.at(i)->get_outputs().at(0), sname);
                    loop_symbol_table.insert(entry);
                }

                std::string tmp_prefix{"tmp"};

                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();

                for (size_t i = 0; i < node_list.size(); i++)
                {
                    auto op_node = node_list[i];
                    auto op = &op_node->get_outputs().at(0);
                    std::string tmp;
                    if (loop_symbol_table.count(op) == 0)
                    {
                        //"allocate" a new temp
                        tmp = tmp_prefix + std::to_string(i);
                        // remember the new temp in symbol name
                        auto entry = std::make_pair(op, tmp);
                        loop_symbol_table.insert(entry);
                        // declare a new tmp
                        writer << op->get_element_type().c_type_string() << " ";
                    }
                    else
                    {
                        // this means we are dealing with an output
                        tmp = loop_symbol_table.at(op);
                    }

                    // prepare arguments
                    std::vector<std::string> sargs;
                    for (auto& input : op_node->get_inputs())
                    {
                        // args are expected to be in a map already
                        sargs.push_back(
                            loop_symbol_table.at(get_goe_input_output(&input.get_output())));
                    }

                    if (std::dynamic_pointer_cast<ngraph::op::Relu>(op_node))
                    {
                        auto casted_zero = std::string("static_cast<") +
                                           op->get_element_type().c_type_string() +
                                           std::string(">(0)");
                        sargs.push_back(casted_zero);
                    }

                    const Node& n = *op_node;
                    auto emitter = inline_emitters.at(TI(n));
                    writer << tmp << " = " << emitter(sargs) << ";\n";
                }

                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GenerateMask)
            {
                auto gm = static_cast<const ngraph::op::GenerateMask*>(node);
                writer.block_begin();
                auto index = external_function->add_state(
                    ngraph::RNGState::create_rng_state(gm->get_seed(), gm->get_probability()));
                writer << "auto state = static_cast<ngraph::RNGState*>(ctx->states[" << index
                       << "]);\n";
                writer << "bool training = static_cast<bool>(" << args[0].get_name() << "[0]);\n";
                writer << "reference::generate_mask(";
                writer << "            " << out[0].get_name() << ",\n";
                writer << "            " << out[0].get_size() << ",\n";
                writer << "            state, training);\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dequantize)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t dequantize_index =
                        mkldnn_emitter->build_dequantization(node, input_data_desc, result_desc);
                    auto& deps = mkldnn_emitter->get_primitive_deps(dequantize_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(dequantize_index) << ");\n";
                }
                else
                {
                    auto dequantize = static_cast<const ngraph::op::Dequantize*>(node);
                    writer << "reference::dequantize(";
                    writer << "            " << args[0].get_name() << ",\n";
                    writer << "            " << args[1].get_name() << ",\n";
                    writer << "            " << args[2].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(args[0].get_shape()) << "},\n";
                    writer << "            {" << join(args[1].get_shape()) << "},\n";
                    writer << "            {" << join(dequantize->get_axes()) << "});\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Quantize)
            {
                auto quantize = static_cast<const ngraph::op::Quantize*>(node);
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto scale_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(quantize->get_argument(1));
                    if (scale_const_op == nullptr)
                    {
                        throw ngraph_error("Quantize scale must be a constant");
                    }
                    auto scale = scale_const_op->get_vector<float>();

                    std::vector<float> scales;
                    scales.push_back(1.0 / scale[0]);

                    size_t quantize_index = 0;
                    quantize_index = mkldnn_emitter->build_quantize_reorder(
                        input_data_desc, result_desc, scales);
                    auto& deps = mkldnn_emitter->get_primitive_deps(quantize_index);
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[0])
                           << ", " << args[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::set_memory_ptr(ctx, " << to_string(deps[1])
                           << ", " << out[0].get_name() << ");\n";
                    writer << "cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, "
                           << to_string(quantize_index) << ");\n";
                }
                else
                {
                    writer << "reference::quantize(";
                    writer << "            " << args[0].get_name() << ",\n";
                    writer << "            " << args[1].get_name() << ",\n";
                    writer << "            " << args[2].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(args[0].get_shape()) << "},\n";
                    writer << "            {" << join(args[1].get_shape()) << "},\n";
                    writer << "            {" << join(quantize->get_axes()) << "},\n";
                    writer << "            static_cast<ngraph::op::Quantize::RoundMode>("
                           << static_cast<int>(quantize->get_round_mode()) << "));\n";
                }
            }

#undef TI
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

std::string runtime::cpu::CPU_Emitter::emit_indices(const std::vector<std::string> indices)
{
    stringstream ss;
    for (auto i : indices)
    {
        ss << "[" << i << "]";
    }
    return ss.str();
}

std::string
    runtime::cpu::CPU_Emitter::emit_for_lt(const std::string& prefix, size_t index, size_t to)
{
    stringstream ss;
    auto iv = prefix + std::to_string(index);
    ss << "for (size_t " << iv << " = 0 ; " << iv << " < " << to << "; " << iv << "++)\n";
    return ss.str();
}

std::string runtime::cpu::CPU_Emitter::emit_vector(const runtime::cpu::TensorViewWrapper& tvi,
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
