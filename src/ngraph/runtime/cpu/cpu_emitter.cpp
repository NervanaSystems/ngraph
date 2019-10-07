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
#include "ngraph/op/broadcast_distributed.hpp"
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
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/experimental/random_uniform.hpp"
#include "ngraph/op/experimental/tile.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
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
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/sigmoid.hpp"
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
#include "ngraph/op/xor.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_emitters.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_mat_mul_transpose.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/state/bernoulli_rng_state.hpp"
#include "ngraph/state/uniform_rng_state.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

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
            static void emit_build_primitives(CPU_ExternalFunction* external_function,
                                              const ngraph::Node* node,
                                              CodeWriter& writer,
                                              size_t& index,
                                              std::vector<std::size_t>& deps)
            {
                writer << "if (ctx->first_iteration)\n";
                writer.block_begin();

                // get the string, deps, and index from the map
                writer << get<0>(external_function->get_primitive_build_tuple(node));
                writer.block_end();

                deps = get<1>(external_function->get_primitive_build_tuple(node));
                index = get<2>(external_function->get_primitive_build_tuple(node));
            }

            template <typename OP>
            static void emit_build_primitives(CPU_ExternalFunction* external_function,
                                              const ngraph::Node* node,
                                              CodeWriter& writer,
                                              size_t& index,
                                              std::vector<std::size_t>& deps,
                                              const std::vector<TensorViewWrapper>& args)
            {
                writer << "if (ctx->first_iteration)\n";
                writer.block_begin();

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto scale_index = mkldnn_emitter->get_scale_index<OP>();
                auto scales_size = shape_size(node->get_input_shape(scale_index));
                writer << "std::vector<float> dyn_scales;\n";
                if (is_same<OP, ngraph::op::QuantizedConvolution>())
                {
                    writer << "dyn_scales.push_back(*" << args[2].get_name() << " * "
                           << " * " << args[4].get_name() << " / "
                           << " * " << args[6].get_name() << ");\n";
                }
                else
                {
                    writer << "dyn_scales.assign(" << args[scale_index].get_name() << ", "
                           << args[scale_index].get_name() << " + " << std::to_string(scales_size)
                           << ");\n";
                }
                // for Quantize
                if (is_same<OP, ngraph::op::Quantize>())
                {
                    writer << "for (size_t i = 0; i < " << std::to_string(scales_size)
                           << "; i++)\n";
                    writer.block_begin();
                    writer << "dyn_scales[i] = 1.0 / dyn_scales[i];\n";
                    writer.block_end();
                }

                // QuantizedConvolutionBiasAdd and QuantizedConvolutionBiasSignedAdd
                if (is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                    is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                {
                    auto sum_scale_index = 5;
                    auto sum_scales_size = shape_size(node->get_input_shape(sum_scale_index));
                    writer << "std::vector<float> dyn_post_op_scales;\n";
                    writer << "dyn_post_op_scales.assign(" << args[sum_scale_index].get_name()
                           << ", " << args[sum_scale_index].get_name() << " + "
                           << std::to_string(sum_scales_size) << ");\n";
                }

                writer << "// quantize across first dim (mask=2^0) if dyn_scales is a "
                          "vector \n";
                writer << "const int mask = " << std::to_string(scales_size) << " == 1 ? 0 : 1;\n";

                // get the string, deps, and index from the map
                writer << get<0>(external_function->get_primitive_build_tuple(node));
                writer.block_end();

                deps = get<1>(external_function->get_primitive_build_tuple(node));
                index = get<2>(external_function->get_primitive_build_tuple(node));
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Add)
            {
                writer.block_begin();
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t add_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, add_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(add_index)
                           << ", deps, OpType::ADD);\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AllReduce)
            {
                (void)external_function;
                const ngraph::op::AllReduce* allreduce =
                    static_cast<const ngraph::op::AllReduce*>(node);
                writer << "ngraph::get_distributed_interface()->all_reduce(" << args[0].get_name()
                       << ", " << out[0].get_name() << ", "
                       << "ngraph::element::Type_t::" << args[0].get_element_type().get_type_name()
                       << ", " << out[0].get_size() << ", "
                       << "ngraph::Reduce_t::" << allreduce->get_reduce_type() << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BroadcastDistributed)
            {
                (void)external_function;
                (void)node;
                (void)out;
                writer << "ngraph::get_distributed_interface()->broadcast(" << args[0].get_name()
                       << ", "
                       << "ngraph::element::Type_t::" << args[0].get_element_type().get_type_name()
                       << ", " << args[0].get_size() << ");\n;";
            }

            static void emitCblasSgemmBatch(CodeWriter& writer,
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
            static void emitBatchMatMul(const ngraph::Node* /* node */,
                                        const Shape& shape_a,
                                        const Shape& shape_b,
                                        const Shape& shape_c,
                                        const std::vector<TensorViewWrapper>& args,
                                        const std::vector<TensorViewWrapper>& out,
                                        const bool transpose_a,
                                        const bool transpose_b,
                                        CodeWriter& writer)
            {
                writer.block_begin();

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
                                    transpose_a,
                                    transpose_b,
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
                (void)external_function;
                const ngraph::op::MatmulBias* cg = static_cast<const ngraph::op::MatmulBias*>(node);

                const Shape& arg0_shape = pad_with(cg->get_a_shape(), 1, 3); // A
                const Shape& arg1_shape = pad_with(cg->get_b_shape(), 1, 3); // B
                const Shape& arg2_shape = node->get_shape();                 // bias (C)
                const Shape& padded_result_shape = pad_with(node->get_shape(), 1, 3);
                // Step 1: dot(A,B)
                emitBatchMatMul<ngraph::op::MatmulBias>(node,
                                                        arg0_shape,
                                                        arg1_shape,
                                                        padded_result_shape,
                                                        args,
                                                        out,
                                                        cg->get_is_a_transposed(),
                                                        cg->get_is_b_transposed(),
                                                        writer);

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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchMatMul)
            {
                (void)external_function;
                const auto* cg = static_cast<const ngraph::op::BatchMatMul*>(node);
                emitBatchMatMul<ngraph::op::BatchMatMul>(node,
                                                         cg->get_input_shape(0),
                                                         cg->get_input_shape(1),
                                                         out[0].get_shape(),
                                                         args,
                                                         out,
                                                         false,
                                                         false,
                                                         writer);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchMatMulTranspose)
            {
                (void)external_function;
                const auto* cg = static_cast<const ngraph::op::BatchMatMulTranspose*>(node);
                emitBatchMatMul<ngraph::op::BatchMatMul>(node,
                                                         cg->get_input_shape(0),
                                                         cg->get_input_shape(1),
                                                         out[0].get_shape(),
                                                         args,
                                                         out,
                                                         cg->get_transpose_arg0(),
                                                         cg->get_transpose_arg1(),
                                                         writer);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Lstm)
            {
                if (args.size() != 6)
                {
                    throw ngraph_error(
                        "Lstm op doesnt have the required number of inputs to emit MKLDNN kernel");
                }

                size_t lstm_index;
                std::vector<std::size_t> deps;
                emit_build_primitives(external_function, node, writer, lstm_index, deps);

                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                       << args[0].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                       << args[1].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                       << args[2].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                       << args[3].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[4]) << ", "
                       << args[4].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[5]) << ", "
                       << args[5].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[6]) << ", "
                       << out[0].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[7]) << ", "
                       << out[1].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[8]) << ", "
                       << out[2].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[9])
                       << ", cg_ctx->mkldnn_workspaces[" << deps[10] << "]);\n";

                writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(lstm_index)
                       << ", deps, OpType::LSTM);\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Rnn)
            {
                size_t rnn_index;
                std::vector<std::size_t> deps;
                emit_build_primitives(external_function, node, writer, rnn_index, deps);

                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                       << args[0].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                       << args[1].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                       << args[2].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                       << args[3].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[4]) << ", "
                       << args[4].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[5]) << ", "
                       << args[5].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[6]) << ", "
                       << out[0].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[7]) << ", "
                       << out[1].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[8]) << ", "
                       << out[2].get_name() << ");\n";
                writer << "cg_ctx->set_memory_ptr(" << to_string(deps[9])
                       << ", cg_ctx->mkldnn_workspaces[" << deps[10] << "]);\n";

                writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(rnn_index)
                       << ", deps, OpType::RNN);\n";
            }

            template <typename T>
            void CPU_Emitter::emitBatchNorm(CPU_ExternalFunction* external_function,
                                            CodeWriter& writer,
                                            const ngraph::Node* node,
                                            const std::vector<TensorViewWrapper>& args,
                                            const std::vector<TensorViewWrapper>& out,
                                            bool /* append_relu */,
                                            bool training)
            {
                writer.block_begin();
                // define weights
                writer << "std::vector<" << args[0].get_element_type().c_type_string()
                       << ">bn_weights(2*" << args[0].get_size() << ");\n";
                writer << "memcpy(&bn_weights[0], " << args[0].get_name() << ", "
                       << args[0].get_size() * args[0].get_element_type().size() << ");\n";
                writer << "memcpy(&bn_weights[0]+" << args[0].get_size() << ", "
                       << args[1].get_name() << ", "
                       << args[1].get_size() * args[1].get_element_type().size() << ");\n";

                size_t batchnorm_index;
                std::vector<std::size_t> deps;
                emit_build_primitives(external_function, node, writer, batchnorm_index, deps);

                if (training && args.size() == 3)
                {
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1])
                           << ", bn_weights.data());\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[4]) << ", "
                           << out[2].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(batchnorm_index)
                           << ", deps, OpType::BATCHNORM3ARGS);\n";
                }
                else
                {
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[3].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[4].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3])
                           << ", bn_weights.data());\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[4]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(batchnorm_index)
                           << ", deps, OpType::BATCHNORM5ARGS);\n";
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
                writer.block_begin();
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const ngraph::op::BatchNormTrainingBackprop* batchnorm =
                        static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);

                    writer << "reference::batch_norm_backprop(" << batchnorm->get_eps_value()
                           << ",\n";
                    writer << "            " << args[0].get_name() << ",\n";
                    writer << "            " << args[1].get_name() << ",\n";
                    writer << "            " << args[2].get_name() << ",\n";
                    writer << "            " << args[3].get_name() << ",\n";
                    writer << "            " << args[4].get_name() << ",\n";
                    writer << "            " << args[5].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            " << out[1].get_name() << ",\n";
                    writer << "            " << out[2].get_name() << ",\n";
                    writer << "            {" << join(args[2].get_shape()) << "});\n";
                }
                else
                {
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

                    size_t batchnorm_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, batchnorm_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0])
                           << ", bn_weights.data());\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[3].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << args[4].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[4]) << ", "
                           << args[5].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[5]) << ", "
                           << out[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[6])
                           << ", bn_dweights.data());\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(batchnorm_index)
                           << ", deps, OpType::BATCHNORMBACKPROP);\n";

                    writer << "memcpy(" << out[1].get_name() << ", &bn_dweights[0], "
                           << args[0].get_size() * args[0].get_element_type().size() << ");\n";
                    writer << "memcpy(" << out[2].get_name() << ", &bn_dweights[0]+"
                           << args[0].get_size() << ", "
                           << args[1].get_size() * args[1].get_element_type().size() << ");\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dot)
            {
                (void)external_function;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
                writer.block_begin();
                writer << "memcpy(" << out[0].get_name() << ", " << args[0].get_name() << ", "
                       << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Abs)
            {
                (void)external_function;
                (void)node;
                writer.block_begin();
                // Some C++ implementations don't like it when we call std::abs on unsigned types,
                // so we will
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
                        for (size_t i = 0; i < args.size(); i++)
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
                    size_t concat_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, concat_index, deps);

                    size_t i;
                    for (i = 0; i < args.size(); i++)
                    {
                        writer << "cg_ctx->set_memory_ptr(" << to_string(deps[i]) << ", "
                               << args[i].get_name() << ");\n";
                    }
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[i]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(concat_index)
                           << ", deps, OpType::CONCAT);\n";
                }
                else
                {
                    auto axis =
                        (static_cast<const ngraph::op::Concat*>(node))->get_concatenation_axis();

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
                (void)external_function;
                writer.block_begin();
                bool integral_type = !node->get_element_type().is_real();
                if (integral_type)
                {
                    // Check for divide by zero for integer types only
                    size_t element_count = args[1].get_size();
                    writer << "for (size_t i=0; i<" << element_count << "; i++)\n";
                    writer.block_begin();
                    writer << "if (" << args.at(1).get_name()
                           << "[i] == 0) throw std::runtime_error(\"integer divide by zero\");\n";
                    writer.block_end();
                }
                auto divop = static_cast<const ngraph::op::Divide*>(node);
                bool pythondiv = divop->is_pythondiv();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                if (integral_type && pythondiv)
                {
                    writer << out[0].get_name() << "[i] = ((" << args[0].get_name() << "[i] % "
                           << args[1].get_name() << "[i] != 0) && (" << args[0].get_name()
                           << "[i] < 0 != " << args[1].get_name() << "[i] < 0)) ?"
                           << args[0].get_name() << "[i] / " << args[1].get_name()
                           << "[i] - 1 :" << args[0].get_name() << "[i] / " << args[1].get_name()
                           << "[i];\n";
                }
                else
                {
                    writer << out[0].get_name() << "[i] = " << args[0].get_name() << "[i] / "
                           << args[1].get_name() << "[i];\n";
                }
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Equal)
            {
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
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
                (void)external_function;
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
                    size_t lrn_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, lrn_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(lrn_index)
                           << ", deps, OpType::LRN);\n";
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
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
                (void)external_function;
                (void)node;
                auto& result_element_type = out[0].get_element_type();

                writer << "if ((void*)" << out[0].get_name() << " != (void*)" << args[0].get_name()
                       << ") \n";
                writer.block_begin();
                writer << "#pragma omp parallel for\n";
                writer << "for (size_t i = 0; i < " << out[0].get_size() << "; i++)\n";
                writer.block_begin();
                writer << out[0].get_name() << "[i] = (";
                if (result_element_type == element::boolean)
                {
                    writer << "bool";
                }
                else
                {
                    writer << result_element_type.c_type_string();
                }
                writer << ")(" << args[0].get_name() << "[i]);\n";
                writer.block_end();
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Constant)
            {
                (void)out;
                (void)args;
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
                (void)external_function;
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
                (void)external_function;
                (void)node;
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
                    size_t slice_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, slice_index, deps);

                    writer.block_begin();
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(slice_index)
                           << ", deps, OpType::SLICE);\n";
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
                (void)external_function;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
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
                writer << "           {" << join(args[1].get_shape()) << "});\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sin)
            {
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                                         CodeWriter& writer)
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
                (void)external_function;
                auto argmin = static_cast<const ngraph::op::ArgMin*>(node);
                emitArgMinArgMax(args, out, argmin->get_reduction_axis(), "argmin", writer);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ArgMax)
            {
                (void)external_function;
                auto argmax = static_cast<const ngraph::op::ArgMax*>(node);
                emitArgMinArgMax(args, out, argmax->get_reduction_axis(), "argmax", writer);
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::TopK)
            {
                (void)external_function;
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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Gather)
            {
                (void)external_function;
                auto gather = static_cast<const ngraph::op::Gather*>(node);
                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }

                writer.block_begin();
                if ((args[0].get_element_type() == element::f64 ||
                     args[0].get_element_type() == element::f32 ||
                     args[0].get_element_type() == element::u8 ||
                     args[0].get_element_type() == element::i8) &&
                    args[0].get_shape().size() <= 3 && out[0].get_shape().size() <= 5)
                {
                    writer << "cpu::kernel::gather<" << args[0].get_type() << ", "
                           << args[1].get_element_type().c_type_string() << ", "
                           << args[0].get_shape().size() << ", " << out[0].get_shape().size()
                           << ">(" << args[0].get_name() << ",\n";
                    writer << "                   " << args[1].get_name() << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   {" << join(args[0].get_shape()) << "},\n";
                    writer << "                   {" << join(args[1].get_shape()) << "},\n";
                    writer << "                   {" << join(out[0].get_shape()) << "},\n";
                    writer << "                   " << gather->get_axis() << ",\n";
                    writer << "                   0);\n";
                }
                else
                {
                    writer << "reference::gather<" << args[0].get_type() << ", "
                           << args[1].get_element_type().c_type_string() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                   " << args[1].get_name() << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   {" << join(args[0].get_shape()) << "},\n";
                    writer << "                   {" << join(args[1].get_shape()) << "},\n";
                    writer << "                   {" << join(out[0].get_shape()) << "},\n";
                    writer << "                   " << gather->get_axis() << ");\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GatherND)
            {
                (void)external_function;
                (void)node;
                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }

                writer.block_begin();
                writer << "reference::gather_nd<" << args[0].get_type() << ", "
                       << args[1].get_element_type().c_type_string() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                   " << args[1].get_name() << ",\n";
                writer << "                   " << out[0].get_name() << ",\n";
                writer << "                   {" << join(args[0].get_shape()) << "},\n";
                writer << "                   {" << join(args[1].get_shape()) << "},\n";
                writer << "                   {" << join(out[0].get_shape()) << "});\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ScatterAdd)
            {
                (void)external_function;
                (void)node;
                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }

                writer.block_begin();
                if ((args[0].get_element_type() == element::f64 ||
                     args[0].get_element_type() == element::f32 ||
                     args[0].get_element_type() == element::u8 ||
                     args[0].get_element_type() == element::i8) &&
                    args[0].get_shape().size() <= 3 && args[2].get_shape().size() <= 5)
                {
                    writer << "cpu::kernel::scatter_add<" << args[0].get_type() << ", "
                           << args[1].get_element_type().c_type_string() << ", "
                           << args[0].get_shape().size() << ", " << args[2].get_shape().size()
                           << ">(" << args[0].get_name() << ",\n";
                    writer << "                   " << args[1].get_name() << ",\n";
                    writer << "                   " << args[2].get_name() << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   {" << join(args[0].get_shape()) << "},\n";
                    writer << "                   {" << join(args[1].get_shape()) << "},\n";
                    writer << "                   {" << join(args[2].get_shape()) << "},\n";
                    writer << "                   0);\n";
                }
                else
                {
                    writer << "reference::scatter_add<" << args[0].get_type() << ", "
                           << args[1].get_element_type().c_type_string() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                   " << args[1].get_name() << ",\n";
                    writer << "                   " << args[2].get_name() << ",\n";
                    writer << "                   " << out[0].get_name() << ",\n";
                    writer << "                   {" << join(args[0].get_shape()) << "},\n";
                    writer << "                   {" << join(args[1].get_shape()) << "},\n";
                    writer << "                   {" << join(args[2].get_shape()) << "},\n";
                    writer << "                   {" << join(out[0].get_shape()) << "});\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ScatterNDAdd)
            {
                (void)external_function;
                (void)node;
                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }

                writer.block_begin();
                writer << "reference::scatter_nd_add<" << args[0].get_type() << ", "
                       << args[1].get_element_type().c_type_string() << ">(" << args[0].get_name()
                       << ",\n";
                writer << "                   " << args[1].get_name() << ",\n";
                writer << "                   " << args[2].get_name() << ",\n";
                writer << "                   " << out[0].get_name() << ",\n";
                writer << "                   {" << join(args[0].get_shape()) << "},\n";
                writer << "                   {" << join(args[1].get_shape()) << "},\n";
                writer << "                   {" << join(args[2].get_shape()) << "},\n";
                writer << "                   {" << join(out[0].get_shape()) << "});\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Power)
            {
                (void)external_function;
                (void)node;
                (void)external_function;
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
                (void)external_function;
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
                (void)external_function;
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
                (void)external_function;
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
                    writer << "throw(std::range_error(\"One-hot: non-integral value in "
                              "input\"));\n";
                    writer.block_end();

                    writer << "size_t pos = pos_raw;\n"
                           << "if (pos < " << bounds << ")\n";
                    writer.block_begin();
                    writer << "out_vector(pos, 0) = 1;\n";
                    writer.block_end();

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
                    writer << "throw(std::range_error(\"One-hot: non-integral value in "
                              "input\"));\n";
                    writer.block_end();

                    writer << "size_t pos = pos_raw;\n";
                    writer << "bool found = false;\n";

                    writer << "if (pos < " << bounds << ")\n";
                    writer.block_begin();
                    writer << "out_vector"
                           << (oh->get_one_hot_axis() == 0 ? "(pos, i)" : "(i, pos)") << " = 1;\n";
                    writer.block_end();

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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                (void)external_function;
                (void)node;
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTIONRELU);\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionRelu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::QuantizedConvolutionRelu>(
                        external_function, node, writer, conv_index, deps, args);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::QUANTIZEDCONVOLUTIONRELU);\n";
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::QuantizedConvolution>(
                        external_function, node, writer, conv_index, deps, args);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::QUANTIZEDCONVOLUTION);\n";
                }
                else
                {
                    auto convolution = static_cast<const ngraph::op::QuantizedConvolution*>(node);

                    auto arg0_shape = args[0].get_shape();
                    auto arg1_shape = args[1].get_shape();
                    auto result_shape = out[0].get_shape();

                    writer << "reference::convolution<" << args[0].get_type() << " , "
                           << args[1].get_type() << " , " << out[0].get_type() << ", int32_t>("
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
                           << join(convolution->get_data_dilation_strides()) << "}, \n";
                    writer << "                         " << args[2].get_name() << ",\n";
                    writer << "                         " << args[3].get_name() << ",\n";
                    writer << "                         " << args[4].get_name() << ",\n";
                    writer << "                         " << args[5].get_name() << ",\n";
                    writer << "                         " << args[6].get_name() << ",\n";
                    writer << "                         " << args[7].get_name() << ");\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GroupConvolution)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    // invoke group convolution
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::GROUPCONVOLUTION);\n";
                }
                else
                {
                    throw ngraph_error("unsupported parameters for GroupConvolution");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GroupConvolutionBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    // invoke group convolution
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::GROUPCONVOLUTIONBIAS);\n";
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTION);\n";
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
                           << join(convolution->get_data_dilation_strides()) << "});\n";
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTIONBACKPROPWEIGHTS);\n";
                }
                else
                {
                    writer << "reference::convolution_backprop_filter<" << out[0].get_type() << ">("
                           << args[0].get_name() << ",\n";
                    writer << "                         " << args[1].get_name() << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(arg0_shape) << "},\n";
                    writer << "                         {" << join(arg1_shape) << "},\n";
                    writer << "                         {" << join(result_shape) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_dilation_strides_forward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_movement_strides_forward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_padding_below_forward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->compute_backward_in_pad_above()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_data_dilation_strides_forward()) << "});\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::DeconvolutionBias)
            {
                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto arg2_shape = args[2].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::DECONVOLUTIONBIAS);\n";
                }
                else
                {
                    throw ngraph_error("DeconvolutionBias is only supported with MKLDNN kernel.");
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTIONBACKPROPDATA);\n";
                }
                else
                {
                    // Note that args[1] and args[0] are switched here from the usual order.
                    writer << "reference::convolution_backprop_in<" << out[0].get_type() << ">("
                           << args[1].get_name() << ",\n";
                    writer << "                         " << args[0].get_name() << ",\n";
                    writer << "                         " << out[0].get_name() << ",\n";
                    writer << "                         {" << join(arg1_shape) << "},\n";
                    writer << "                         {" << join(arg0_shape) << "},\n";
                    writer << "                         {" << join(result_shape) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_data_dilation_strides_forward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_dilation_strides_forward()) << "},\n";
                    writer << "                         {"
                           << join(convolution->compute_backward_delta_out_pad_below()) << "},\n";
                    writer << "                         {"
                           << join(convolution->compute_backward_delta_out_pad_above()) << "},\n";
                    writer << "                         {"
                           << join(convolution->get_window_movement_strides_forward()) << "});\n";
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::QuantizedConvolutionBias>(
                        external_function, node, writer, conv_index, deps, args);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::QUANTIZEDCONVOLUTIONBIAS);\n";
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::QuantizedConvolutionBiasAdd>(
                        external_function, node, writer, conv_index, deps, args);

                    writer << "if (" << out[0].get_name() << " != " << args[3].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[3].get_name() << ", "
                           << args[3].get_size() * args[3].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::QUANTIZEDCONVOLUTIONBIASADD);\n";
                }
                else
                {
                    throw ngraph_error(
                        "QuantizedConvolutionBiasAdd is only supported with MKLDNN "
                        "kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBiasSignedAdd)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::QuantizedConvolutionBiasSignedAdd>(
                        external_function, node, writer, conv_index, deps, args);

                    writer << "if (" << out[0].get_name() << " != " << args[3].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[3].get_name() << ", "
                           << args[3].get_size() * args[3].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::QUANTIZEDCONVOLUTIONBIASSIGNEDADD);\n";
                }
                else
                {
                    throw ngraph_error(
                        "QuantizedConvolutionBiasSignedAdd is only supported with MKLDNN "
                        "kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedDotBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t qip_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::QuantizedDotBias>(
                        external_function, node, writer, qip_index, deps, args);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(qip_index)
                           << ", deps, OpType::QUANTIZEDDOTBIAS);\n";
                }
                else
                {
                    throw ngraph_error("QuantizedDotBias is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedDot)
            {
                (void)external_function;
                (void)node;
                writer << "reference::dot<" << args[0].get_type() << " , " << args[1].get_type()
                       << " , " << out[0].get_type() << ", int32_t>(" << args[0].get_name()
                       << ",\n";
                writer << "            " << args[1].get_name() << ",\n";
                writer << "            " << out[0].get_name() << ",\n";
                writer << "            {" << join(args[0].get_shape()) << "},\n";
                writer << "            {" << join(args[1].get_shape()) << "},\n";
                writer << "            {" << join(out[0].get_shape()) << "},\n";
                writer << "            1,\n";
                writer << "            " << args[2].get_name() << ",\n";
                writer << "            " << args[3].get_name() << ",\n";
                writer << "            " << args[4].get_name() << ",\n";
                writer << "            " << args[5].get_name() << ",\n";
                writer << "            " << args[6].get_name() << ",\n";
                writer << "            " << args[7].get_name() << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedMatmul)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t qip_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::QuantizedMatmul>(
                        external_function, node, writer, qip_index, deps, args);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(qip_index)
                           << ", deps, OpType::QUANTIZEDMATMUL);\n";
                }
                else
                {
                    throw ngraph_error("QuantizedMatmul is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTIONBIAS);\n";
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "if (" << out[0].get_name() << " != " << args[3].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[3].get_name() << ", "
                           << args[3].get_size() * args[3].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTIONBIASADD);\n";
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "if (" << out[0].get_name() << " != " << args[2].get_name() << ")\n";
                    writer.block_begin();
                    writer << "memcpy(" << out[0].get_name() << ", " << args[2].get_name() << ", "
                           << args[2].get_size() * args[2].get_element_type().size() << ");\n";
                    writer.block_end();
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTIONADD);\n";
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
                    size_t conv_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, conv_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3]) << ", "
                           << out[1].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(conv_index)
                           << ", deps, OpType::CONVOLUTIONBIASBACKPROPWEIGHTSBIAS);\n";
                }
                else
                {
                    throw ngraph_error(
                        "ConvolutionBiasBackpropFiltersBias is only supported with MKLDNN "
                        "kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Not)
            {
                (void)external_function;
                (void)node;
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
                    size_t max_pool_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, max_pool_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(max_pool_index)
                           << ", deps, OpType::MAXPOOL);\n";
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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolWithIndices)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t max_pool_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, max_pool_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[1].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(max_pool_index)
                           << ", deps, OpType::MAXPOOLWITHINDICES);\n";
                }
                else
                {
                    throw ngraph_error("MaxPoolWithIndices isn't supported");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reverse)
            {
                (void)external_function;
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
                (void)external_function;
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
                    size_t avg_pool_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, avg_pool_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(avg_pool_index)
                           << ", deps, OpType::AVGPOOL);\n";
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
                (void)external_function;
                auto pad = static_cast<const ngraph::op::Pad*>(node);

                auto arg0_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                if (arg0_shape.size() == 4 && args[0].get_element_type() == element::f32 &&
                    pad->get_pad_mode() == ngraph::op::PadMode::CONSTANT)
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
                    std::string pad_mode_string;
                    switch (pad->get_pad_mode())
                    {
                    case ngraph::op::PadMode::CONSTANT:
                        pad_mode_string = "ngraph::op::PadMode::CONSTANT";
                        break;
                    case ngraph::op::PadMode::EDGE:
                        pad_mode_string = "ngraph::op::PadMode::EDGE";
                        break;
                    case ngraph::op::PadMode::REFLECT:
                        pad_mode_string = "ngraph::op::PadMode::REFLECT";
                        break;
                    case ngraph::op::PadMode::SYMMETRIC:
                        pad_mode_string = "ngraph::op::PadMode::SYMMETRIC";
                        break;
                    }
                    writer << "reference::pad<" << out[0].get_type() << ">(" << args[0].get_name()
                           << ",\n";
                    writer << "            " << args[1].get_name() << ",\n";
                    writer << "            " << out[0].get_name() << ",\n";
                    writer << "            {" << join(arg0_shape) << "},\n";
                    writer << "            {" << join(result_shape) << "},\n";
                    writer << "            {" << join(pad->get_padding_below()) << "},\n";
                    writer << "            {" << join(pad->get_padding_above()) << "},\n";
                    writer << "            " << pad_mode_string << ");\n";
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
                    size_t avg_pool_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, avg_pool_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(avg_pool_index)
                           << ", deps, OpType::AVGPOOLBACKPROP);\n";
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
                    size_t max_pool_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, max_pool_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3])
                           << ", cg_ctx->mkldnn_workspaces[" << deps[5] << "]);\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(deps[4])
                           << ",deps, OpType::MAXPOOLBACKPROPFORWARD);\n";

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[3])
                           << ", cg_ctx->mkldnn_workspaces[" << deps[5] << "]);\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(max_pool_index)
                           << ", deps, OpType::MAXPOOLBACKPROPBACKWARD);\n";
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
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t max_pool_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, max_pool_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << args[2].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(max_pool_index)
                           << ", deps, OpType::MAXPOOLWITHINDICESBACKPROP);\n";
                }
                else
                {
                    throw ngraph_error("MaxPoolWithIndicesBackprop isn't supported");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Product)
            {
                (void)external_function;
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
                (void)external_function;
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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Erf)
            {
                (void)external_function;
                (void)node;
                writer.block_begin();
                auto element_count = out[0].get_size();
                if (args[0].get_element_type() == element::f32 ||
                    args[0].get_element_type() == element::f64)
                {
                    writer << "cpu::kernel::erf<" << args[0].get_element_type().c_type_string()
                           << ">(" << args[0].get_name() << ", " << out[0].get_name() << ", "
                           << element_count << ", 0);\n";
                }
                else
                {
                    writer << "cpu::kernel::reference_erf<"
                           << args[0].get_element_type().c_type_string() << ">("
                           << args[0].get_name() << ", " << out[0].get_name() << ", "
                           << element_count << ");\n";
                }
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Min)
            {
                (void)external_function;
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
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t reorder_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, reorder_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(reorder_index)
                           << ", deps, OpType::CONVERTLAYOUT);\n";
                }
                else
                {
                    throw ngraph_error("ConvertLayout is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReluBackprop)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t relu_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, relu_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(relu_index)
                           << ", deps, OpType::RELUBACKPROP);\n";
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
                    size_t relu_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, relu_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(relu_index)
                           << ", deps, OpType::RELU);\n";
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
            void CPU_Emitter::EMITTER_DECL(ngraph::op::CPULeakyRelu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t leaky_relu_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, leaky_relu_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(leaky_relu_index)
                           << ", deps, OpType::LEAKYRELU);\n";
                }
                else
                {
                    auto leaky_relu_node = static_cast<const ngraph::op::CPULeakyRelu*>(node);
                    float alpha = leaky_relu_node->get_alpha();
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
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t bounded_relu_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(
                        external_function, node, writer, bounded_relu_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(bounded_relu_index)
                           << ", deps, OpType::BOUNDEDRELU);\n";
                }
                else
                {
                    auto bounded_relu_node = static_cast<const ngraph::op::BoundedRelu*>(node);
                    float alpha = bounded_relu_node->get_alpha();
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
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t sigmoid_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, sigmoid_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(sigmoid_index)
                           << ", deps, OpType::SIGMOID);\n";
                }
                else
                {
                    throw ngraph_error("Sigmoid is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidBackprop)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t sigmoid_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, sigmoid_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << args[1].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[2]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(sigmoid_index)
                           << ", deps, OpType::SIGMOIDBACKPROP);\n";
                }
                else
                {
                    throw ngraph_error("SigmoidBackprop is only supported with MKLDNN kernel.");
                }
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
                case ngraph::op::SigmoidMultiply::FunctionType::NumTypes:
                default:
                    throw ngraph_error(
                        "generate_sigmoid_mul_func input function type not supported");
                }

                NGRAPH_CHECK(!func_block.empty(), "'func_block' must not be empty");

                return func_block;
            }
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidMultiply)
            {
                (void)external_function;
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
                (void)external_function;
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
                    size_t softmax_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives(external_function, node, writer, softmax_index, deps);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(softmax_index)
                           << ", deps, OpType::SOFTMAX);\n";
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
                (void)external_function;
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
                (void)external_function;
                (void)node;
                writer << "reference::logical_and(" << args[0].get_name() << ",\n"
                       << "                       " << args[1].get_name() << ",\n"
                       << "                       " << out[0].get_name() << ",\n"
                       << "                       " << out[0].get_size() << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Or)
            {
                (void)external_function;
                (void)node;
                writer << "reference::logical_or(" << args[0].get_name() << ",\n"
                       << "                      " << args[1].get_name() << ",\n"
                       << "                      " << out[0].get_name() << ",\n"
                       << "                      " << out[0].get_size() << ");\n";
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Xor)
            {
                (void)external_function;
                (void)node;
                writer << "reference::logical_xor(" << args[0].get_name() << ",\n"
                       << "                       " << args[1].get_name() << ",\n"
                       << "                       " << out[0].get_name() << ",\n"
                       << "                       " << out[0].get_size() << ");\n";
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

            // GOEE doesn't see GOEs in subgraphs that are hidden inside CompiledKernels
            // we have to manually propagate the source output
            static const ngraph::descriptor::Output*
                get_goe_input_output(ngraph::descriptor::Output* output)
            {
                auto it = output;
                while (auto goe =
                           std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(it->get_node()))
                {
                    it = &goe->get_inputs().at(0).get_output();
                }
                return it;
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::CompiledKernel)
            {
                (void)external_function;
                std::unordered_map<const ngraph::descriptor::Output*, std::string>
                    loop_symbol_table;
                // pre-fill symbol table with inputs

                const ngraph::op::CompiledKernel* ck =
                    static_cast<const ngraph::op::CompiledKernel*>(node);

                NodeVector output_nodes = ck->get_kernel_outputs();
                NodeVector node_list = ck->get_node_list();

                for (size_t i = 0; i < args.size(); i++)
                {
                    std::string sname = std::string(args[i].get_name()) + "[i]";
                    auto entry = std::make_pair(&ck->get_inputs().at(i).get_output(), sname);
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
                    new ngraph::BernoulliRNGState(gm->get_seed(), gm->get_probability()));
                writer << "auto state = static_cast<ngraph::BernoulliRNGState*>(ctx->states["
                       << index << "]);\n";
                writer << "bool training = static_cast<bool>(" << args[0].get_name() << "[0]);\n";
                writer << "bool use_seed = static_cast<bool>(" << args[2].get_name() << "[0]);\n";

                writer << "uint64_t seed = static_cast<uint64_t>(" << args[3].get_name()
                       << "[0]);\n";
                writer << "double keep_prob = static_cast<double>(" << args[4].get_name()
                       << "[0]);\n";
                writer << "if (use_seed == false) \n";
                writer << "{\n";
                writer << "    reference::generate_mask(\n";
                writer << "                " << out[0].get_name() << ",\n";
                writer << "                " << out[0].get_size() << ",\n";
                writer << "                state, training);\n";
                writer << "}\n";
                writer << "else {\n";
                writer << "       reference::generate_mask_no_state(\n";
                writer << "           " << out[0].get_name() << ",\n";
                writer << "           " << out[0].get_size() << ",\n";
                writer << "           training, seed, keep_prob);\n";
                writer << "}\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::RandomUniform)
            {
                auto ru = static_cast<const ngraph::op::RandomUniform*>(node);
                if (args[2].get_element_type() != element::i64)
                {
                    throw ngraph_error("Unsupported index 2 element type");
                }

                writer.block_begin();
                auto index = external_function->add_state(new UniformRNGState());
                auto fixed_seed = ru->get_fixed_seed();

                writer << "auto state = static_cast<ngraph::RandomUniformRNGState*>(ctx->states["
                       << index << "]);\n";
                writer << "bool use_fixed_seed = static_cast<bool>(" << args[3].get_name()
                       << "[0]);\n";

                writer << "if (use_fixed_seed == false) \n";
                writer << "{\n";
                writer << "    reference::random_uniform<" << args[0].get_type() << ">(\n";
                writer << "                   " << out[0].get_name() << ",\n";
                writer << "                   " << args[0].get_name() << ",\n";
                writer << "                   " << args[1].get_name() << ",\n";
                writer << "                   " << out[0].get_size() << ",\n";
                writer << "                   state);\n";
                writer << "}\n";
                writer << "else {\n";
                writer << "    reference::random_uniform_with_fixed_seed<" << args[0].get_type()
                       << ">(\n";
                writer << "                   " << out[0].get_name() << ",\n";
                writer << "                   " << args[0].get_name() << ",\n";
                writer << "                   " << args[1].get_name() << ",\n";
                writer << "                   " << out[0].get_size() << ",\n";
                writer << "                   " << fixed_seed << ");\n";
                writer << "}\n";
                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dropout)
            {
                (void)external_function;
                auto dropout = static_cast<const ngraph::op::Dropout*>(node);
                size_t ncr = ngraph::runtime::cpu::executor::GetCPUExecutor().get_num_cores();

                writer.block_begin();
                writer << "bool training = static_cast<bool>(" << args[1].get_name() << "[0]);\n";
                writer << "bool use_seed = " << to_string(dropout->get_use_seed()) << ";\n";
                writer << "int32_t seed = use_seed ? " << to_string(dropout->get_seed())
                       << " : rand();\n";
                writer << "double keep_prob = static_cast<double>(" << args[4].get_name()
                       << "[0]);\n";
                writer << "size_t count = " << args[0].get_size() << ";\n";
                writer << "size_t nthr = " << to_string(ncr) << ";\n";
                // writer << "size_t nthr = "
                //        << to_string(
                //               ngraph::runtime::cpu::executor::GetCPUExecutor().get_num_cores())
                //        << ";\n";
                writer << "size_t chunk_size = (count + nthr - 1) / nthr;\n";
                writer << "std::vector<std::minstd_rand> vmsr(nthr);\n";
                writer << "for (size_t i = 0; i < nthr; i++)\n\
                {\n\
                    std::minstd_rand msr;\n\
                    msr.seed(seed+i);\n\
                    vmsr[i] = msr;\n\
                }\n";

                writer << "double dropout_prob = 1 - keep_prob;\n";
                writer << "std::uniform_real_distribution<> gen(0, 1);\n";
                writer << "#pragma omp parallel num_threads(nthr)\n";
                writer << "{\n";
                writer << "size_t tid = omp_get_thread_num();\n";
                writer << "std::minstd_rand msr;\n msr.seed(seed+tid);\n";
                writer << "size_t idx_start = tid * chunk_size;\n";
                writer << "size_t idx_end = std::min(idx_start + chunk_size, count);\n";
                writer << "for (size_t i = idx_start; i < idx_end; i++)\n";
                writer << "{\n";
                writer << "    //out[i] = training ? static_cast<T>(bd(gen)) : "
                          "static_cast<float>(1);\n";
                writer << "    //out0[i] = training ? input[i] : static_cast<float>(1);\n";
                writer << "    if (static_cast<float>(gen(msr)) < dropout_prob)\n";
                writer << "    {\n";
                writer << "        " << out[0].get_name() << "[i] = 0;\n";
                writer << "        " << out[1].get_name() << "[i] = 0;\n";
                writer << "    }\n";
                writer << "    else\n";
                writer << "    {\n";
                writer << "        " << out[1].get_name() << "[i] = 1;\n";
                writer << "        " << out[0].get_name() << "[i] = " << args[0].get_name()
                       << "[i] / static_cast<float>(keep_prob);\n";
                writer << "    }\n";
                writer << "}\n"; // for loop ends
                writer << "}\n"; //#pragma ends

                writer.block_end();
            }

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dequantize)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    size_t dequantize_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::Dequantize>(
                        external_function, node, writer, dequantize_index, deps, args);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(dequantize_index)
                           << ", deps, OpType::DEQUANTIZE);\n";
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
                    size_t quantize_index;
                    std::vector<std::size_t> deps;
                    emit_build_primitives<ngraph::op::Quantize>(
                        external_function, node, writer, quantize_index, deps, args);

                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[0]) << ", "
                           << args[0].get_name() << ");\n";
                    writer << "cg_ctx->set_memory_ptr(" << to_string(deps[1]) << ", "
                           << out[0].get_name() << ");\n";

                    writer << "std::vector<size_t> deps{" << join(deps) << "};\n";
                    writer << "cg_ctx->mkldnn_invoke_primitive(" << to_string(quantize_index)
                           << ", deps, OpType::QUANTIZE);\n";
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

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tile)
            {
                (void)external_function;
                (void)node;
                auto arg_shape = args[0].get_shape();
                auto arg_rank = arg_shape.size();
                auto out_shape = out[0].get_shape();
                const element::Type& et = args[0].get_element_type();

                if (arg_rank == 0)
                {
                    size_t repeats = shape_size(out_shape);

                    writer.block_begin();
                    writer << "cpu::kernel::tile_rank_0<" << et.c_type_string() << ">("
                           << args[0].get_name() << ", " << out[0].get_name() << ", "
                           << std::to_string(repeats) << ");\n";

                    writer.block_end();
                }
                else
                {
                    writer.block_begin();
                    writer << "cpu::kernel::tile<" << et.c_type_string() << ", "
                           << std::to_string(arg_rank) << ">(" << args[0].get_name() << ", "
                           << out[0].get_name() << ", {" << join(arg_shape) << "}, {"
                           << join(out_shape) << "}, 0);\n";

                    writer.block_end();
                }
            }

#undef TI
        } // namespace cpu
    }     // namespace runtime
} // namespace ngraph

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
