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

#pragma once

#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/generic_cpu/kernel/broadcast.hpp"
#include "ngraph/runtime/generic_cpu/kernel/dot.hpp"
#include "ngraph/runtime/generic_cpu/kernel/reshape.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/allreduce.hpp"
#include "ngraph/runtime/reference/and.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/reference/argmax.hpp"
#include "ngraph/runtime/reference/argmin.hpp"
#include "ngraph/runtime/reference/asin.hpp"
#include "ngraph/runtime/reference/atan.hpp"
#include "ngraph/runtime/reference/atan2.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/batch_mat_mul.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/broadcast_distributed.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/constant.hpp"
#include "ngraph/runtime/reference/convert.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/runtime/reference/cos.hpp"
#include "ngraph/runtime/reference/cosh.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/embedding_lookup.hpp"
#include "ngraph/runtime/reference/equal.hpp"
#include "ngraph/runtime/reference/erf.hpp"
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "ngraph/runtime/reference/generate_mask.hpp"
#include "ngraph/runtime/reference/greater.hpp"
#include "ngraph/runtime/reference/greater_eq.hpp"
#include "ngraph/runtime/reference/less.hpp"
#include "ngraph/runtime/reference/less_eq.hpp"
#include "ngraph/runtime/reference/log.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/minimum.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/not_equal.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/power.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/recv.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/scatter_add.hpp"
#include "ngraph/runtime/reference/scatter_nd_add.hpp"
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/send.hpp"
#include "ngraph/runtime/reference/shape_of.hpp"
#include "ngraph/runtime/reference/sigmoid.hpp"
#include "ngraph/runtime/reference/sign.hpp"
#include "ngraph/runtime/reference/sin.hpp"
#include "ngraph/runtime/reference/sinh.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"
#include "ngraph/runtime/reference/subtract.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/tan.hpp"
#include "ngraph/runtime/reference/tanh.hpp"
#include "ngraph/runtime/reference/topk.hpp"
#include "ngraph/runtime/reference/xor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/state/bernoulli_rng_state.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gcpu
        {
            class GCPUBackend;
            class GCPUExecutable;

            namespace
            {
                // This expands the op list in op_tbl.hpp into a list of enumerations that look like
                // this:
                // Abs,
                // Acos,
                // ...
                enum class OP_TYPEID
                {
#define NGRAPH_OP(NAME, NAMESPACE) a,
#include "ngraph/opsets/opset0_tbl.hpp"
#undef NGRAPH_OP
                    UnknownOp
                };
            }
        } // namespace gcpu
    }     // namespace runtime
} // namespace ngraph

class ngraph::runtime::gcpu::GCPUExecutable : public Executable
{
    friend class GCPUBackend;

public:
    GCPUExecutable(const std::shared_ptr<Function>& function,
                   bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& intputs) override;

    virtual void save(std::ostream& output_stream) override;

    void set_nan_check(bool enable);

    std::vector<PerformanceCounter> get_performance_data() const override;

private:
    GCPUExecutable(const std::string& model_string);

    int get_alignment() const { return 64; }
    bool m_is_compiled = false;
    bool m_nan_check_enabled = false;
    bool m_performance_counters_enabled = false;
    std::shared_ptr<Function> m_function;
    std::unordered_map<std::shared_ptr<const Node>, stopwatch> m_timer_map;
    std::vector<std::shared_ptr<Node>> m_nodes;
    std::unordered_map<const Node*, std::shared_ptr<ngraph::State>> m_states;
    std::set<std::string> m_unsupported_op_name_list;

    static OP_TYPEID get_typeid(const NodeTypeInfo& type_info);

    static void perform_nan_check(const std::vector<std::shared_ptr<HostTensor>>&,
                                  const Node* op = nullptr);

    void generate_calls(const element::Type& type,
                        const Node& op,
                        const std::vector<std::shared_ptr<HostTensor>>& outputs,
                        const std::vector<std::shared_ptr<HostTensor>>& inputs);

    template <typename T>
    void op_engine(const Node& node,
                   const std::vector<std::shared_ptr<HostTensor>>& out,
                   const std::vector<std::shared_ptr<HostTensor>>& args)
    {
// We want to check that every OP_TYPEID enumeration is included in the list.
// These GCC flags enable compile-time checking so that if an enumeration
// is not in the list an error is generated.
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
// #pragma GCC diagnostic error "-Wcovered-switch-default"
#endif
        switch (get_typeid(node.get_type_info()))
        {
        case OP_TYPEID::Abs:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::abs<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Acos:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::acos<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Add:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::add<T>(args[0]->get_data_ptr<const T>(),
                              args[1]->get_data_ptr<const T>(),
                              out[0]->get_data_ptr<T>(),
                              element_count);
            break;
        }
        case OP_TYPEID::All:
        {
            const op::All* all = static_cast<const op::All*>(&node);
            reference::all(args[0]->get_data_ptr<const char>(),
                           out[0]->get_data_ptr<char>(),
                           node.get_input_shape(0),
                           node.get_output_shape(0),
                           all->get_reduction_axes());
            break;
        }
        case OP_TYPEID::AllReduce:
        {
            const ngraph::op::AllReduce* allreduce =
                static_cast<const ngraph::op::AllReduce*>(&node);
            reference::allreduce<T>(args[0]->get_data_ptr<T>(),
                                    out[0]->get_data_ptr<T>(),
                                    node.get_input_element_type(0),
                                    allreduce->get_reduce_type(),
                                    static_cast<int>(shape_size(node.get_input_shape(0))));
            break;
        }
        case OP_TYPEID::And:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_and(args[0]->get_data_ptr<const T>(),
                                   args[1]->get_data_ptr<const T>(),
                                   out[0]->get_data_ptr<T>(),
                                   element_count);
            break;
        }
        case OP_TYPEID::Any:
        {
            const op::Any* any = static_cast<const op::Any*>(&node);
            reference::any(args[0]->get_data_ptr<const char>(),
                           out[0]->get_data_ptr<char>(),
                           node.get_input_shape(0),
                           node.get_output_shape(0),
                           any->get_reduction_axes());
            break;
        }
        case OP_TYPEID::ArgMin:
        {
            const op::ArgMin* argmin = static_cast<const op::ArgMin*>(&node);
            auto element_type = node.get_output_element_type(0);
            if (element_type == element::i64)
            {
                reference::argmin<T, int64_t>(args[0]->get_data_ptr<const T>(),
                                              out[0]->get_data_ptr<int64_t>(),
                                              node.get_input_shape(0),
                                              node.get_output_shape(0),
                                              argmin->get_reduction_axis());
            }
            else if (element_type == element::i32)
            {
                reference::argmin<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                              out[0]->get_data_ptr<int32_t>(),
                                              node.get_input_shape(0),
                                              node.get_output_shape(0),
                                              argmin->get_reduction_axis());
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::ArgMax:
        {
            const op::ArgMax* argmax = static_cast<const op::ArgMax*>(&node);
            auto element_type = node.get_output_element_type(0);
            if (element_type == element::i64)
            {
                reference::argmax<T, int64_t>(args[0]->get_data_ptr<const T>(),
                                              out[0]->get_data_ptr<int64_t>(),
                                              node.get_input_shape(0),
                                              node.get_output_shape(0),
                                              argmax->get_reduction_axis());
            }
            else if (element_type == element::i32)
            {
                reference::argmax<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                              out[0]->get_data_ptr<int32_t>(),
                                              node.get_input_shape(0),
                                              node.get_output_shape(0),
                                              argmax->get_reduction_axis());
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::Asin:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::asin<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Atan:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::atan<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Atan2:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::atan2<T>(args[0]->get_data_ptr<const T>(),
                                args[1]->get_data_ptr<const T>(),
                                out[0]->get_data_ptr<T>(),
                                element_count);
            break;
        }
        case OP_TYPEID::AvgPool:
        {
            const op::AvgPool* avg_pool = static_cast<const op::AvgPool*>(&node);

            reference::avg_pool<T>(args[0]->get_data_ptr<const T>(),
                                   out[0]->get_data_ptr<T>(),
                                   node.get_input_shape(0),
                                   node.get_output_shape(0),
                                   avg_pool->get_window_shape(),
                                   avg_pool->get_window_movement_strides(),
                                   avg_pool->get_padding_below(),
                                   avg_pool->get_padding_above(),
                                   avg_pool->get_include_padding_in_avg_computation());
            break;
        }
        case OP_TYPEID::GenerateMask:
        {
            bool use_seed = static_cast<bool>(args[2]->get_data_ptr<const int32_t>()[0]);
            if (m_states.count(&node) == 0)
            {
                const op::GenerateMask* gm = static_cast<const op::GenerateMask*>(&node);
                auto seed = use_seed ? gm->get_seed() : 0;
                m_states[&node] = std::unique_ptr<ngraph::State>(
                    new ngraph::BernoulliRNGState(seed, gm->get_probability()));
            }

            bool training = static_cast<bool>(args[0]->get_data_ptr<const T>()[0]);
            auto state = static_cast<ngraph::BernoulliRNGState*>(m_states.at(&node).get());
            size_t element_count = shape_size(node.get_output_shape(0));
            if (!use_seed)
            {
                reference::generate_mask<T>(
                    out[0]->get_data_ptr<T>(), element_count, state, training);
            }
            else
            {
                uint64_t seed = static_cast<uint64_t>(args[3]->get_data_ptr<const T>()[0]);
                double prob = static_cast<double>(args[4]->get_data_ptr<const T>()[0]);
                reference::generate_mask_no_state<T>(
                    out[0]->get_data_ptr<T>(), element_count, training, seed, prob);
            }
            break;
        }
        case OP_TYPEID::GetOutputElement:
        {
            const op::GetOutputElement* get_output_element =
                static_cast<const op::GetOutputElement*>(&node);
            size_t n = get_output_element->get_n();
            size_t element_count = shape_size(node.get_output_shape(0));
            size_t num_bytes = element_count * node.get_output_element_type(0).size();
            std::memcpy(out[0]->get_data_ptr<T>(), args[n]->get_data_ptr<T>(), num_bytes);
            break;
        }
        case OP_TYPEID::BatchMatMul:
        {
            reference::batch_mat_mul(args[0]->get_data_ptr<const T>(),
                                     args[1]->get_data_ptr<const T>(),
                                     out[0]->get_data_ptr<T>(),
                                     node.get_input_shape(0),
                                     node.get_input_shape(1),
                                     node.get_output_shape(0));
            break;
        }

        case OP_TYPEID::BatchNormTraining:
        {
            const ngraph::op::BatchNormTraining* bn =
                static_cast<const ngraph::op::BatchNormTraining*>(&node);
            reference::batch_norm_training<T>(bn->get_eps_value(),
                                              args[0]->get_data_ptr<const T>(),
                                              args[1]->get_data_ptr<const T>(),
                                              args[2]->get_data_ptr<const T>(),
                                              out[0]->get_data_ptr<T>(),
                                              out[1]->get_data_ptr<T>(),
                                              out[2]->get_data_ptr<T>(),
                                              node.get_input_shape(2));
            break;
        }
        case OP_TYPEID::BatchNormInference:
        {
            const ngraph::op::BatchNormInference* bn =
                static_cast<const ngraph::op::BatchNormInference*>(&node);
            reference::batch_norm_inference<T>(bn->get_eps_value(),
                                               args[0]->get_data_ptr<const T>(),
                                               args[1]->get_data_ptr<const T>(),
                                               args[2]->get_data_ptr<const T>(),
                                               args[3]->get_data_ptr<const T>(),
                                               args[4]->get_data_ptr<const T>(),
                                               out[0]->get_data_ptr<T>(),
                                               node.get_input_shape(2));
            break;
        }
        case OP_TYPEID::BatchNormTrainingBackprop:
        {
            const ngraph::op::BatchNormTrainingBackprop* bn_bprop =
                static_cast<const ngraph::op::BatchNormTrainingBackprop*>(&node);
            reference::batch_norm_backprop(bn_bprop->get_eps_value(),
                                           args[0]->get_data_ptr<const T>(),
                                           args[1]->get_data_ptr<const T>(),
                                           args[2]->get_data_ptr<const T>(),
                                           args[3]->get_data_ptr<const T>(),
                                           args[4]->get_data_ptr<const T>(),
                                           args[5]->get_data_ptr<const T>(),
                                           out[0]->get_data_ptr<T>(),
                                           out[1]->get_data_ptr<T>(),
                                           out[2]->get_data_ptr<T>(),
                                           node.get_input_shape(2));
            break;
        }
        case OP_TYPEID::AvgPoolBackprop:
        {
            const op::AvgPoolBackprop* apb = static_cast<const op::AvgPoolBackprop*>(&node);
            reference::avg_pool_backprop<T>(args[0]->get_data_ptr<const T>(),
                                            out[0]->get_data_ptr<T>(),
                                            node.get_input_shape(0),
                                            node.get_output_shape(0),
                                            apb->get_window_shape(),
                                            apb->get_window_movement_strides(),
                                            apb->get_padding_below(),
                                            apb->get_padding_above(),
                                            apb->get_include_padding_in_avg_computation());
            break;
        }
        case OP_TYPEID::Broadcast:
        {
            const op::Broadcast* broadcast = static_cast<const op::Broadcast*>(&node);
            Shape in_shape = node.get_input_shape(0);
            Shape out_shape = node.get_output_shape(0);
            AxisSet broadcast_axes = broadcast->get_broadcast_axes();
            kernel::broadcast<T>(args[0]->get_data_ptr<const T>(),
                                 out[0]->get_data_ptr<T>(),
                                 in_shape,
                                 out_shape,
                                 broadcast_axes);
            break;
        }
        case OP_TYPEID::BroadcastDistributed:
        {
            const ngraph::op::BroadcastDistributed* broadcast =
                static_cast<const ngraph::op::BroadcastDistributed*>(&node);
            int rank_ID;
            rank_ID = get_distributed_interface()->get_rank();
            int root_id = broadcast->get_root_id();
            if (rank_ID == root_id)
            {
                reference::broadcastdistributed<T>(
                    args[0]->get_data_ptr<T>(),
                    node.get_input_element_type(0),
                    static_cast<int>(shape_size(node.get_input_shape(0))),
                    root_id);
                auto memSize = static_cast<int>(shape_size(node.get_input_shape(0))) * sizeof(T);
                memcpy(out[0]->get_data_ptr<T>(), args[0]->get_data_ptr<T>(), memSize);
            }
            else
            {
                reference::broadcastdistributed<T>(
                    out[0]->get_data_ptr<T>(),
                    node.get_input_element_type(0),
                    static_cast<int>(shape_size(node.get_input_shape(0))),
                    root_id);
            }
            break;
        }
        case OP_TYPEID::BroadcastLike: break;
        case OP_TYPEID::Ceiling:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::ceiling<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Concat:
        {
            const op::Concat* concat = static_cast<const op::Concat*>(&node);
            std::vector<const T*> in_args;
            std::vector<Shape> in_shapes;
            for (size_t i = 0; i < node.get_input_size(); i++)
            {
                in_args.push_back(args[i]->get_data_ptr<const T>());
                in_shapes.push_back(node.get_input_shape(i));
            }
            reference::concat<T>(in_args,
                                 out[0]->get_data_ptr<T>(),
                                 in_shapes,
                                 node.get_output_shape(0),
                                 concat->get_concatenation_axis());
            break;
        }
        case OP_TYPEID::Constant:
        {
            const op::Constant* c = static_cast<const op::Constant*>(&node);
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::constant<T>(c->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::ScalarConstantLike: break;
        case OP_TYPEID::Convert:
        {
            // const op::Convert* c = static_cast<const op::Convert*>(&node);
            element::Type type = node.get_element_type();
            std::stringstream ss;
            size_t element_count = shape_size(node.get_output_shape(0));
            switch (type)
            {
            case element::Type_t::boolean:
                reference::convert_to_bool<T>(
                    args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<char>(), element_count);
                break;
            case element::Type_t::f32:
                reference::convert<T>(
                    args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<float>(), element_count);
                break;
            case element::Type_t::f64:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<double>(),
                                      element_count);
                break;
            case element::Type_t::i8:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int8_t>(),
                                      element_count);
                break;
            case element::Type_t::i16:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int16_t>(),
                                      element_count);
                break;
            case element::Type_t::i32:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int32_t>(),
                                      element_count);
                break;
            case element::Type_t::i64:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int64_t>(),
                                      element_count);
                break;
            case element::Type_t::u8:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint8_t>(),
                                      element_count);
                break;
            case element::Type_t::u16:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint16_t>(),
                                      element_count);
                break;
            case element::Type_t::u32:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint32_t>(),
                                      element_count);
                break;
            case element::Type_t::u64:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint64_t>(),
                                      element_count);
                break;
            case element::Type_t::undefined:
            case element::Type_t::dynamic:
            case element::Type_t::u1:
            case element::Type_t::bf16:
            case element::Type_t::f16:
                ss << "unsupported element type " << type << " op Convert";
                throw std::runtime_error(ss.str());
            }
            break;
        }
        case OP_TYPEID::Convolution:
        {
            const op::Convolution* c = static_cast<const op::Convolution*>(&node);
            reference::convolution<T>(args[0]->get_data_ptr<const T>(),
                                      args[1]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<T>(),
                                      node.get_input_shape(0),
                                      node.get_input_shape(1),
                                      node.get_output_shape(0),
                                      c->get_window_movement_strides(),
                                      c->get_window_dilation_strides(),
                                      c->get_padding_below(),
                                      c->get_padding_above(),
                                      c->get_data_dilation_strides());

            break;
        }
        case OP_TYPEID::ConvolutionBackpropFilters:
        {
            const op::ConvolutionBackpropFilters* c =
                static_cast<const op::ConvolutionBackpropFilters*>(&node);
            reference::convolution_backprop_filter<T>(
                args[0]->get_data_ptr<const T>(), // input
                args[1]->get_data_ptr<const T>(), // delta_convolution_output
                out[0]->get_data_ptr<T>(),        // delta_filter
                c->get_input_shape(0),            // input_shape
                c->get_input_shape(1),            // convolution_output_shape
                c->get_filters_shape(),           // filter_shape
                c->get_window_dilation_strides_forward(),
                c->get_window_movement_strides_forward(),
                c->get_padding_below_forward(),
                c->compute_backward_in_pad_above(),
                c->get_data_dilation_strides_forward());
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData:
        {
            // Note that args[1] and args[0] are switched here from the usual order.
            const op::ConvolutionBackpropData* c =
                static_cast<const op::ConvolutionBackpropData*>(&node);
            reference::convolution_backprop_in<T>(args[1]->get_data_ptr<const T>(),
                                                  args[0]->get_data_ptr<const T>(),
                                                  out[0]->get_data_ptr<T>(),
                                                  c->get_input_shape(1),
                                                  c->get_input_shape(0),
                                                  c->get_data_batch_shape(),
                                                  c->get_data_dilation_strides_forward(),
                                                  c->get_window_dilation_strides_forward(),
                                                  c->compute_backward_delta_out_pad_below(),
                                                  c->compute_backward_delta_out_pad_above(),
                                                  c->get_window_movement_strides_forward());
            break;
        }
        case OP_TYPEID::Cos:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::cos<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Cosh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::cosh<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Dequantize:
        {
            const op::Dequantize* dequantize = static_cast<const op::Dequantize*>(&node);
            auto type = dequantize->get_element_type();

            if (type == element::f32)
            {
                reference::dequantize<T>(args[0]->get_data_ptr<const T>(),
                                         args[1]->get_data_ptr<const float>(),
                                         args[2]->get_data_ptr<const T>(),
                                         out[0]->get_data_ptr<float>(),
                                         node.get_input_shape(0),
                                         node.get_input_shape(1),
                                         dequantize->get_axes());
            }
            else if (type == element::f64)
            {
                reference::dequantize<T>(args[0]->get_data_ptr<const T>(),
                                         args[1]->get_data_ptr<const double>(),
                                         args[2]->get_data_ptr<const T>(),
                                         out[0]->get_data_ptr<double>(),
                                         node.get_input_shape(0),
                                         node.get_input_shape(1),
                                         dequantize->get_axes());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op Dequantize";
                throw std::runtime_error(ss.str());
            }

            break;
        }
        case OP_TYPEID::Divide:
        {
            const op::Divide* divop = static_cast<const op::Divide*>(&node);
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::divide<T>(args[0]->get_data_ptr<const T>(),
                                 args[1]->get_data_ptr<const T>(),
                                 out[0]->get_data_ptr<T>(),
                                 element_count,
                                 divop->is_pythondiv());
            break;
        }
        case OP_TYPEID::Dot:
        {
            const op::Dot* dot = static_cast<const op::Dot*>(&node);

            kernel::dot(args[0]->get_data_ptr<const T>(),
                        args[1]->get_data_ptr<const T>(),
                        out[0]->get_data_ptr<T>(),
                        node.get_input_shape(0),
                        node.get_input_shape(1),
                        node.get_output_shape(0),
                        dot->get_reduction_axes_count());
            break;
        }
        case OP_TYPEID::DynReshape:
        {
            throw unsupported_op("Unsupported op '" + node.description() + "'");
            break;
        }
        case OP_TYPEID::DynSlice:
        {
            throw unsupported_op("Unsupported op '" + node.description() + "'");
            break;
        }
        case OP_TYPEID::EmbeddingLookup:
        {
            const op::EmbeddingLookup* embed = static_cast<const op::EmbeddingLookup*>(&node);
            auto type = embed->get_argument(0)->get_element_type();
            size_t element_count = shape_size(embed->get_argument(0)->get_shape());

            if (type == element::f32)
            {
                reference::embedding<T, float>(args[0]->get_data_ptr<const float>(),
                                               args[1]->get_data_ptr<const T>(),
                                               out[0]->get_data_ptr<T>(),
                                               element_count,
                                               embed->get_shape());
            }
            else if (type == element::f64)
            {
                reference::embedding<T, double>(args[0]->get_data_ptr<const double>(),
                                                args[1]->get_data_ptr<const T>(),
                                                out[0]->get_data_ptr<T>(),
                                                element_count,
                                                embed->get_shape());
            }
            else if (type == element::i32)
            {
                reference::embedding<T, int32_t>(args[0]->get_data_ptr<const int>(),
                                                 args[1]->get_data_ptr<const T>(),
                                                 out[0]->get_data_ptr<T>(),
                                                 element_count,
                                                 embed->get_shape());
            }
            else if (type == element::i64)
            {
                reference::embedding<T, int64_t>(args[0]->get_data_ptr<const int64_t>(),
                                                 args[1]->get_data_ptr<const T>(),
                                                 out[0]->get_data_ptr<T>(),
                                                 element_count,
                                                 embed->get_shape());
            }
            else
            {
                throw ngraph_error(std::string("Unsupported index type ") + type.c_type_string() +
                                   std::string("in EmbeddingLookup"));
            }
            break;
        }
        case OP_TYPEID::Equal:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::equal<T>(args[0]->get_data_ptr<const T>(),
                                args[1]->get_data_ptr<const T>(),
                                out[0]->get_data_ptr<char>(),
                                element_count);
            break;
        }
        case OP_TYPEID::Erf:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::erf<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Exp:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::exp<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
#ifdef INTERPRETER_USE_HYBRID
        case OP_TYPEID::FunctionCall:
        {
            auto f = static_cast<const runtime::hybrid::op::FunctionCall*>(&node);
            auto backend = f->get_backend();
            auto executable = f->get_executable();

            std::vector<std::shared_ptr<Tensor>> outputs;
            std::vector<std::shared_ptr<Tensor>> inputs;
            for (const std::shared_ptr<HostTensor>& t : out)
            {
                auto backend_tensor = backend->create_tensor(
                    t->get_element_type(), t->get_shape(), t->get_data_ptr());
                outputs.push_back(backend_tensor);
            }
            for (const std::shared_ptr<HostTensor>& t : args)
            {
                auto backend_tensor = backend->create_tensor(
                    t->get_element_type(), t->get_shape(), t->get_data_ptr());
                inputs.push_back(backend_tensor);
            }
            executable->call(outputs, inputs);
            break;
        }
#endif
        case OP_TYPEID::Floor:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::floor<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Gather:
        {
            const op::Gather* gather = static_cast<const op::Gather*>(&node);
            if (node.get_input_element_type(1) == element::i64)
            {
                reference::gather<T, int64_t>(args[0]->get_data_ptr<T>(),
                                              args[1]->get_data_ptr<int64_t>(),
                                              out[0]->get_data_ptr<T>(),
                                              node.get_input_shape(0),
                                              node.get_input_shape(1),
                                              node.get_output_shape(0),
                                              gather->get_axis());
            }
            else if (node.get_input_element_type(1) == element::i32)
            {
                reference::gather<T, int32_t>(args[0]->get_data_ptr<T>(),
                                              args[1]->get_data_ptr<int32_t>(),
                                              out[0]->get_data_ptr<T>(),
                                              node.get_input_shape(0),
                                              node.get_input_shape(1),
                                              node.get_output_shape(0),
                                              gather->get_axis());
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::GatherND:
        {
            if (node.get_input_element_type(1) == element::i64)
            {
                reference::gather_nd<T, int64_t>(args[0]->get_data_ptr<T>(),
                                                 args[1]->get_data_ptr<int64_t>(),
                                                 out[0]->get_data_ptr<T>(),
                                                 node.get_input_shape(0),
                                                 node.get_input_shape(1),
                                                 node.get_output_shape(0));
            }
            else if (node.get_input_element_type(1) == element::i32)
            {
                reference::gather_nd<T, int32_t>(args[0]->get_data_ptr<T>(),
                                                 args[1]->get_data_ptr<int32_t>(),
                                                 out[0]->get_data_ptr<T>(),
                                                 node.get_input_shape(0),
                                                 node.get_input_shape(1),
                                                 node.get_output_shape(0));
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::Greater:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::greater<T>(args[0]->get_data_ptr<const T>(),
                                  args[1]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<char>(),
                                  element_count);
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::greater_eq<T>(args[0]->get_data_ptr<const T>(),
                                     args[1]->get_data_ptr<const T>(),
                                     out[0]->get_data_ptr<char>(),
                                     element_count);
            break;
        }
        case OP_TYPEID::Less:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::less<T>(args[0]->get_data_ptr<const T>(),
                               args[1]->get_data_ptr<const T>(),
                               out[0]->get_data_ptr<char>(),
                               element_count);
            break;
        }
        case OP_TYPEID::LessEq:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::less_eq<T>(args[0]->get_data_ptr<const T>(),
                                  args[1]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<char>(),
                                  element_count);
            break;
        }
        case OP_TYPEID::Log:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::log<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::LRN:
        {
            const op::LRN* lrn = static_cast<const op::LRN*>(&node);
            reference::lrn<T>(args[0]->get_data_ptr<const T>(),
                              lrn->get_reduction_axes(),
                              out[0]->get_data_ptr<T>(),
                              node.get_input_shape(0),
                              lrn->get_alpha(),
                              lrn->get_beta(),
                              lrn->get_bias(),
                              lrn->get_nsize());
            break;
        }
        case OP_TYPEID::Max:
        {
            const op::Max* max = static_cast<const op::Max*>(&node);
            reference::max<T>(args[0]->get_data_ptr<const T>(),
                              out[0]->get_data_ptr<T>(),
                              node.get_input_shape(0),
                              node.get_output_shape(0),
                              max->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Maximum:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::maximum<T>(args[0]->get_data_ptr<const T>(),
                                  args[1]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<T>(),
                                  element_count);
            break;
        }
        case OP_TYPEID::MaxPool:
        {
            const op::MaxPool* max_pool = static_cast<const op::MaxPool*>(&node);

            reference::max_pool<T>(args[0]->get_data_ptr<const T>(),
                                   out[0]->get_data_ptr<T>(),
                                   node.get_input_shape(0),
                                   node.get_output_shape(0),
                                   max_pool->get_window_shape(),
                                   max_pool->get_window_movement_strides(),
                                   max_pool->get_padding_below(),
                                   max_pool->get_padding_above());
            break;
        }
        case OP_TYPEID::MaxPoolBackprop:
        {
            const op::MaxPoolBackprop* max_pool_backprop =
                static_cast<const op::MaxPoolBackprop*>(&node);

            reference::max_pool_backprop<T>(args[0]->get_data_ptr<const T>(),
                                            args[1]->get_data_ptr<const T>(),
                                            out[0]->get_data_ptr<T>(),
                                            node.get_input_shape(1),
                                            node.get_output_shape(0),
                                            max_pool_backprop->get_window_shape(),
                                            max_pool_backprop->get_window_movement_strides(),
                                            max_pool_backprop->get_padding_below(),
                                            max_pool_backprop->get_padding_above());
            break;
        }
        case OP_TYPEID::Min:
        {
            const op::Min* min = static_cast<const op::Min*>(&node);
            reference::min<T>(args[0]->get_data_ptr<const T>(),
                              out[0]->get_data_ptr<T>(),
                              node.get_input_shape(0),
                              node.get_output_shape(0),
                              min->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Minimum:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::minimum<T>(args[0]->get_data_ptr<const T>(),
                                  args[1]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<T>(),
                                  element_count);
            break;
        }
        case OP_TYPEID::Multiply:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::multiply<T>(args[0]->get_data_ptr<const T>(),
                                   args[1]->get_data_ptr<const T>(),
                                   out[0]->get_data_ptr<T>(),
                                   element_count);
            break;
        }
        case OP_TYPEID::Negative:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::negate<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Not:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_not(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::not_equal<T>(args[0]->get_data_ptr<const T>(),
                                    args[1]->get_data_ptr<const T>(),
                                    out[0]->get_data_ptr<char>(),
                                    element_count);
            break;
        }
        case OP_TYPEID::OneHot:
        {
            const op::OneHot* oh = static_cast<const op::OneHot*>(&node);
            reference::one_hot<T>(args[0]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<T>(),
                                  node.get_input_shape(0),
                                  node.get_output_shape(0),
                                  oh->get_one_hot_axis());
            break;
        }
        case OP_TYPEID::Or:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_or(args[0]->get_data_ptr<const T>(),
                                  args[1]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<T>(),
                                  element_count);
            break;
        }
        case OP_TYPEID::Parameter: break;
        case OP_TYPEID::Passthrough:
        {
            const op::Passthrough* passthrough = static_cast<const op::Passthrough*>(&node);
            throw unsupported_op{"Unsupported operation language: " + passthrough->language()};
        }
        case OP_TYPEID::Pad:
        {
            const op::Pad* pad = static_cast<const op::Pad*>(&node);

            reference::pad(args[0]->get_data_ptr<const T>(),
                           args[1]->get_data_ptr<const T>(),
                           out[0]->get_data_ptr<T>(),
                           node.input(0).get_shape(),
                           node.output(0).get_shape(),
                           pad->get_padding_below(),
                           pad->get_padding_above(),
                           pad->get_pad_mode());
            break;
        }
        case OP_TYPEID::Power:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::power<T>(args[0]->get_data_ptr<const T>(),
                                args[1]->get_data_ptr<const T>(),
                                out[0]->get_data_ptr<T>(),
                                element_count);
            break;
        }
        case OP_TYPEID::Product:
        {
            const op::Product* product = static_cast<const op::Product*>(&node);
            reference::product<T>(args[0]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<T>(),
                                  node.get_input_shape(0),
                                  node.get_output_shape(0),
                                  product->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Quantize:
        {
            const op::Quantize* quantize = static_cast<const op::Quantize*>(&node);
            auto type = quantize->get_element_type();

            if (type == element::u8)
            {
                reference::quantize<T>(args[0]->get_data_ptr<const T>(),
                                       args[1]->get_data_ptr<const T>(),
                                       args[2]->get_data_ptr<const uint8_t>(),
                                       out[0]->get_data_ptr<uint8_t>(),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else if (type == element::i8)
            {
                reference::quantize<T>(args[0]->get_data_ptr<const T>(),
                                       args[1]->get_data_ptr<const T>(),
                                       args[2]->get_data_ptr<const int8_t>(),
                                       out[0]->get_data_ptr<int8_t>(),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else if (type == element::i32)
            {
                reference::quantize<T>(args[0]->get_data_ptr<const T>(),
                                       args[1]->get_data_ptr<const T>(),
                                       args[2]->get_data_ptr<const int32_t>(),
                                       out[0]->get_data_ptr<int32_t>(),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op Quantize";
                throw std::runtime_error(ss.str());
            }

            break;
        }

        case OP_TYPEID::QuantizedConvolution:
        {
            const op::QuantizedConvolution* qc =
                static_cast<const op::QuantizedConvolution*>(&node);

            auto input_element_type = qc->get_input_element_type(0);
            auto filter_element_type = qc->get_input_element_type(1);
            auto output_element_type = qc->get_output_element_type(0);

            if (input_element_type == element::u8 && filter_element_type == element::i8 &&
                output_element_type == element::i8)
            {
                reference::convolution<uint8_t, int8_t, int8_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const int8_t>(),
                    out[0]->get_data_ptr<int8_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const int8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int8_t>());
            }
            else if (input_element_type == element::u8 && filter_element_type == element::u8 &&
                     output_element_type == element::u8)
            {
                reference::convolution<uint8_t, uint8_t, uint8_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const uint8_t>(),
                    out[0]->get_data_ptr<uint8_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const uint8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const uint8_t>());
            }
            else if (input_element_type == element::u8 && filter_element_type == element::i8 &&
                     output_element_type == element::i32)
            {
                reference::convolution<uint8_t, int8_t, int32_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const int8_t>(),
                    out[0]->get_data_ptr<int32_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const int8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int32_t>());
            }
            else if (input_element_type == element::u8 && filter_element_type == element::u8 &&
                     output_element_type == element::i32)
            {
                reference::convolution<uint8_t, uint8_t, int32_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const uint8_t>(),
                    out[0]->get_data_ptr<int32_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const uint8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int32_t>());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type";
                throw std::runtime_error(ss.str());
            }

            break;
        }

        case OP_TYPEID::QuantizedConvolutionBias:
        case OP_TYPEID::QuantizedConvolutionBiasAdd:
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd:
        case OP_TYPEID::QuantizedConvolutionRelu:
        case OP_TYPEID::QuantizedDotBias:
        case OP_TYPEID::QuantizedDot:
        {
            throw unsupported_op("Unsupported op '" + node.description() +
                                 "' in Interpreter back end.");
        }
        case OP_TYPEID::Recv:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            size_t memSize = element_count * sizeof(T);
            const auto* op = static_cast<const ngraph::op::Recv*>(&node);
            int src_id = op->get_src_id();

            reference::recv<T>(
                args[0]->get_data_ptr<T>(), node.get_input_element_type(0), element_count, src_id);

            memcpy(out[0]->get_data_ptr<T>(), args[0]->get_data_ptr<T>(), memSize);
            break;
        }
        case OP_TYPEID::Range:
        {
            throw unsupported_op("Unsupported op '" + node.description() + "'");
            break;
        }
        case OP_TYPEID::Relu:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::relu<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::ReluBackprop:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::relu_backprop<T>(args[0]->get_data_ptr<const T>(),
                                        args[1]->get_data_ptr<const T>(),
                                        out[0]->get_data_ptr<T>(),
                                        element_count);
            break;
        }
        case OP_TYPEID::ReplaceSlice:
        {
            const op::ReplaceSlice* slice = static_cast<const op::ReplaceSlice*>(&node);
            reference::replace_slice<T>(args[0]->get_data_ptr<const T>(),
                                        args[1]->get_data_ptr<const T>(),
                                        out[0]->get_data_ptr<T>(),
                                        node.get_input_shape(1),
                                        slice->get_lower_bounds(),
                                        slice->get_upper_bounds(),
                                        slice->get_strides(),
                                        node.get_output_shape(0));
            break;
        }
        case OP_TYPEID::Reshape:
        {
            const op::Reshape* reshape = static_cast<const op::Reshape*>(&node);
            kernel::reshape(args[0]->get_data_ptr<const T>(),
                            out[0]->get_data_ptr<T>(),
                            node.get_input_shape(0),
                            reshape->get_input_order(),
                            node.get_output_shape(0));
            break;
        }
        case OP_TYPEID::Result:
        {
            const op::Result* res = static_cast<const op::Result*>(&node);
            reference::result(args[0]->get_data_ptr<const T>(),
                              out[0]->get_data_ptr<T>(),
                              shape_size(res->get_shape()));
            break;
        }
        case OP_TYPEID::Reverse:
        {
            const op::Reverse* reverse = static_cast<const op::Reverse*>(&node);
            reference::reverse(args[0]->get_data_ptr<const T>(),
                               out[0]->get_data_ptr<T>(),
                               node.get_input_shape(0),
                               node.get_output_shape(0),
                               reverse->get_reversed_axes());
            break;
        }
        case OP_TYPEID::ReverseSequence:
        {
            const op::ReverseSequence* reverse = static_cast<const op::ReverseSequence*>(&node);

            if (node.get_input_element_type(1) == element::i32)
            {
                reference::reverse_sequence<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                                        out[0]->get_data_ptr<T>(),
                                                        node.get_input_shape(0),
                                                        reverse->get_batch_axis(),
                                                        reverse->get_sequence_axis(),
                                                        args[1]->get_data_ptr<const int32_t>());
            }
            else
            {
                throw ngraph_error("only int32 indices are supported");
            }
            break;
        }
        case OP_TYPEID::ScatterAdd:
        {
            if (node.get_input_element_type(1) == element::i64)
            {
                reference::scatter_add<T, int64_t>(args[0]->get_data_ptr<T>(),
                                                   args[1]->get_data_ptr<int64_t>(),
                                                   args[2]->get_data_ptr<T>(),
                                                   out[0]->get_data_ptr<T>(),
                                                   node.get_input_shape(0),
                                                   node.get_input_shape(1),
                                                   node.get_input_shape(2),
                                                   node.get_output_shape(0));
            }
            else if (node.get_input_element_type(1) == element::i32)
            {
                reference::scatter_add<T, int32_t>(args[0]->get_data_ptr<T>(),
                                                   args[1]->get_data_ptr<int32_t>(),
                                                   args[2]->get_data_ptr<T>(),
                                                   out[0]->get_data_ptr<T>(),
                                                   node.get_input_shape(0),
                                                   node.get_input_shape(1),
                                                   node.get_input_shape(2),
                                                   node.get_output_shape(0));
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::ScatterNDAdd:
        {
            if (node.get_input_element_type(1) == element::i64)
            {
                reference::scatter_nd_add<T, int64_t>(args[0]->get_data_ptr<T>(),
                                                      args[1]->get_data_ptr<int64_t>(),
                                                      args[2]->get_data_ptr<T>(),
                                                      out[0]->get_data_ptr<T>(),
                                                      node.get_input_shape(0),
                                                      node.get_input_shape(1),
                                                      node.get_input_shape(2),
                                                      node.get_output_shape(0));
            }
            else if (node.get_input_element_type(1) == element::i32)
            {
                reference::scatter_nd_add<T, int32_t>(args[0]->get_data_ptr<T>(),
                                                      args[1]->get_data_ptr<int32_t>(),
                                                      args[2]->get_data_ptr<T>(),
                                                      out[0]->get_data_ptr<T>(),
                                                      node.get_input_shape(0),
                                                      node.get_input_shape(1),
                                                      node.get_input_shape(2),
                                                      node.get_output_shape(0));
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::Select:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::select<T>(args[0]->get_data_ptr<const char>(),
                                 args[1]->get_data_ptr<const T>(),
                                 args[2]->get_data_ptr<const T>(),
                                 out[0]->get_data_ptr<T>(),
                                 element_count);
            break;
        }
        case OP_TYPEID::Send:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            size_t memSize = element_count * sizeof(T);
            const auto* op = static_cast<const ngraph::op::Send*>(&node);
            int dest_id = op->get_dest_id();

            reference::send<T>(args[0]->get_data_ptr<const T>(),
                               node.get_input_element_type(0),
                               element_count,
                               dest_id);

            memcpy(out[0]->get_data_ptr<T>(), args[0]->get_data_ptr<T>(), memSize);
            break;
        }
        case OP_TYPEID::ShapeOf:
        {
            reference::shape_of(node.get_input_shape(0), out[0]->get_data_ptr<uint64_t>());
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sigmoid<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::SigmoidBackprop:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sigmoid_backprop<T>(args[0]->get_data_ptr<const T>(),
                                           args[1]->get_data_ptr<const T>(),
                                           out[0]->get_data_ptr<T>(),
                                           element_count);
            break;
        }
        case OP_TYPEID::Sign:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sign<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Sin:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sin<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Sinh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sinh<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Slice:
        {
            const op::Slice* slice = static_cast<const op::Slice*>(&node);
            reference::slice<T>(args[0]->get_data_ptr<const T>(),
                                out[0]->get_data_ptr<T>(),
                                node.get_input_shape(0),
                                slice->get_lower_bounds(),
                                slice->get_upper_bounds(),
                                slice->get_strides(),
                                node.get_output_shape(0));
            break;
        }
        case OP_TYPEID::Softmax:
        {
            const op::Softmax* softmax = static_cast<const op::Softmax*>(&node);
            reference::softmax<T>(args[0]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<T>(),
                                  node.get_output_shape(0),
                                  softmax->get_axes());
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sqrt<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::StopGradient: { throw unsupported_op("Unsupported op 'StopGradient'");
        }
        case OP_TYPEID::Subtract:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::subtract<T>(args[0]->get_data_ptr<const T>(),
                                   args[1]->get_data_ptr<const T>(),
                                   out[0]->get_data_ptr<T>(),
                                   element_count);
            break;
        }
        case OP_TYPEID::Sum:
        {
            const op::Sum* sum = static_cast<const op::Sum*>(&node);
            reference::sum<T>(args[0]->get_data_ptr<const T>(),
                              out[0]->get_data_ptr<T>(),
                              node.get_input_shape(0),
                              node.get_output_shape(0),
                              sum->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Tan:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::tan<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Tanh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::tanh<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::TopK:
        {
            const op::TopK* topk = static_cast<const op::TopK*>(&node);
            if (node.get_output_element_type(0) == element::i64)
            {
                reference::topk<T, int64_t>(args[0]->get_data_ptr<const T>(),
                                            out[0]->get_data_ptr<int64_t>(),
                                            out[1]->get_data_ptr<T>(),
                                            node.get_input_shape(0),
                                            node.get_output_shape(0),
                                            topk->get_top_k_axis(),
                                            topk->get_k(),
                                            topk->get_compute_max(),
                                            topk->get_sort());
            }
            else if (node.get_output_element_type(0) == element::i32)
            {
                reference::topk<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                            out[0]->get_data_ptr<int32_t>(),
                                            out[1]->get_data_ptr<T>(),
                                            node.get_input_shape(0),
                                            node.get_output_shape(0),
                                            topk->get_top_k_axis(),
                                            topk->get_k(),
                                            topk->get_compute_max(),
                                            topk->get_sort());
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::Xor:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_xor(args[0]->get_data_ptr<const T>(),
                                   args[1]->get_data_ptr<const T>(),
                                   out[0]->get_data_ptr<T>(),
                                   element_count);
            break;
        }
        case OP_TYPEID::BatchMatMulTranspose:
        case OP_TYPEID::Clamp:
        case OP_TYPEID::ConvolutionBias:
        case OP_TYPEID::ConvolutionBiasAdd:
        case OP_TYPEID::ConvolutionBiasBackpropFiltersBias:
        case OP_TYPEID::CropAndResize:
        case OP_TYPEID::CrossEntropy:
        case OP_TYPEID::CrossEntropyBackprop:
        case OP_TYPEID::DepthToSpace:
        case OP_TYPEID::DynBroadcast:
        case OP_TYPEID::DynPad:
        case OP_TYPEID::DynReplaceSlice:
        case OP_TYPEID::Elu:
        case OP_TYPEID::FakeQuantize:
        case OP_TYPEID::GroupConvolution:
        case OP_TYPEID::GRN:
        case OP_TYPEID::GRUCell:
        case OP_TYPEID::Gelu:
        case OP_TYPEID::GeluBackpropFactor:
        case OP_TYPEID::Gemm:
        case OP_TYPEID::GroupConvolutionTranspose:
        case OP_TYPEID::HardSigmoid:
        case OP_TYPEID::Interpolate:
        case OP_TYPEID::LayerNorm:
        case OP_TYPEID::LayerNormBackprop:
        case OP_TYPEID::LogSoftmax:
        case OP_TYPEID::LSTMCell:
        case OP_TYPEID::LSTMSequence:
        case OP_TYPEID::MatMul:
        case OP_TYPEID::MVN:
        case OP_TYPEID::NormalizeL2:
        case OP_TYPEID::PRelu:
        case OP_TYPEID::PartialSlice:
        case OP_TYPEID::PartialSliceBackprop:
        case OP_TYPEID::RandomUniform:
        case OP_TYPEID::Reciprocal:
        case OP_TYPEID::RNNCell:
        case OP_TYPEID::ScaleShift:
        case OP_TYPEID::Selu:
        case OP_TYPEID::ShuffleChannels:
        case OP_TYPEID::SoftmaxCrossEntropy:
        case OP_TYPEID::SoftmaxCrossEntropyBackprop:
        case OP_TYPEID::SpaceToDepth:
        case OP_TYPEID::Split:
        case OP_TYPEID::SquaredDifference:
        case OP_TYPEID::Squeeze:
        // Tensor Iterator not yet supported
        case OP_TYPEID::TensorIterator:
        case OP_TYPEID::Tile:
        case OP_TYPEID::Transpose:
        case OP_TYPEID::Unsqueeze:
        case OP_TYPEID::UnknownOp:
            throw unsupported_op("Unsupported op '" + node.description() + "'");
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
        }
    }
};
