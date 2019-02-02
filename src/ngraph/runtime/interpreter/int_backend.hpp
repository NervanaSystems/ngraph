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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/interpreter/node_wrapper.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/and.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/reference/argmax.hpp"
#include "ngraph/runtime/reference/argmin.hpp"
#include "ngraph/runtime/reference/asin.hpp"
#include "ngraph/runtime/reference/atan.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
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
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/floor.hpp"
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
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/select.hpp"
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
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/state/rng_state.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/runtime/reference/allreduce.hpp"
#endif

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INTBackend;
        }
    } // namespace runtime
} // namespace ngraph

class ngraph::runtime::interpreter::INTBackend : public Backend
{
public:
    INTBackend();
    INTBackend(const std::vector<std::string>& unsupported_op_name_list);
    INTBackend(const INTBackend&) = delete;
    INTBackend(INTBackend&&) = delete;
    INTBackend& operator=(const INTBackend&) = delete;

    std::shared_ptr<Tensor>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;

    Handle compile(std::shared_ptr<Function> function) override;

    bool call(std::shared_ptr<Function> function,
              const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& intputs) override;

    void set_nan_check(std::shared_ptr<Function> func, bool);

    void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
    std::vector<PerformanceCounter>
        get_performance_data(std::shared_ptr<Function> func) const override;

    bool is_supported(const Node& node) const override;

private:
    int get_alignment() const { return 64; }
    class FunctionInstance
    {
    public:
        bool m_is_compiled = false;
        bool m_nan_check_enabled = false;
        bool m_performance_counters_enabled = false;
        std::unordered_map<const Node*, stopwatch> m_timer_map;
        std::vector<NodeWrapper> m_wrapped_nodes;
        std::unordered_map<const Node*, std::shared_ptr<RNGState>> m_states;
    };
    std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;
    std::set<std::string> m_unsupported_op_name_list;

    static void perform_nan_check(const std::vector<std::shared_ptr<HostTensor>>&,
                                  const Node* op = nullptr);

    void generate_calls(const element::Type& type,
                        const NodeWrapper& op,
                        const std::vector<std::shared_ptr<HostTensor>>& outputs,
                        const std::vector<std::shared_ptr<HostTensor>>& inputs,
                        FunctionInstance& instance);

    template <typename T>
    void op_engine(const NodeWrapper& node_wrapper,
                   const std::vector<void*>& out,
                   const std::vector<const void*>& args,
                   FunctionInstance& instance)
    {
        const Node& node = node_wrapper.get_node();
        std::string node_op = node.description();

// We want to check that every OP_TYPEID enumeration is included in the list.
// These GCC flags enable compile-time checking so that if an enumeration
// is not in the list an error is generated.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
        // #pragma GCC diagnostic error "-Wcovered-switch-default"
        switch (node_wrapper.get_typeid())
        {
        case OP_TYPEID::Abs:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::abs<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Acos:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::acos<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Add:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::add<T>(static_cast<const T*>(args[0]),
                              static_cast<const T*>(args[1]),
                              static_cast<T*>(out[0]),
                              element_count);
            break;
        }
        case OP_TYPEID::All:
        {
            const op::All* all = static_cast<const op::All*>(&node);
            reference::all(static_cast<const char*>(args[0]),
                           static_cast<char*>(out[0]),
                           node.get_input_shape(0),
                           node.get_output_shape(0),
                           all->get_reduction_axes());
            break;
        }
        case OP_TYPEID::AllReduce: {
#ifdef NGRAPH_DISTRIBUTED
            reference::allreduce<T>(static_cast<T*>(const_cast<void*>(args[0])),
                                    static_cast<T*>(out[0]),
                                    node.get_input_element_type(0),
                                    static_cast<int>(shape_size(node.get_input_shape(0))));
#endif
            break;
        }
        case OP_TYPEID::And:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_and(static_cast<const T*>(args[0]),
                                   static_cast<const T*>(args[1]),
                                   static_cast<T*>(out[0]),
                                   element_count);
            break;
        }
        case OP_TYPEID::Any:
        {
            const op::Any* any = static_cast<const op::Any*>(&node);
            reference::any(static_cast<const char*>(args[0]),
                           static_cast<char*>(out[0]),
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
                reference::argmin<T, int64_t>(static_cast<const T*>(args[0]),
                                              static_cast<int64_t*>(out[0]),
                                              node.get_input_shape(0),
                                              node.get_output_shape(0),
                                              argmin->get_reduction_axis());
            }
            else if (element_type == element::i32)
            {
                reference::argmin<T, int32_t>(static_cast<const T*>(args[0]),
                                              static_cast<int32_t*>(out[0]),
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
                reference::argmax<T, int64_t>(static_cast<const T*>(args[0]),
                                              static_cast<int64_t*>(out[0]),
                                              node.get_input_shape(0),
                                              node.get_output_shape(0),
                                              argmax->get_reduction_axis());
            }
            else if (element_type == element::i32)
            {
                reference::argmax<T, int32_t>(static_cast<const T*>(args[0]),
                                              static_cast<int32_t*>(out[0]),
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
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Atan:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::atan<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::AvgPool:
        {
            const op::AvgPool* avg_pool = static_cast<const op::AvgPool*>(&node);

            reference::avg_pool<T>(static_cast<const T*>(args[0]),
                                   static_cast<T*>(out[0]),
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
            if (instance.m_states.count(&node) == 0)
            {
                const op::GenerateMask* gm = static_cast<const op::GenerateMask*>(&node);
                instance.m_states[&node] = std::unique_ptr<ngraph::RNGState>(
                    ngraph::RNGState::create_rng_state(gm->get_seed(), gm->get_probability()));
            }

            bool training = static_cast<bool>(static_cast<const T*>(args[0])[0]);
            auto state = instance.m_states.at(&node).get();
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::generate_mask<T>(
                reinterpret_cast<T*>(out[0]), element_count, state, training);
            break;
        }
        case OP_TYPEID::GetOutputElement:
        {
            const op::GetOutputElement* get_output_element =
                static_cast<const op::GetOutputElement*>(&node);
            size_t n = get_output_element->get_n();
            size_t element_count = shape_size(node.get_output_shape(0));
            size_t num_bytes = element_count * node.get_output_element_type(0).size();
            std::memcpy(static_cast<T*>(out[0]), args[n], num_bytes);
            break;
        }
        case OP_TYPEID::BatchNormTraining:
        {
            const ngraph::op::BatchNormTraining* bn =
                static_cast<const ngraph::op::BatchNormTraining*>(&node);
            reference::batch_norm_training<T>(bn->get_eps_value(),
                                              static_cast<const T*>(args[0]),
                                              static_cast<const T*>(args[1]),
                                              static_cast<const T*>(args[2]),
                                              static_cast<T*>(out[0]),
                                              static_cast<T*>(out[1]),
                                              static_cast<T*>(out[2]),
                                              node.get_input_shape(2));
            break;
        }
        case OP_TYPEID::BatchNormInference:
        {
            const ngraph::op::BatchNormInference* bn =
                static_cast<const ngraph::op::BatchNormInference*>(&node);
            reference::batch_norm_inference<T>(bn->get_eps_value(),
                                               static_cast<const T*>(args[0]),
                                               static_cast<const T*>(args[1]),
                                               static_cast<const T*>(args[2]),
                                               static_cast<const T*>(args[3]),
                                               static_cast<const T*>(args[4]),
                                               static_cast<T*>(out[0]),
                                               node.get_input_shape(2));
            break;
        }
        case OP_TYPEID::BatchNormTrainingBackprop:
        {
            const ngraph::op::BatchNormTrainingBackprop* bn_bprop =
                static_cast<const ngraph::op::BatchNormTrainingBackprop*>(&node);
            reference::batch_norm_backprop(bn_bprop->get_eps_value(),
                                           static_cast<const T*>(args[0]),
                                           static_cast<const T*>(args[1]),
                                           static_cast<const T*>(args[2]),
                                           static_cast<const T*>(args[3]),
                                           static_cast<const T*>(args[4]),
                                           static_cast<const T*>(args[5]),
                                           static_cast<T*>(out[0]),
                                           static_cast<T*>(out[1]),
                                           static_cast<T*>(out[2]),
                                           node.get_input_shape(2));
            break;
        }
        case OP_TYPEID::AvgPoolBackprop:
        {
            const op::AvgPoolBackprop* apb = static_cast<const op::AvgPoolBackprop*>(&node);
            reference::avg_pool_backprop<T>(static_cast<const T*>(args[0]),
                                            static_cast<T*>(out[0]),
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
            reference::broadcast<T>(static_cast<const T*>(args[0]),
                                    static_cast<T*>(out[0]),
                                    in_shape,
                                    out_shape,
                                    broadcast_axes);
            break;
        }
        case OP_TYPEID::BroadcastLike: break;
        case OP_TYPEID::Ceiling:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::ceiling<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Concat:
        {
            const op::Concat* concat = static_cast<const op::Concat*>(&node);
            std::vector<const T*> in_args;
            std::vector<Shape> in_shapes;
            for (size_t i = 0; i < node.get_input_size(); i++)
            {
                in_args.push_back(static_cast<const T*>(args[i]));
                in_shapes.push_back(node.get_input_shape(i));
            }
            reference::concat<T>(in_args,
                                 static_cast<T*>(out[0]),
                                 in_shapes,
                                 node.get_output_shape(0),
                                 concat->get_concatenation_axis());
            break;
        }
        case OP_TYPEID::Constant:
        {
            const op::Constant* c = static_cast<const op::Constant*>(&node);
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::constant<T>(c->get_data_ptr<T>(), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::ScalarConstantLike: break;
        case OP_TYPEID::Convert:
        {
            // const op::Convert* c = static_cast<const op::Convert*>(&node);
            element::Type type = node.get_element_type();
            std::stringstream ss;
            size_t element_count = shape_size(node.get_output_shape(0));
            switch (type.get_type_enum())
            {
            case element::Type_t::boolean:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<char*>(out[0]), element_count);
                break;
            case element::Type_t::f32:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<float*>(out[0]), element_count);
                break;
            case element::Type_t::f64:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<double*>(out[0]), element_count);
                break;
            case element::Type_t::i8:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<int8_t*>(out[0]), element_count);
                break;
            case element::Type_t::i16:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<int16_t*>(out[0]), element_count);
                break;
            case element::Type_t::i32:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<int32_t*>(out[0]), element_count);
                break;
            case element::Type_t::i64:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<int64_t*>(out[0]), element_count);
                break;
            case element::Type_t::u8:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<uint8_t*>(out[0]), element_count);
                break;
            case element::Type_t::u16:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<uint16_t*>(out[0]), element_count);
                break;
            case element::Type_t::u32:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<uint32_t*>(out[0]), element_count);
                break;
            case element::Type_t::u64:
                reference::convert<T>(
                    static_cast<const T*>(args[0]), static_cast<uint64_t*>(out[0]), element_count);
                break;
            case element::Type_t::undefined:
            case element::Type_t::dynamic:
            case element::Type_t::bf16:
                ss << "unsupported element type " << type << " op Convert";
                throw std::runtime_error(ss.str());
            }
            break;
        }
        case OP_TYPEID::Convolution:
        {
            const op::Convolution* c = static_cast<const op::Convolution*>(&node);
            reference::convolution<T>(static_cast<const T*>(args[0]),
                                      static_cast<const T*>(args[1]),
                                      static_cast<T*>(out[0]),
                                      node.get_input_shape(0),
                                      node.get_input_shape(1),
                                      node.get_output_shape(0),
                                      c->get_window_movement_strides(),
                                      c->get_window_dilation_strides(),
                                      c->get_padding_below(),
                                      c->get_padding_above(),
                                      c->get_data_dilation_strides(),
                                      0,
                                      1,
                                      1,
                                      0,
                                      0,
                                      1,
                                      false);
            break;
        }
        case OP_TYPEID::ConvolutionBackpropFilters:
        {
            const op::ConvolutionBackpropFilters* c =
                static_cast<const op::ConvolutionBackpropFilters*>(&node);
            reference::convolution<T>(static_cast<const T*>(args[0]),
                                      static_cast<const T*>(args[1]),
                                      static_cast<T*>(out[0]),
                                      node.get_input_shape(0),
                                      node.get_input_shape(1),
                                      node.get_output_shape(0),
                                      c->get_window_movement_strides_backward(),
                                      c->get_window_dilation_strides_backward(),
                                      c->get_padding_below_backward(),
                                      c->get_padding_above_backward(),
                                      c->get_data_dilation_strides_backward(),
                                      1,
                                      0,
                                      0,
                                      1,
                                      1,
                                      0,
                                      false);
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData:
        {
            // Note that args[1] and args[0] are switched here from the usual order.
            const op::ConvolutionBackpropData* c =
                static_cast<const op::ConvolutionBackpropData*>(&node);
            reference::convolution<T>(static_cast<const T*>(args[1]),
                                      static_cast<const T*>(args[0]),
                                      static_cast<T*>(out[0]),
                                      node.get_input_shape(1),
                                      node.get_input_shape(0),
                                      node.get_output_shape(0),
                                      c->get_window_movement_strides_backward(),
                                      c->get_window_dilation_strides_backward(),
                                      c->get_padding_below_backward(),
                                      c->get_padding_above_backward(),
                                      c->get_data_dilation_strides_backward(),
                                      0,
                                      1,
                                      0,
                                      1,
                                      0,
                                      1,
                                      true);
            break;
        }
        case OP_TYPEID::Cos:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::cos<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Cosh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::cosh<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Dequantize:
        {
            const op::Dequantize* dequantize = static_cast<const op::Dequantize*>(&node);
            auto type = dequantize->get_element_type();

            if (type == element::f32)
            {
                reference::dequantize<T>(static_cast<const T*>(args[0]),
                                         static_cast<const float*>(args[1]),
                                         static_cast<const T*>(args[2]),
                                         static_cast<float*>(out[0]),
                                         node.get_input_shape(0),
                                         node.get_input_shape(1),
                                         dequantize->get_axes());
            }
            else if (type == element::f64)
            {
                reference::dequantize<T>(static_cast<const T*>(args[0]),
                                         static_cast<const double*>(args[1]),
                                         static_cast<const T*>(args[2]),
                                         static_cast<double*>(out[0]),
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
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::divide<T>(static_cast<const T*>(args[0]),
                                 static_cast<const T*>(args[1]),
                                 static_cast<T*>(out[0]),
                                 element_count);
            break;
        }
        case OP_TYPEID::Dot:
        {
            const op::Dot* dot = static_cast<const op::Dot*>(&node);

            reference::dot(static_cast<const T*>(args[0]),
                           static_cast<const T*>(args[1]),
                           static_cast<T*>(out[0]),
                           node.get_input_shape(0),
                           node.get_input_shape(1),
                           node.get_output_shape(0),
                           dot->get_reduction_axes_count());
            break;
        }
        case OP_TYPEID::EmbeddingLookup:
        {
            const op::EmbeddingLookup* embed = static_cast<const op::EmbeddingLookup*>(&node);
            auto type = embed->get_argument(0)->get_element_type();
            size_t element_count = shape_size(embed->get_argument(0)->get_shape());

            if (type == element::f32)
            {
                reference::embedding<T, float>(static_cast<const float*>(args[0]),
                                               static_cast<const T*>(args[1]),
                                               static_cast<T*>(out[0]),
                                               element_count,
                                               embed->get_shape());
            }
            else if (type == element::f64)
            {
                reference::embedding<T, double>(static_cast<const double*>(args[0]),
                                                static_cast<const T*>(args[1]),
                                                static_cast<T*>(out[0]),
                                                element_count,
                                                embed->get_shape());
            }
            else if (type == element::i32)
            {
                reference::embedding<T, int>(static_cast<const int*>(args[0]),
                                             static_cast<const T*>(args[1]),
                                             static_cast<T*>(out[0]),
                                             element_count,
                                             embed->get_shape());
            }
            else if (type == element::i64)
            {
                reference::embedding<T, int64_t>(static_cast<const int64_t*>(args[0]),
                                                 static_cast<const T*>(args[1]),
                                                 static_cast<T*>(out[0]),
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
            reference::equal<T>(static_cast<const T*>(args[0]),
                                static_cast<const T*>(args[1]),
                                static_cast<char*>(out[0]),
                                element_count);
            break;
        }
        case OP_TYPEID::Exp:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::exp<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Floor:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::floor<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Greater:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::greater<T>(static_cast<const T*>(args[0]),
                                  static_cast<const T*>(args[1]),
                                  static_cast<char*>(out[0]),
                                  element_count);
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::greater_eq<T>(static_cast<const T*>(args[0]),
                                     static_cast<const T*>(args[1]),
                                     static_cast<char*>(out[0]),
                                     element_count);
            break;
        }
        case OP_TYPEID::Less:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::less<T>(static_cast<const T*>(args[0]),
                               static_cast<const T*>(args[1]),
                               static_cast<char*>(out[0]),
                               element_count);
            break;
        }
        case OP_TYPEID::LessEq:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::less_eq<T>(static_cast<const T*>(args[0]),
                                  static_cast<const T*>(args[1]),
                                  static_cast<char*>(out[0]),
                                  element_count);
            break;
        }
        case OP_TYPEID::Log:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::log<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::LRN:
        {
            const op::LRN* lrn = static_cast<const op::LRN*>(&node);
            reference::lrn<T>(static_cast<const T*>(args[0]),
                              static_cast<T*>(out[0]),
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
            reference::max<T>(static_cast<const T*>(args[0]),
                              static_cast<T*>(out[0]),
                              node.get_input_shape(0),
                              node.get_output_shape(0),
                              max->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Maximum:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::maximum<T>(static_cast<const T*>(args[0]),
                                  static_cast<const T*>(args[1]),
                                  static_cast<T*>(out[0]),
                                  element_count);
            break;
        }
        case OP_TYPEID::MaxPool:
        {
            const op::MaxPool* max_pool = static_cast<const op::MaxPool*>(&node);

            reference::max_pool<T>(static_cast<const T*>(args[0]),
                                   static_cast<T*>(out[0]),
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

            reference::max_pool_backprop<T>(static_cast<const T*>(args[0]),
                                            static_cast<const T*>(args[1]),
                                            static_cast<T*>(out[0]),
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
            reference::min<T>(static_cast<const T*>(args[0]),
                              static_cast<T*>(out[0]),
                              node.get_input_shape(0),
                              node.get_output_shape(0),
                              min->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Minimum:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::minimum<T>(static_cast<const T*>(args[0]),
                                  static_cast<const T*>(args[1]),
                                  static_cast<T*>(out[0]),
                                  element_count);
            break;
        }
        case OP_TYPEID::Multiply:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::multiply<T>(static_cast<const T*>(args[0]),
                                   static_cast<const T*>(args[1]),
                                   static_cast<T*>(out[0]),
                                   element_count);
            break;
        }
        case OP_TYPEID::Negative:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::negate<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Not:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_not(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::not_equal<T>(static_cast<const T*>(args[0]),
                                    static_cast<const T*>(args[1]),
                                    static_cast<char*>(out[0]),
                                    element_count);
            break;
        }
        case OP_TYPEID::OneHot:
        {
            const op::OneHot* oh = static_cast<const op::OneHot*>(&node);
            reference::one_hot<T>(static_cast<const T*>(args[0]),
                                  static_cast<T*>(out[0]),
                                  node.get_input_shape(0),
                                  node.get_output_shape(0),
                                  oh->get_one_hot_axis());
            break;
        }
        case OP_TYPEID::Or:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_or(static_cast<const T*>(args[0]),
                                  static_cast<const T*>(args[1]),
                                  static_cast<T*>(out[0]),
                                  element_count);
            break;
        }
        case OP_TYPEID::Parameter: break;
        case OP_TYPEID::Pad:
        {
            const op::Pad* pad = static_cast<const op::Pad*>(&node);

            reference::pad(static_cast<const T*>(args[0]),
                           static_cast<const T*>(args[1]),
                           static_cast<T*>(out[0]),
                           node.get_inputs().at(0).get_shape(),
                           node.get_output_shape(0),
                           pad->get_padding_below(),
                           pad->get_padding_above(),
                           pad->get_padding_interior());
            break;
        }
        case OP_TYPEID::Power:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::power<T>(static_cast<const T*>(args[0]),
                                static_cast<const T*>(args[1]),
                                static_cast<T*>(out[0]),
                                element_count);
            break;
        }
        case OP_TYPEID::Product:
        {
            const op::Product* product = static_cast<const op::Product*>(&node);
            reference::product<T>(static_cast<const T*>(args[0]),
                                  static_cast<T*>(out[0]),
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
                reference::quantize<T>(static_cast<const T*>(args[0]),
                                       static_cast<const T*>(args[1]),
                                       static_cast<const uint8_t*>(args[2]),
                                       static_cast<uint8_t*>(out[0]),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else if (type == element::i8)
            {
                reference::quantize<T>(static_cast<const T*>(args[0]),
                                       static_cast<const T*>(args[1]),
                                       static_cast<const int8_t*>(args[2]),
                                       static_cast<int8_t*>(out[0]),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else if (type == element::i32)
            {
                reference::quantize<T>(static_cast<const T*>(args[0]),
                                       static_cast<const T*>(args[1]),
                                       static_cast<const int32_t*>(args[2]),
                                       static_cast<int32_t*>(out[0]),
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
        case OP_TYPEID::QuantizedAvgPool:
        case OP_TYPEID::QuantizedConvolutionBias:
        case OP_TYPEID::QuantizedConvolutionBiasAdd:
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd:
        case OP_TYPEID::QuantizedConvolutionRelu:
        case OP_TYPEID::QuantizedConvolution:
        case OP_TYPEID::QuantizedMaxPool:
        {
            throw unsupported_op("Unsupported op '" + node.description() +
                                 "' in Interpreter back end.");
        }
        case OP_TYPEID::Relu:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::relu<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::ReluBackprop:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::relu_backprop<T>(static_cast<const T*>(args[0]),
                                        static_cast<const T*>(args[1]),
                                        static_cast<T*>(out[0]),
                                        element_count);
            break;
        }
        case OP_TYPEID::ReplaceSlice:
        {
            const op::ReplaceSlice* slice = static_cast<const op::ReplaceSlice*>(&node);
            reference::replace_slice<T>(static_cast<const T*>(args[0]),
                                        static_cast<const T*>(args[1]),
                                        static_cast<T*>(out[0]),
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
            reference::reshape(static_cast<const T*>(args[0]),
                               static_cast<T*>(out[0]),
                               node.get_input_shape(0),
                               reshape->get_input_order(),
                               node.get_output_shape(0));
            break;
        }
        case OP_TYPEID::Result:
        {
            const op::Result* res = static_cast<const op::Result*>(&node);
            reference::result(static_cast<const T*>(args[0]),
                              static_cast<T*>(out[0]),
                              shape_size(res->get_shape()));
            break;
        }
        case OP_TYPEID::Reverse:
        {
            const op::Reverse* reverse = static_cast<const op::Reverse*>(&node);
            reference::reverse(static_cast<const T*>(args[0]),
                               static_cast<T*>(out[0]),
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
                reference::reverse_sequence<T, int32_t>(static_cast<const T*>(args[0]),
                                                        static_cast<T*>(out[0]),
                                                        node.get_input_shape(0),
                                                        reverse->get_batch_axis(),
                                                        reverse->get_sequence_axis(),
                                                        static_cast<const int32_t*>(args[1]));
            }
            else
            {
                throw ngraph_error("only int32 indices are supported");
            }
            break;
        }
        case OP_TYPEID::Select:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::select<T>(static_cast<const char*>(args[0]),
                                 static_cast<const T*>(args[1]),
                                 static_cast<const T*>(args[2]),
                                 static_cast<T*>(out[0]),
                                 element_count);
            break;
        }
        case OP_TYPEID::ShapeOf:
        {
            reference::shape_of(node.get_input_shape(0), static_cast<uint64_t*>(out[0]));
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sigmoid<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::SigmoidBackprop:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sigmoid_backprop<T>(static_cast<const T*>(args[0]),
                                           static_cast<const T*>(args[1]),
                                           static_cast<T*>(out[0]),
                                           element_count);
            break;
        }
        case OP_TYPEID::Sign:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sign<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Sin:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sin<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Sinh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sinh<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Slice:
        {
            const op::Slice* slice = static_cast<const op::Slice*>(&node);
            reference::slice<T>(static_cast<const T*>(args[0]),
                                static_cast<T*>(out[0]),
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
            reference::softmax<T>(static_cast<const T*>(args[0]),
                                  static_cast<T*>(out[0]),
                                  node.get_output_shape(0),
                                  softmax->get_axes());
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sqrt<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::StopGradient: { throw unsupported_op("Unsupported op 'StopGradient'");
        }
        case OP_TYPEID::Subtract:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::subtract<T>(static_cast<const T*>(args[0]),
                                   static_cast<const T*>(args[1]),
                                   static_cast<T*>(out[0]),
                                   element_count);
            break;
        }
        case OP_TYPEID::Sum:
        {
            const op::Sum* sum = static_cast<const op::Sum*>(&node);
            reference::sum<T>(static_cast<const T*>(args[0]),
                              static_cast<T*>(out[0]),
                              node.get_input_shape(0),
                              node.get_output_shape(0),
                              sum->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Tan:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::tan<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::Tanh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::tanh<T>(
                static_cast<const T*>(args[0]), static_cast<T*>(out[0]), element_count);
            break;
        }
        case OP_TYPEID::TopK:
        {
            const op::TopK* topk = static_cast<const op::TopK*>(&node);
            if (node.get_output_element_type(0) == element::i64)
            {
                reference::topk<T, int64_t>(static_cast<const T*>(args[0]),
                                            static_cast<int64_t*>(out[0]),
                                            static_cast<T*>(out[1]),
                                            node.get_input_shape(0),
                                            node.get_output_shape(0),
                                            topk->get_top_k_axis(),
                                            topk->get_k(),
                                            topk->get_compute_max());
            }
            else if (node.get_output_element_type(0) == element::i32)
            {
                reference::topk<T, int32_t>(static_cast<const T*>(args[0]),
                                            static_cast<int32_t*>(out[0]),
                                            static_cast<T*>(out[1]),
                                            node.get_input_shape(0),
                                            node.get_output_shape(0),
                                            topk->get_top_k_axis(),
                                            topk->get_k(),
                                            topk->get_compute_max());
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        default: throw unsupported_op("Unsupported op '" + node.description() + "'");
#pragma GCC diagnostic pop
        }
    }
};
