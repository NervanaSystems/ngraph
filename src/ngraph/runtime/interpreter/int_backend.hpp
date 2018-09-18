//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/dyn_reshape.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/get_shape.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/node_wrapper.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/and.hpp"
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
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/dyn_reshape.hpp"
#include "ngraph/runtime/reference/equal.hpp"
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/get_shape.hpp"
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
#include "ngraph/runtime/reference/reduce.hpp"
#include "ngraph/runtime/reference/reduce_window.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/select_and_scatter.hpp"
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
#include "ngraph/runtime/tensor_view.hpp"

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
    }
}

class ngraph::runtime::interpreter::INTBackend : public Backend
{
public:
    std::shared_ptr<TensorView>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<TensorView> create_tensor(const element::Type& type,
                                              const Shape& shape) override;

    bool compile(std::shared_ptr<Function> function) override;

    bool call(std::shared_ptr<Function> function,
              const std::vector<std::shared_ptr<TensorView>>& outputs,
              const std::vector<std::shared_ptr<TensorView>>& intputs) override;

    void set_nan_check(std::shared_ptr<Function> func, bool);

    void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
    std::vector<PerformanceCounter>
        get_performance_data(std::shared_ptr<Function> func) const override;

private:
    class FunctionInstance
    {
    public:
        bool m_is_compiled = false;
        bool m_nan_check_enabled = false;
        bool m_performance_counters_enabled = false;
        std::unordered_map<const Node*, stopwatch> m_timer_map;
        std::vector<NodeWrapper> m_wrapped_nodes;
    };
    std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;

    static void perform_nan_check(const std::vector<std::shared_ptr<HostTensorView>>&,
                                  const Node* op = nullptr);

    void generate_calls(const element::Type& type,
                        const NodeWrapper& op,
                        const std::vector<std::shared_ptr<HostTensorView>>& outputs,
                        const std::vector<std::shared_ptr<HostTensorView>>& inputs);

    template <typename T>
    void op_engine(const NodeWrapper& node_wrapper,
                   const std::vector<std::shared_ptr<HostTensorView>>& out,
                   const std::vector<std::shared_ptr<HostTensorView>>& args)
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
            reference::abs<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Acos:
        {
            reference::acos<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Add:
        {
            reference::add<T>(args[0]->get_data_ptr<T>(),
                              args[1]->get_data_ptr<T>(),
                              out[0]->get_data_ptr<T>(),
                              out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::AllReduce: {
#ifdef NGRAPH_DISTRIBUTED
            reference::allreduce<T>(args[0]->get_data_ptr<T>(),
                                    out[0]->get_data_ptr<T>(),
                                    args[0]->get_element_type(),
                                    static_cast<int>(args[0]->get_element_count()));
#endif
            break;
        }
        case OP_TYPEID::And:
        {
            reference::logical_and(args[0]->get_data_ptr<T>(),
                                   args[1]->get_data_ptr<T>(),
                                   out[0]->get_data_ptr<T>(),
                                   out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::ArgMin:
        {
            const op::ArgMin* argmin = static_cast<const op::ArgMin*>(&node);
            if (out[0]->get_element_type() == element::i64)
            {
                reference::argmin<T, int64_t>(args[0]->get_data_ptr<T>(),
                                              out[0]->get_data_ptr<int64_t>(),
                                              args[0]->get_shape(),
                                              out[0]->get_shape(),
                                              argmin->get_reduction_axis());
            }
            else if (out[0]->get_element_type() == element::i32)
            {
                reference::argmin<T, int32_t>(args[0]->get_data_ptr<T>(),
                                              out[0]->get_data_ptr<int32_t>(),
                                              args[0]->get_shape(),
                                              out[0]->get_shape(),
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
            if (out[0]->get_element_type() == element::i64)
            {
                reference::argmax<T, int64_t>(args[0]->get_data_ptr<T>(),
                                              out[0]->get_data_ptr<int64_t>(),
                                              args[0]->get_shape(),
                                              out[0]->get_shape(),
                                              argmax->get_reduction_axis());
            }
            else if (out[0]->get_element_type() == element::i32)
            {
                reference::argmax<T, int32_t>(args[0]->get_data_ptr<T>(),
                                              out[0]->get_data_ptr<int32_t>(),
                                              args[0]->get_shape(),
                                              out[0]->get_shape(),
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
            reference::asin<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Atan:
        {
            reference::atan<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::AvgPool:
        {
            const op::AvgPool* avg_pool = static_cast<const op::AvgPool*>(&node);

            reference::avg_pool<T>(args[0]->get_data_ptr<T>(),
                                   out[0]->get_data_ptr<T>(),
                                   args[0]->get_shape(),
                                   out[0]->get_shape(),
                                   avg_pool->get_window_shape(),
                                   avg_pool->get_window_movement_strides(),
                                   avg_pool->get_padding_below(),
                                   avg_pool->get_padding_above(),
                                   avg_pool->get_include_padding_in_avg_computation());
            break;
        }
        case OP_TYPEID::GetOutputElement:
        {
            const op::GetOutputElement* get_output_element =
                static_cast<const op::GetOutputElement*>(&node);
            size_t n = get_output_element->get_n();
            size_t num_bytes = out[0]->get_element_count() * out[0]->get_element_type().size();
            std::memcpy(out[0]->get_data_ptr(), args[n]->get_data_ptr(), num_bytes);
            break;
        }
        case OP_TYPEID::BatchNorm:
        {
            const ngraph::op::BatchNorm* bn = static_cast<const ngraph::op::BatchNorm*>(&node);
            if (bn->get_output_size() == 3)
            {
                reference::batch_norm_three_outputs<T>(
                    bn->get_eps_value(),
                    reinterpret_cast<T*>(args[0]->get_data_ptr()),
                    reinterpret_cast<T*>(args[1]->get_data_ptr()),
                    reinterpret_cast<T*>(args[2]->get_data_ptr()),
                    reinterpret_cast<T*>(out[0]->get_data_ptr()),
                    reinterpret_cast<T*>(out[1]->get_data_ptr()),
                    reinterpret_cast<T*>(out[2]->get_data_ptr()),
                    args[2]->get_shape());
            }
            else
            {
                reference::batch_norm_one_output<T>(bn->get_eps_value(),
                                                    reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                                    reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                                    reinterpret_cast<T*>(args[2]->get_data_ptr()),
                                                    reinterpret_cast<T*>(args[3]->get_data_ptr()),
                                                    reinterpret_cast<T*>(args[4]->get_data_ptr()),
                                                    reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                                    args[2]->get_shape());
            }
            break;
        }
        case OP_TYPEID::BatchNormBackprop:
        {
            const ngraph::op::BatchNormBackprop* bn_bprop =
                static_cast<const ngraph::op::BatchNormBackprop*>(&node);
            reference::batch_norm_backprop(bn_bprop->get_eps_value(),
                                           reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                           reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                           reinterpret_cast<T*>(args[2]->get_data_ptr()),
                                           reinterpret_cast<T*>(args[3]->get_data_ptr()),
                                           reinterpret_cast<T*>(args[4]->get_data_ptr()),
                                           reinterpret_cast<T*>(args[5]->get_data_ptr()),
                                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                           reinterpret_cast<T*>(out[1]->get_data_ptr()),
                                           reinterpret_cast<T*>(out[2]->get_data_ptr()),
                                           args[2]->get_shape());
            break;
        }
        case OP_TYPEID::AvgPoolBackprop:
        {
            const op::AvgPoolBackprop* apb = static_cast<const op::AvgPoolBackprop*>(&node);
            reference::avg_pool_backprop<T>(args[0]->get_data_ptr<T>(),
                                            out[0]->get_data_ptr<T>(),
                                            args[0]->get_shape(),
                                            out[0]->get_shape(),
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
            Shape in_shape = args[0]->get_shape();
            Shape out_shape = out[0]->get_shape();
            AxisSet broadcast_axes = broadcast->get_broadcast_axes();
            reference::broadcast<T>(args[0]->get_data_ptr<T>(),
                                    out[0]->get_data_ptr<T>(),
                                    in_shape,
                                    out_shape,
                                    broadcast_axes);
            break;
        }
        case OP_TYPEID::Ceiling:
        {
            reference::ceiling<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Concat:
        {
            const op::Concat* concat = static_cast<const op::Concat*>(&node);
            std::vector<const T*> in_args;
            std::vector<Shape> in_shapes;
            for (std::shared_ptr<HostTensorView> arg : args)
            {
                in_args.push_back(arg->get_data_ptr<T>());
                in_shapes.push_back(arg->get_shape());
            }
            reference::concat<T>(in_args,
                                 out[0]->get_data_ptr<T>(),
                                 in_shapes,
                                 out[0]->get_shape(),
                                 concat->get_concatenation_axis());
            break;
        }
        case OP_TYPEID::Constant:
        {
            const op::Constant* c = static_cast<const op::Constant*>(&node);
            reference::constant<T>(
                c->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Convert:
        {
            // const op::Convert* c = static_cast<const op::Convert*>(&node);
            element::Type type = node.get_element_type();
            if (type == element::boolean)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<char>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::f32)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<float>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::f64)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<double>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::i8)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<int8_t>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::i16)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<int16_t>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::i32)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<int32_t>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::i64)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<int64_t>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::u8)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<uint8_t>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::u16)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<uint16_t>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::u32)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<uint32_t>(),
                                      out[0]->get_element_count());
            }
            else if (type == element::u64)
            {
                reference::convert<T>(args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<uint64_t>(),
                                      out[0]->get_element_count());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op Convert";
                throw std::runtime_error(ss.str());
            }
            break;
        }
        case OP_TYPEID::Convolution:
        {
            const op::Convolution* c = static_cast<const op::Convolution*>(&node);
            reference::convolution<T>(args[0]->get_data_ptr<T>(),
                                      args[1]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<T>(),
                                      args[0]->get_shape(),
                                      args[1]->get_shape(),
                                      out[0]->get_shape(),
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
            reference::convolution<T>(args[0]->get_data_ptr<T>(),
                                      args[1]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<T>(),
                                      args[0]->get_shape(),
                                      args[1]->get_shape(),
                                      out[0]->get_shape(),
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
            reference::convolution<T>(args[1]->get_data_ptr<T>(),
                                      args[0]->get_data_ptr<T>(),
                                      out[0]->get_data_ptr<T>(),
                                      args[1]->get_shape(),
                                      args[0]->get_shape(),
                                      out[0]->get_shape(),
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
            reference::cos<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Cosh:
        {
            reference::cosh<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Divide:
        {
            reference::divide<T>(args[0]->get_data_ptr<T>(),
                                 args[1]->get_data_ptr<T>(),
                                 out[0]->get_data_ptr<T>(),
                                 out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Dot:
        {
            const op::Dot* dot = static_cast<const op::Dot*>(&node);

            reference::dot(args[0]->get_data_ptr<T>(),
                           args[1]->get_data_ptr<T>(),
                           out[0]->get_data_ptr<T>(),
                           args[0]->get_shape(),
                           args[1]->get_shape(),
                           out[0]->get_shape(),
                           dot->get_reduction_axes_count());
            break;
        }
        case OP_TYPEID::DynReshape:
        {
            reference::dyn_reshape(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Equal:
        {
            reference::equal<T>(args[0]->get_data_ptr<T>(),
                                args[1]->get_data_ptr<T>(),
                                out[0]->get_data_ptr<char>(),
                                out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Exp:
        {
            reference::exp<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Floor:
        {
            reference::floor<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::FunctionCall:
        {
            std::shared_ptr<Function> function = node.get_functions()[0];

            std::vector<std::shared_ptr<runtime::TensorView>> outputs;
            for (auto tv : out)
            {
                outputs.push_back(std::static_pointer_cast<runtime::TensorView>(tv));
            }

            std::vector<std::shared_ptr<runtime::TensorView>> inputs;
            for (auto tv : args)
            {
                inputs.push_back(std::static_pointer_cast<runtime::TensorView>(tv));
            }

            call(function, outputs, inputs);
            break;
        }
        case OP_TYPEID::GetShape:
        {
            reference::get_shape(out[0]->get_data_ptr<uint64_t>(), args[0]->get_shape());
            break;
        }
        case OP_TYPEID::Greater:
        {
            reference::greater<T>(args[0]->get_data_ptr<T>(),
                                  args[1]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<char>(),
                                  out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            reference::greater_eq<T>(args[0]->get_data_ptr<T>(),
                                     args[1]->get_data_ptr<T>(),
                                     out[0]->get_data_ptr<char>(),
                                     out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Less:
        {
            reference::less<T>(args[0]->get_data_ptr<T>(),
                               args[1]->get_data_ptr<T>(),
                               out[0]->get_data_ptr<char>(),
                               out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::LessEq:
        {
            reference::less_eq<T>(args[0]->get_data_ptr<T>(),
                                  args[1]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<char>(),
                                  out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Log:
        {
            reference::log<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::LRN:
        {
            const op::LRN* lrn = static_cast<const op::LRN*>(&node);
            reference::lrn<T>(args[0]->get_data_ptr<T>(),
                              out[0]->get_data_ptr<T>(),
                              args[0]->get_shape(),
                              lrn->get_alpha(),
                              lrn->get_beta(),
                              lrn->get_bias(),
                              lrn->get_nsize());
            break;
        }
        case OP_TYPEID::Max:
        {
            const op::Max* max = static_cast<const op::Max*>(&node);
            reference::max<T>(args[0]->get_data_ptr<T>(),
                              out[0]->get_data_ptr<T>(),
                              args[0]->get_shape(),
                              out[0]->get_shape(),
                              max->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Maximum:
        {
            reference::maximum<T>(args[0]->get_data_ptr<T>(),
                                  args[1]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<T>(),
                                  out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::MaxPool:
        {
            const op::MaxPool* max_pool = static_cast<const op::MaxPool*>(&node);

            reference::max_pool<T>(args[0]->get_data_ptr<T>(),
                                   out[0]->get_data_ptr<T>(),
                                   args[0]->get_shape(),
                                   out[0]->get_shape(),
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

            reference::max_pool_backprop<T>(args[0]->get_data_ptr<T>(),
                                            args[1]->get_data_ptr<T>(),
                                            out[0]->get_data_ptr<T>(),
                                            args[1]->get_shape(),
                                            out[0]->get_shape(),
                                            max_pool_backprop->get_window_shape(),
                                            max_pool_backprop->get_window_movement_strides(),
                                            max_pool_backprop->get_padding_below(),
                                            max_pool_backprop->get_padding_above());
            break;
        }
        case OP_TYPEID::Min:
        {
            const op::Min* min = static_cast<const op::Min*>(&node);
            reference::min<T>(args[0]->get_data_ptr<T>(),
                              out[0]->get_data_ptr<T>(),
                              args[0]->get_shape(),
                              out[0]->get_shape(),
                              min->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Minimum:
        {
            reference::minimum<T>(args[0]->get_data_ptr<T>(),
                                  args[1]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<T>(),
                                  out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Multiply:
        {
            reference::multiply<T>(args[0]->get_data_ptr<T>(),
                                   args[1]->get_data_ptr<T>(),
                                   out[0]->get_data_ptr<T>(),
                                   out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Negative:
        {
            reference::negate<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Not:
        {
            reference::logical_not(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            reference::not_equal<T>(args[0]->get_data_ptr<T>(),
                                    args[1]->get_data_ptr<T>(),
                                    out[0]->get_data_ptr<char>(),
                                    out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::OneHot:
        {
            const op::OneHot* oh = static_cast<const op::OneHot*>(&node);
            reference::one_hot<T>(args[0]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<T>(),
                                  args[0]->get_shape(),
                                  out[0]->get_shape(),
                                  oh->get_one_hot_axis());
            break;
        }
        case OP_TYPEID::Or:
        {
            reference::logical_or(args[0]->get_data_ptr<T>(),
                                  args[1]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<T>(),
                                  out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Parameter: break;
        case OP_TYPEID::Pad:
        {
            const op::Pad* pad = static_cast<const op::Pad*>(&node);

            reference::pad(args[0]->get_data_ptr<T>(),
                           args[1]->get_data_ptr<T>(),
                           out[0]->get_data_ptr<T>(),
                           node.get_inputs().at(0).get_shape(),
                           node.get_output_shape(0),
                           pad->get_padding_below(),
                           pad->get_padding_above(),
                           pad->get_padding_interior());
            break;
        }
        case OP_TYPEID::Power:
        {
            reference::power<T>(args[0]->get_data_ptr<T>(),
                                args[1]->get_data_ptr<T>(),
                                out[0]->get_data_ptr<T>(),
                                out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Product:
        {
            const op::Product* product = static_cast<const op::Product*>(&node);
            reference::product<T>(args[0]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<T>(),
                                  args[0]->get_shape(),
                                  out[0]->get_shape(),
                                  product->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Reduce:
        {
            const op::Reduce* reduce = static_cast<const op::Reduce*>(&node);
            std::shared_ptr<Function> reduction_function = reduce->get_functions()[0];

            std::function<T(T, T)> f = [this, &node, reduction_function](T x, T y) -> T {
                auto tx = std::make_shared<HostTensorView>(
                    node.get_inputs().at(0).get_element_type(), Shape{}, "reduce_temp_x");
                auto ty = std::make_shared<HostTensorView>(
                    node.get_inputs().at(1).get_element_type(), Shape{}, "reduce_temp_y");
                auto tr = std::make_shared<HostTensorView>(
                    node.get_output_element_type(0), Shape{}, "reduce_temp_r");
                *(tx->get_data_ptr<T>()) = x;
                *(ty->get_data_ptr<T>()) = y;
                call(reduction_function, {tr}, {tx, ty});
                return *(tr->get_data_ptr<T>());
            };

            reference::reduce(args[0]->get_data_ptr<T>(),
                              args[1]->get_data_ptr<T>(),
                              out[0]->get_data_ptr<T>(),
                              node.get_inputs().at(0).get_shape(),
                              node.get_output_shape(0),
                              reduce->get_reduction_axes(),
                              f);
            break;
        }
        case OP_TYPEID::ReduceWindow:
        {
            const op::ReduceWindow* reduce_window = static_cast<const op::ReduceWindow*>(&node);
            std::shared_ptr<Function> reduction_function = reduce_window->get_functions()[0];

            std::function<T(T, T)> f = [this, &node, reduction_function](T x, T y) -> T {
                auto tx = std::make_shared<HostTensorView>(
                    node.get_inputs().at(0).get_element_type(), Shape{}, "reduce_window_temp_x");
                auto ty = std::make_shared<HostTensorView>(
                    node.get_inputs().at(1).get_element_type(), Shape{}, "reduce_window_temp_y");
                auto tr = std::make_shared<HostTensorView>(
                    node.get_output_element_type(0), Shape{}, "reduce_window_temp_r");
                *(tx->get_data_ptr<T>()) = x;
                *(ty->get_data_ptr<T>()) = y;
                call(reduction_function, {tr}, {tx, ty});
                return *(tr->get_data_ptr<T>());
            };

            reference::reduce_window(args[0]->get_data_ptr<T>(),
                                     args[1]->get_data_ptr<T>(),
                                     out[0]->get_data_ptr<T>(),
                                     node.get_inputs().at(0).get_shape(),
                                     node.get_output_shape(0),
                                     f,
                                     reduce_window->get_window_shape(),
                                     reduce_window->get_window_movement_strides());
            break;
        }
        case OP_TYPEID::Relu:
        {
            reference::relu<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::ReluBackprop:
        {
            reference::relu_backprop<T>(args[0]->get_data_ptr<T>(),
                                        args[1]->get_data_ptr<T>(),
                                        out[0]->get_data_ptr<T>(),
                                        out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::ReplaceSlice:
        {
            const op::ReplaceSlice* slice = static_cast<const op::ReplaceSlice*>(&node);
            reference::replace_slice<T>(args[0]->get_data_ptr<T>(),
                                        args[1]->get_data_ptr<T>(),
                                        out[0]->get_data_ptr<T>(),
                                        args[1]->get_shape(),
                                        slice->get_lower_bounds(),
                                        slice->get_upper_bounds(),
                                        slice->get_strides(),
                                        out[0]->get_shape());
            break;
        }
        case OP_TYPEID::Reshape:
        {
            const op::Reshape* reshape = static_cast<const op::Reshape*>(&node);
            reference::reshape(args[0]->get_data_ptr<T>(),
                               out[0]->get_data_ptr<T>(),
                               args[0]->get_shape(),
                               reshape->get_input_order(),
                               out[0]->get_shape());
            break;
        }
        case OP_TYPEID::Result:
        {
            const op::Result* res = static_cast<const op::Result*>(&node);
            reference::result(args[0]->get_data_ptr<T>(),
                              out[0]->get_data_ptr<T>(),
                              shape_size(res->get_shape()));
            break;
        }
        case OP_TYPEID::Reverse:
        {
            const op::Reverse* reverse = static_cast<const op::Reverse*>(&node);
            reference::reverse(args[0]->get_data_ptr<T>(),
                               out[0]->get_data_ptr<T>(),
                               args[0]->get_shape(),
                               out[0]->get_shape(),
                               reverse->get_reversed_axes());
            break;
        }
        case OP_TYPEID::ReverseSequence:
        {
            const op::ReverseSequence* reverse = static_cast<const op::ReverseSequence*>(&node);

            if (args[1]->get_element_type() == element::i32)
            {
                reference::reverse_sequence<T, int>(args[0]->get_data_ptr<T>(),
                                                    out[0]->get_data_ptr<T>(),
                                                    args[0]->get_shape(),
                                                    reverse->get_batch_axis(),
                                                    reverse->get_sequence_axis(),
                                                    args[1]->get_data_ptr<int>());
            }
            else
            {
                throw ngraph_error("only int32 indices are supported");
            }
            break;
        }
        case OP_TYPEID::Select:
        {
            reference::select<T>(args[0]->get_data_ptr<char>(),
                                 args[1]->get_data_ptr<T>(),
                                 args[2]->get_data_ptr<T>(),
                                 out[0]->get_data_ptr<T>(),
                                 out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::SelectAndScatter:
        {
            const ngraph::op::SelectAndScatter* select_and_scatter =
                static_cast<const ngraph::op::SelectAndScatter*>(&node);

            std::shared_ptr<ngraph::Function> selection_function =
                select_and_scatter->get_functions()[0];
            std::function<bool(T, T)> f_selection = [this, &node, selection_function](T x,
                                                                                      T y) -> bool {
                auto tx = std::make_shared<runtime::HostTensorView>(
                    node.get_inputs().at(0).get_element_type(), Shape{}, "selection_temp_x");
                auto ty = std::make_shared<runtime::HostTensorView>(
                    node.get_inputs().at(1).get_element_type(), Shape{}, "selection_temp_y");
                auto tr = std::make_shared<runtime::HostTensorView>(
                    element::boolean, Shape{}, "selection_temp_r");
                *(tx->get_data_ptr<T>()) = x;
                *(ty->get_data_ptr<T>()) = y;
                call(selection_function, {tr}, {tx, ty});
                return *(tr->get_data_ptr<char>());
            };

            std::shared_ptr<ngraph::Function> scatter_function =
                select_and_scatter->get_functions()[1];
            std::function<T(T, T)> f_scatter = [this, &node, scatter_function](T x, T y) -> T {
                auto tx = std::make_shared<runtime::HostTensorView>(
                    node.get_inputs().at(0).get_element_type(), Shape{}, "scatter_temp_x");
                auto ty = std::make_shared<runtime::HostTensorView>(
                    node.get_inputs().at(1).get_element_type(), Shape{}, "scatter_temp_y");
                auto tr = std::make_shared<runtime::HostTensorView>(
                    node.get_output_element_type(0), Shape{}, "scatter_temp_r");
                *(tx->get_data_ptr<T>()) = x;
                *(ty->get_data_ptr<T>()) = y;
                call(scatter_function, {tr}, {tx, ty});
                return *(tr->get_data_ptr<T>());
            };

            reference::select_and_scatter<T>(args[0]->get_data_ptr<T>(),
                                             args[1]->get_data_ptr<T>(),
                                             args[2]->get_data_ptr<T>(),
                                             out[0]->get_data_ptr<T>(),
                                             args[0]->get_shape(),
                                             args[1]->get_shape(),
                                             out[0]->get_shape(),
                                             f_selection,
                                             f_scatter,
                                             select_and_scatter->get_window_shape(),
                                             select_and_scatter->get_window_movement_strides());
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            reference::sigmoid<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::SigmoidBackprop:
        {
            reference::sigmoid_backprop<T>(args[0]->get_data_ptr<T>(),
                                           args[1]->get_data_ptr<T>(),
                                           out[0]->get_data_ptr<T>(),
                                           out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Sign:
        {
            reference::sign<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Sin:
        {
            reference::sin<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Sinh:
        {
            reference::sinh<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Slice:
        {
            const op::Slice* slice = static_cast<const op::Slice*>(&node);
            reference::slice<T>(args[0]->get_data_ptr<T>(),
                                out[0]->get_data_ptr<T>(),
                                args[0]->get_shape(),
                                slice->get_lower_bounds(),
                                slice->get_upper_bounds(),
                                slice->get_strides(),
                                out[0]->get_shape());
            break;
        }
        case OP_TYPEID::Softmax:
        {
            const op::Softmax* softmax = static_cast<const op::Softmax*>(&node);
            reference::softmax<T>(args[0]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<T>(),
                                  out[0]->get_shape(),
                                  softmax->get_axes());
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            reference::sqrt<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::StopGradient: { throw unsupported_op("Unsupported op 'StopGradient'");
        }
        case OP_TYPEID::Subtract:
        {
            reference::subtract<T>(args[0]->get_data_ptr<T>(),
                                   args[1]->get_data_ptr<T>(),
                                   out[0]->get_data_ptr<T>(),
                                   out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Sum:
        {
            const op::Sum* sum = static_cast<const op::Sum*>(&node);
            reference::sum<T>(args[0]->get_data_ptr<T>(),
                              out[0]->get_data_ptr<T>(),
                              args[0]->get_shape(),
                              out[0]->get_shape(),
                              sum->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Tan:
        {
            reference::tan<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::Tanh:
        {
            reference::tanh<T>(
                args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
            break;
        }
        case OP_TYPEID::TopK:
        {
            const op::TopK* topk = static_cast<const op::TopK*>(&node);
            if (out[0]->get_element_type() == element::i64)
            {
                reference::topk<T, int64_t>(args[0]->get_data_ptr<T>(),
                                            out[0]->get_data_ptr<int64_t>(),
                                            out[1]->get_data_ptr<T>(),
                                            args[0]->get_shape(),
                                            out[0]->get_shape(),
                                            topk->get_top_k_axis(),
                                            topk->get_k(),
                                            topk->get_compute_max());
            }
            else if (out[0]->get_element_type() == element::i32)
            {
                reference::topk<T, int32_t>(args[0]->get_data_ptr<T>(),
                                            out[0]->get_data_ptr<int32_t>(),
                                            out[1]->get_data_ptr<T>(),
                                            args[0]->get_shape(),
                                            out[0]->get_shape(),
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
