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

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
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
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"

#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/and.hpp"
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
#include "ngraph/runtime/reference/equal.hpp"
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/greater.hpp"
#include "ngraph/runtime/reference/greater_eq.hpp"
#include "ngraph/runtime/reference/less.hpp"
#include "ngraph/runtime/reference/less_eq.hpp"
#include "ngraph/runtime/reference/log.hpp"
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
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/select_and_scatter.hpp"
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

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/runtime/reference/allreduce.hpp"
#endif

namespace ngraph
{
    namespace runtime
    {
        namespace ie
        {
            class IE_Backend;
        }
    }
}
class ngraph::runtime::ie::IE_Backend : public Backend
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

private:
    static bool init;
    void generate_calls(const element::Type& type,
                        Node& op,
                        const std::vector<std::shared_ptr<HostTensorView>>& outputs,
                        const std::vector<std::shared_ptr<HostTensorView>>& inputs);

    template <typename T>
    void op_engine(Node& node,
                   const std::vector<std::shared_ptr<HostTensorView>>& out,
                   const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        std::string node_op = node.description();
        if (node_op == "Abs")
        {
            reference::abs<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Acos")
        {
            reference::acos<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Add")
        {
            reference::add<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(args[1]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
#ifdef NGRAPH_DISTRIBUTED
        else if (node_op == "AllReduce")
        {
            reference::allreduce<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                    reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                    args[0]->get_element_type(),
                                    static_cast<int>(args[0]->get_element_count()));
        }
#endif
        else if (node_op == "And")
        {
            reference::logical_and(reinterpret_cast<char*>(args[0]->get_data_ptr()),
                                   reinterpret_cast<char*>(args[1]->get_data_ptr()),
                                   reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                   out[0]->get_element_count());
        }
        else if (node_op == "Asin")
        {
            reference::asin<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Atan")
        {
            reference::atan<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "AvgPool")
        {
            op::AvgPool* avg_pool = dynamic_cast<op::AvgPool*>(&node);

            reference::avg_pool<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                   reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                   args[0]->get_shape(),
                                   out[0]->get_shape(),
                                   avg_pool->get_window_shape(),
                                   avg_pool->get_window_movement_strides(),
                                   avg_pool->get_padding_below(),
                                   avg_pool->get_padding_above(),
                                   avg_pool->get_include_padding_in_avg_computation());
        }
        else if (node_op == "AvgPoolBackprop")
        {
            op::AvgPoolBackprop* apb = dynamic_cast<op::AvgPoolBackprop*>(&node);
            reference::avg_pool_backprop<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                            args[0]->get_shape(),
                                            out[0]->get_shape(),
                                            apb->get_window_shape(),
                                            apb->get_window_movement_strides(),
                                            apb->get_padding_below(),
                                            apb->get_padding_above(),
                                            apb->get_include_padding_in_avg_computation());
        }
        else if (node_op == "BatchNorm")
        {
            ngraph::op::BatchNorm* bn = dynamic_cast<ngraph::op::BatchNorm*>(&node);
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
        }
        else if (node_op == "Broadcast")
        {
            op::Broadcast* broadcast = dynamic_cast<op::Broadcast*>(&node);
            Shape in_shape = args[0]->get_shape();
            Shape out_shape = out[0]->get_shape();
            AxisSet broadcast_axes = broadcast->get_broadcast_axes();
            reference::broadcast<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                    reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                    in_shape,
                                    out_shape,
                                    broadcast_axes);
        }
        else if (node_op == "Ceiling")
        {
            reference::ceiling<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                  out[0]->get_element_count());
        }
        else if (node_op == "Concat")
        {
            const op::Concat* concat = static_cast<const op::Concat*>(&node);
            std::vector<const T*> in_args;
            std::vector<Shape> in_shapes;
            for (std::shared_ptr<HostTensorView> arg : args)
            {
                in_args.push_back(reinterpret_cast<T*>(arg->get_data_ptr()));
                in_shapes.push_back(arg->get_shape());
            }
            reference::concat<T>(in_args,
                                 reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                 in_shapes,
                                 out[0]->get_shape(),
                                 concat->get_concatenation_axis());
        }
        else if (node_op == "Constant")
        {
            const op::Constant* c = static_cast<const op::Constant*>(&node);
            reference::constant<T>(reinterpret_cast<const T*>(c->get_data_ptr()),
                                   reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                   out[0]->get_element_count());
        }
        else if (node_op == "Convert")
        {
            // const op::Convert* c = static_cast<const op::Convert*>(&node);
            element::Type type = node.get_element_type();
            if (type == element::boolean)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::f32)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<float*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::f64)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<double*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::i8)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<int8_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::i16)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<int16_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::i32)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<int32_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::i64)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<int64_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::u8)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<uint8_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::u16)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<uint16_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::u32)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<uint32_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else if (type == element::u64)
            {
                reference::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<uint64_t*>(out[0]->get_data_ptr()),
                                      out[0]->get_element_count());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op Convert";
                throw std::runtime_error(ss.str());
            }
        }
        else if (node_op == "Convolution")
        {
            auto c = static_cast<const op::Convolution*>(&node);
            reference::convolution<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                      reinterpret_cast<T*>(out[0]->get_data_ptr()),
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
        }
        else if (node_op == "ConvolutionBackpropFilters")
        {
            auto c = static_cast<const op::ConvolutionBackpropFilters*>(&node);
            reference::convolution<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                      reinterpret_cast<T*>(out[0]->get_data_ptr()),
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
        }
        else if (node_op == "ConvolutionBackpropData")
        {
            // Note that args[1] and args[0] are switched here from the usual order.
            auto c = static_cast<const op::ConvolutionBackpropData*>(&node);
            reference::convolution<T>(reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                      reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                      reinterpret_cast<T*>(out[0]->get_data_ptr()),
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
        }
        else if (node_op == "Cos")
        {
            reference::cos<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Cosh")
        {
            reference::cosh<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Divide")
        {
            reference::divide<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                 reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                 reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                 out[0]->get_element_count());
        }
        else if (node_op == "Dot")
        {
            op::Dot* dot = dynamic_cast<op::Dot*>(&node);

            reference::dot(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(args[1]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           args[0]->get_shape(),
                           args[1]->get_shape(),
                           out[0]->get_shape(),
                           dot->get_reduction_axes_count());
        }

        else if (node_op == "Equal")
        {
            reference::equal<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                out[0]->get_element_count());
        }
        else if (node_op == "Exp")
        {
            reference::exp<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Floor")
        {
            reference::floor<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                out[0]->get_element_count());
        }
        else if (node_op == "FunctionCall")
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
        }
        else if (node_op == "Greater")
        {
            reference::greater<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                  reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                  out[0]->get_element_count());
        }
        else if (node_op == "GreaterEq")
        {
            reference::greater_eq<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                     reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                     reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                     out[0]->get_element_count());
        }
        else if (node_op == "Less")
        {
            reference::less<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(args[1]->get_data_ptr()),
                               reinterpret_cast<char*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "LessEq")
        {
            reference::less_eq<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                  reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                  out[0]->get_element_count());
        }
        else if (node_op == "Log")
        {
            reference::log<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Max")
        {
            const op::Max* max = static_cast<const op::Max*>(&node);
            reference::max<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              args[0]->get_shape(),
                              out[0]->get_shape(),
                              max->get_reduction_axes());
        }
        else if (node_op == "Maximum")
        {
            reference::maximum<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                  reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                  out[0]->get_element_count());
        }
        else if (node_op == "MaxPool")
        {
            op::MaxPool* max_pool = dynamic_cast<op::MaxPool*>(&node);

            reference::max_pool<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                   reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                   args[0]->get_shape(),
                                   out[0]->get_shape(),
                                   max_pool->get_window_shape(),
                                   max_pool->get_window_movement_strides(),
                                   max_pool->get_padding_below(),
                                   max_pool->get_padding_above());
        }
        else if (node_op == "MaxPoolBackprop")
        {
            op::MaxPoolBackprop* max_pool_backprop = dynamic_cast<op::MaxPoolBackprop*>(&node);

            reference::max_pool_backprop<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                            reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                            args[1]->get_shape(),
                                            out[0]->get_shape(),
                                            max_pool_backprop->get_window_shape(),
                                            max_pool_backprop->get_window_movement_strides(),
                                            max_pool_backprop->get_padding_below(),
                                            max_pool_backprop->get_padding_above());
        }
        else if (node_op == "Min")
        {
            const op::Min* min = static_cast<const op::Min*>(&node);
            reference::min<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              args[0]->get_shape(),
                              out[0]->get_shape(),
                              min->get_reduction_axes());
        }
        else if (node_op == "Minimum")
        {
            reference::minimum<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                  reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                  out[0]->get_element_count());
        }
        else if (node_op == "Multiply")
        {
            reference::multiply<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                   reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                   reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                   out[0]->get_element_count());
        }
        else if (node_op == "Negative")
        {
            reference::negate<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                 reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                 out[0]->get_element_count());
        }
        else if (node_op == "Not")
        {
            reference::logical_not(reinterpret_cast<char*>(args[0]->get_data_ptr()),
                                   reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                   out[0]->get_element_count());
        }
        else if (node_op == "NotEqual")
        {
            reference::not_equal<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                    reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                    reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                    out[0]->get_element_count());
        }
        else if (node_op == "OneHot")
        {
            auto oh = static_cast<const op::OneHot*>(&node);
            reference::one_hot<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                  args[0]->get_shape(),
                                  out[0]->get_shape(),
                                  oh->get_one_hot_axis());
        }
        else if (node_op == "Or")
        {
            reference::logical_or(reinterpret_cast<char*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<char*>(args[1]->get_data_ptr()),
                                  reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                  out[0]->get_element_count());
        }
        else if (node_op == "Parameter")
        {
        }
        else if (node_op == "Pad")
        {
            op::Pad* pad = dynamic_cast<op::Pad*>(&node);

            reference::pad(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(args[1]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           node.get_inputs().at(0).get_shape(),
                           node.get_output_shape(0),
                           pad->get_padding_below(),
                           pad->get_padding_above(),
                           pad->get_padding_interior());
        }
        else if (node_op == "Power")
        {
            reference::power<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                out[0]->get_element_count());
        }
        else if (node_op == "Product")
        {
            const op::Product* product = static_cast<const op::Product*>(&node);
            reference::product<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                  args[0]->get_shape(),
                                  out[0]->get_shape(),
                                  product->get_reduction_axes());
        }
        else if (node_op == "Reduce")
        {
            op::Reduce* reduce = dynamic_cast<op::Reduce*>(&node);
            std::shared_ptr<Function> reduction_function = reduce->get_functions()[0];

            std::function<T(T, T)> f = [this, &node, reduction_function](T x, T y) -> T {
                auto tx = std::make_shared<HostTensorView>(
                    node.get_inputs().at(0).get_element_type(), Shape{}, "reduce_temp_x");
                auto ty = std::make_shared<HostTensorView>(
                    node.get_inputs().at(1).get_element_type(), Shape{}, "reduce_temp_y");
                auto tr = std::make_shared<HostTensorView>(
                    node.get_output_element_type(0), Shape{}, "reduce_temp_r");
                *(reinterpret_cast<T*>(tx->get_data_ptr())) = x;
                *(reinterpret_cast<T*>(ty->get_data_ptr())) = y;
                call(reduction_function, {tr}, {tx, ty});
                return *(reinterpret_cast<T*>(tr->get_data_ptr()));
            };

            reference::reduce(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(args[1]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              node.get_inputs().at(0).get_shape(),
                              node.get_output_shape(0),
                              reduce->get_reduction_axes(),
                              f);
        }
        else if (node_op == "ReduceWindow")
        {
            op::ReduceWindow* reduce_window = dynamic_cast<op::ReduceWindow*>(&node);
            std::shared_ptr<Function> reduction_function = reduce_window->get_functions()[0];

            std::function<T(T, T)> f = [this, &node, reduction_function](T x, T y) -> T {
                auto tx = std::make_shared<HostTensorView>(
                    node.get_inputs().at(0).get_element_type(), Shape{}, "reduce_window_temp_x");
                auto ty = std::make_shared<HostTensorView>(
                    node.get_inputs().at(1).get_element_type(), Shape{}, "reduce_window_temp_y");
                auto tr = std::make_shared<HostTensorView>(
                    node.get_output_element_type(0), Shape{}, "reduce_window_temp_r");
                *(reinterpret_cast<T*>(tx->get_data_ptr())) = x;
                *(reinterpret_cast<T*>(ty->get_data_ptr())) = y;
                call(reduction_function, {tr}, {tx, ty});
                return *(reinterpret_cast<T*>(tr->get_data_ptr()));
            };

            reference::reduce_window(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                     reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                     reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                     node.get_inputs().at(0).get_shape(),
                                     node.get_output_shape(0),
                                     f,
                                     reduce_window->get_window_shape(),
                                     reduce_window->get_window_movement_strides());
        }
        else if (node_op == "Relu")
        {
            reference::relu<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "ReluBackprop")
        {
            reference::relu_backprop<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                        reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                        reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                        out[0]->get_element_count());
        }
        else if (node_op == "ReplaceSlice")
        {
            const op::ReplaceSlice* slice = static_cast<const op::ReplaceSlice*>(&node);
            reference::replace_slice<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                        reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                        reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                        args[1]->get_shape(),
                                        slice->get_lower_bounds(),
                                        slice->get_upper_bounds(),
                                        slice->get_strides(),
                                        out[0]->get_shape());
        }
        else if (node_op == "Reshape")
        {
            op::Reshape* reshape = dynamic_cast<op::Reshape*>(&node);
            reference::reshape(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               args[0]->get_shape(),
                               reshape->get_input_order(),
                               out[0]->get_shape());
        }
        else if (node_op == "Result")
        {
            op::Result* res = dynamic_cast<op::Result*>(&node);
            reference::result(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              shape_size(res->get_shape()));
        }
        else if (node_op == "Reverse")
        {
            op::Reverse* reverse = dynamic_cast<op::Reverse*>(&node);
            reference::reverse(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               args[0]->get_shape(),
                               out[0]->get_shape(),
                               reverse->get_reversed_axes());
        }
        else if (node_op == "Select")
        {
            reference::select<T>(reinterpret_cast<char*>(args[0]->get_data_ptr()),
                                 reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                 reinterpret_cast<T*>(args[2]->get_data_ptr()),
                                 reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                 out[0]->get_element_count());
        }
        else if (node_op == "SelectAndScatter")
        {
            ngraph::op::SelectAndScatter* select_and_scatter =
                dynamic_cast<ngraph::op::SelectAndScatter*>(&node);

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
                *(reinterpret_cast<T*>(tx->get_data_ptr())) = x;
                *(reinterpret_cast<T*>(ty->get_data_ptr())) = y;
                call(selection_function, {tr}, {tx, ty});
                return *(reinterpret_cast<char*>(tr->get_data_ptr()));
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
                *(reinterpret_cast<T*>(tx->get_data_ptr())) = x;
                *(reinterpret_cast<T*>(ty->get_data_ptr())) = y;
                call(scatter_function, {tr}, {tx, ty});
                return *(reinterpret_cast<T*>(tr->get_data_ptr()));
            };

            reference::select_and_scatter<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                             reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                             reinterpret_cast<T*>(args[2]->get_data_ptr()),
                                             reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                             args[0]->get_shape(),
                                             args[1]->get_shape(),
                                             out[0]->get_shape(),
                                             f_selection,
                                             f_scatter,
                                             select_and_scatter->get_window_shape(),
                                             select_and_scatter->get_window_movement_strides());
        }
        else if (node_op == "Sign")
        {
            reference::sign<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Sin")
        {
            reference::sin<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Sinh")
        {
            reference::sinh<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Slice")
        {
            const op::Slice* slice = static_cast<const op::Slice*>(&node);
            reference::slice<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                args[0]->get_shape(),
                                slice->get_lower_bounds(),
                                slice->get_upper_bounds(),
                                slice->get_strides(),
                                out[0]->get_shape());
        }
        else if (node_op == "Softmax")
        {
            const op::Softmax* softmax = static_cast<const op::Softmax*>(&node);
            reference::softmax<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                  out[0]->get_shape(),
                                  softmax->get_axes());
        }
        else if (node_op == "Sqrt")
        {
            reference::sqrt<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Subtract")
        {
            reference::subtract<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                   reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                   reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                   out[0]->get_element_count());
        }
        else if (node_op == "Sum")
        {
            const op::Sum* sum = static_cast<const op::Sum*>(&node);
            reference::sum<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              args[0]->get_shape(),
                              out[0]->get_shape(),
                              sum->get_reduction_axes());
        }
        else if (node_op == "Tan")
        {
            reference::tan<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Tanh")
        {
            reference::tanh<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else
        {
            std::stringstream ss;
            ss << "unsupported op " << node_op;
            throw ngraph_error(ss.str());
        }
    }
};
