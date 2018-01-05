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

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/interpreter/int_tensor_view.hpp"
#include "ngraph/runtime/interpreter/int_tensor_view.hpp"
#include "ngraph/runtime/kernel/abs.hpp"
#include "ngraph/runtime/kernel/acos.hpp"
#include "ngraph/runtime/kernel/add.hpp"
#include "ngraph/runtime/kernel/asin.hpp"
#include "ngraph/runtime/kernel/atan.hpp"
#include "ngraph/runtime/kernel/broadcast.hpp"
#include "ngraph/runtime/kernel/ceiling.hpp"
#include "ngraph/runtime/kernel/concat.hpp"
#include "ngraph/runtime/kernel/constant.hpp"
#include "ngraph/runtime/kernel/convert.hpp"
#include "ngraph/runtime/kernel/convolution.hpp"
#include "ngraph/runtime/kernel/copy.hpp"
#include "ngraph/runtime/kernel/cos.hpp"
#include "ngraph/runtime/kernel/cosh.hpp"
#include "ngraph/runtime/kernel/divide.hpp"
#include "ngraph/runtime/kernel/dot.hpp"
#include "ngraph/runtime/kernel/equal.hpp"
#include "ngraph/runtime/kernel/exp.hpp"
#include "ngraph/runtime/kernel/floor.hpp"
#include "ngraph/runtime/kernel/greater.hpp"
#include "ngraph/runtime/kernel/greater_eq.hpp"
#include "ngraph/runtime/kernel/less.hpp"
#include "ngraph/runtime/kernel/less_eq.hpp"
#include "ngraph/runtime/kernel/log.hpp"
#include "ngraph/runtime/kernel/max_pool.hpp"
#include "ngraph/runtime/kernel/maximum.hpp"
#include "ngraph/runtime/kernel/minimum.hpp"
#include "ngraph/runtime/kernel/multiply.hpp"
#include "ngraph/runtime/kernel/negate.hpp"
#include "ngraph/runtime/kernel/not.hpp"
#include "ngraph/runtime/kernel/not_equal.hpp"
#include "ngraph/runtime/kernel/one_hot.hpp"
#include "ngraph/runtime/kernel/power.hpp"
#include "ngraph/runtime/kernel/reduce.hpp"
#include "ngraph/runtime/kernel/replace_slice.hpp"
#include "ngraph/runtime/kernel/reshape.hpp"
#include "ngraph/runtime/kernel/select.hpp"
#include "ngraph/runtime/kernel/sign.hpp"
#include "ngraph/runtime/kernel/sin.hpp"
#include "ngraph/runtime/kernel/sinh.hpp"
#include "ngraph/runtime/kernel/slice.hpp"
#include "ngraph/runtime/kernel/sqrt.hpp"
#include "ngraph/runtime/kernel/subtract.hpp"
#include "ngraph/runtime/kernel/sum.hpp"
#include "ngraph/runtime/kernel/tan.hpp"
#include "ngraph/runtime/kernel/tanh.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        class PrimaryTensorView;

        namespace interpreter
        {
            class ExternalFunction;
            class INT_CallFrame;
        }
    }
}

// Compile and execute graphs
class ngraph::runtime::interpreter::INT_CallFrame : public runtime::CallFrame
{
public:
    INT_CallFrame(std::shared_ptr<ExternalFunction> external_function,
                  std::shared_ptr<Function> func);

    /// @brief Invoke the function with values matching the signature of the function.
    ///
    /// Tuples will be expanded into their tensor views to build the call frame.
    void call(const std::vector<std::shared_ptr<runtime::TensorView>>& inputs,
              const std::vector<std::shared_ptr<runtime::TensorView>>& outputs);

private:
    /// @brief Invoke the function with tuples pre-expanded to their underlying
    /// tensor views.
    void tensor_call(const std::vector<std::shared_ptr<TensorView>>& inputs,
                     const std::vector<std::shared_ptr<TensorView>>& outputs);
    void tensor_call(const std::vector<std::shared_ptr<INT_TensorView>>& inputs,
                     const std::vector<std::shared_ptr<INT_TensorView>>& outputs);
    void call(std::shared_ptr<Function> function,
              const std::vector<std::shared_ptr<runtime::interpreter::INT_TensorView>>& input_tvs,
              const std::vector<std::shared_ptr<runtime::interpreter::INT_TensorView>>& output_tvs);
    void handle_output_alias(
        const Node& node,
        const std::unordered_map<descriptor::TensorView*, std::vector<size_t>>& output_alias_map,
        const std::vector<std::shared_ptr<runtime::interpreter::INT_TensorView>>& output_tvs);

    std::shared_ptr<ExternalFunction> m_external_function;
    std::shared_ptr<Function> m_function;
    void generate_calls(const element::Type& base_type,
                        const element::Type& secondary_type,
                        ngraph::Node& op,
                        const std::vector<std::shared_ptr<INT_TensorView>>& args,
                        const std::vector<std::shared_ptr<INT_TensorView>>& out);

    template <typename BASE>
    void generate_calls(const element::Type& type,
                        ngraph::Node& op,
                        const std::vector<std::shared_ptr<INT_TensorView>>& args,
                        const std::vector<std::shared_ptr<INT_TensorView>>& out)
    {
        if (type == element::boolean)
        {
            op_engine<BASE, char>(op, args, out);
        }
        else if (type == element::f32)
        {
            op_engine<BASE, float>(op, args, out);
        }
        else if (type == element::f64)
        {
            op_engine<BASE, double>(op, args, out);
        }
        else if (type == element::i8)
        {
            op_engine<BASE, int8_t>(op, args, out);
        }
        else if (type == element::i16)
        {
            op_engine<BASE, int16_t>(op, args, out);
        }
        else if (type == element::i32)
        {
            op_engine<BASE, int32_t>(op, args, out);
        }
        else if (type == element::i64)
        {
            op_engine<BASE, int64_t>(op, args, out);
        }
        else if (type == element::u8)
        {
            op_engine<BASE, uint8_t>(op, args, out);
        }
        else if (type == element::u16)
        {
            op_engine<BASE, uint16_t>(op, args, out);
        }
        else if (type == element::u32)
        {
            op_engine<BASE, uint32_t>(op, args, out);
        }
        else if (type == element::u64)
        {
            op_engine<BASE, uint64_t>(op, args, out);
        }
        else
        {
            std::stringstream ss;
            ss << "unsupported element type " << type << " op " << op.get_name();
            throw std::runtime_error(ss.str());
        }
    }

    template <typename T, typename S>
    void op_engine(ngraph::Node& node,
                   const std::vector<std::shared_ptr<INT_TensorView>>& args,
                   const std::vector<std::shared_ptr<INT_TensorView>>& out)
    {
        std::string node_op = node.description();
        if (node_op == "Abs")
        {
            kernel::abs<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           out[0]->get_element_count());
        }
        else if (node_op == "Acos")
        {
            kernel::acos<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "Add")
        {
            kernel::add<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(args[1]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           out[0]->get_element_count());
        }
        else if (node_op == "Asin")
        {
            kernel::asin<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "Atan")
        {
            kernel::atan<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "Broadcast")
        {
            ngraph::op::Broadcast* broadcast = dynamic_cast<ngraph::op::Broadcast*>(&node);
            Shape in_shape = args[0]->get_shape();
            Shape out_shape = out[0]->get_shape();
            AxisSet broadcast_axes = broadcast->get_broadcast_axes();
            kernel::broadcast<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                 reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                 in_shape,
                                 out_shape,
                                 broadcast_axes);
        }
        else if (node_op == "Ceiling")
        {
            kernel::ceiling<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Concat")
        {
            const op::Concat* concat = static_cast<const op::Concat*>(&node);
            std::vector<T*> in_args;
            std::vector<Shape> in_shapes;
            for (std::shared_ptr<INT_TensorView> arg : args)
            {
                in_args.push_back(reinterpret_cast<T*>(arg->get_data_ptr()));
                in_shapes.push_back(arg->get_shape());
            }
            kernel::concat<T>(in_args,
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              in_shapes,
                              out[0]->get_shape(),
                              concat->get_concatenation_axis());
        }
        else if (node_op == "Constant")
        {
            const op::Constant* c = static_cast<const op::Constant*>(&node);
            kernel::constant<T>(reinterpret_cast<const T*>(c->get_data_ptr()),
                                reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                out[0]->get_element_count());
        }
        else if (node_op == "Convert")
        {
            kernel::convert<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<S*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Convolution")
        {
            auto c = static_cast<const op::Convolution*>(&node);
            kernel::convolution<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                   reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                   reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                   args[0]->get_shape(),
                                   args[1]->get_shape(),
                                   out[0]->get_shape(),
                                   c->get_window_movement_strides(),
                                   c->get_window_dilation_strides());
        }
        else if (node_op == "Cos")
        {
            kernel::cos<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           out[0]->get_element_count());
        }
        else if (node_op == "Cosh")
        {
            kernel::cosh<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "Divide")
        {
            kernel::divide<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(args[1]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Dot")
        {
            ngraph::op::Dot* dot = dynamic_cast<ngraph::op::Dot*>(&node);

            kernel::dot(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                        reinterpret_cast<T*>(args[1]->get_data_ptr()),
                        reinterpret_cast<T*>(out[0]->get_data_ptr()),
                        args[0]->get_shape(),
                        args[1]->get_shape(),
                        out[0]->get_shape(),
                        dot->get_reduction_axes_count());
        }

        else if (node_op == "Equal")
        {
            kernel::equal<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                             reinterpret_cast<T*>(args[1]->get_data_ptr()),
                             reinterpret_cast<char*>(out[0]->get_data_ptr()),
                             out[0]->get_element_count());
        }
        else if (node_op == "Exp")
        {
            kernel::exp<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           out[0]->get_element_count());
        }
        else if (node_op == "Floor")
        {
            kernel::floor<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                             reinterpret_cast<T*>(out[0]->get_data_ptr()),
                             out[0]->get_element_count());
        }
        else if (node_op == "FunctionCall")
        {
            std::shared_ptr<Function> function = node.get_function();
            call(function, args, out);
        }
        else if (node_op == "Greater")
        {
            kernel::greater<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(args[1]->get_data_ptr()),
                               reinterpret_cast<char*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "GreaterEq")
        {
            kernel::greater_eq<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                  reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                  reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                  out[0]->get_element_count());
        }
        else if (node_op == "Less")
        {
            kernel::less<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(args[1]->get_data_ptr()),
                            reinterpret_cast<char*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "LessEq")
        {
            kernel::less_eq<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(args[1]->get_data_ptr()),
                               reinterpret_cast<char*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Log")
        {
            kernel::log<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           out[0]->get_element_count());
        }
        else if (node_op == "Maximum")
        {
            kernel::maximum<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(args[1]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "MaxPool")
        {
            ngraph::op::MaxPool* max_pool = dynamic_cast<ngraph::op::MaxPool*>(&node);

            kernel::max_pool<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                args[0]->get_shape(),
                                out[0]->get_shape(),
                                max_pool->get_window_shape(),
                                max_pool->get_window_movement_strides());
        }
        else if (node_op == "Minimum")
        {
            kernel::minimum<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(args[1]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               out[0]->get_element_count());
        }
        else if (node_op == "Multiply")
        {
            kernel::multiply<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                out[0]->get_element_count());
        }
        else if (node_op == "Negative")
        {
            kernel::negate<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Not")
        {
            kernel::logical_not(reinterpret_cast<char*>(args[0]->get_data_ptr()),
                                reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                out[0]->get_element_count());
        }
        else if (node_op == "NotEqual")
        {
            kernel::not_equal<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                 reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                 reinterpret_cast<char*>(out[0]->get_data_ptr()),
                                 out[0]->get_element_count());
        }
        else if (node_op == "OneHot")
        {
            auto oh = static_cast<const op::OneHot*>(&node);
            kernel::one_hot<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                               reinterpret_cast<T*>(out[0]->get_data_ptr()),
                               args[0]->get_shape(),
                               out[0]->get_shape(),
                               oh->get_one_hot_axis());
        }
        else if (node_op == "Parameter")
        {
        }
        else if (node_op == "Power")
        {
            kernel::power<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                             reinterpret_cast<T*>(args[1]->get_data_ptr()),
                             reinterpret_cast<T*>(out[0]->get_data_ptr()),
                             out[0]->get_element_count());
        }
        else if (node_op == "Reduce")
        {
            ngraph::op::Reduce* reduce = dynamic_cast<ngraph::op::Reduce*>(&node);
            std::shared_ptr<ngraph::Function> reduction_function = reduce->get_function();

            std::function<T(T, T)> f = [this, &node, reduction_function](T x, T y) -> T {
                auto tx = std::make_shared<runtime::interpreter::INT_TensorView>(
                    node.get_inputs().at(0).get_element_type(), Shape{}, "reduce_temp_x");
                auto ty = std::make_shared<runtime::interpreter::INT_TensorView>(
                    node.get_inputs().at(1).get_element_type(), Shape{}, "reduce_temp_y");
                auto tr = std::make_shared<runtime::interpreter::INT_TensorView>(
                    node.get_output_element_type(0), Shape{}, "reduce_temp_r");
                *(reinterpret_cast<T*>(tx->get_data_ptr())) = x;
                *(reinterpret_cast<T*>(ty->get_data_ptr())) = y;
                call(reduction_function, {tx, ty}, {tr});
                return *(reinterpret_cast<T*>(tr->get_data_ptr()));
            };

            kernel::reduce(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(args[1]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           node.get_inputs().at(0).get_shape(),
                           node.get_output_shape(0),
                           reduce->get_reduction_axes(),
                           f);
        }
        // else if (node_op == "Remainder")
        // {
        //     // node = make_shared<op::Remainder>(args[0], args[1]);
        // }
        else if (node_op == "ReplaceSlice")
        {
            const op::ReplaceSlice* slice = static_cast<const op::ReplaceSlice*>(&node);
            kernel::replace_slice<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
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
            ngraph::op::Reshape* reshape = dynamic_cast<ngraph::op::Reshape*>(&node);
            kernel::reshape(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            args[0]->get_shape(),
                            reshape->get_input_order(),
                            out[0]->get_shape());
        }
        else if (node_op == "Select")
        {
            kernel::select<T>(reinterpret_cast<char*>(args[0]->get_data_ptr()),
                              reinterpret_cast<T*>(args[1]->get_data_ptr()),
                              reinterpret_cast<T*>(args[2]->get_data_ptr()),
                              reinterpret_cast<T*>(out[0]->get_data_ptr()),
                              out[0]->get_element_count());
        }
        else if (node_op == "Sign")
        {
            kernel::sign<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "Sin")
        {
            kernel::sin<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           out[0]->get_element_count());
        }
        else if (node_op == "Sinh")
        {
            kernel::sinh<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "Slice")
        {
            const op::Slice* slice = static_cast<const op::Slice*>(&node);
            kernel::slice<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                             reinterpret_cast<T*>(out[0]->get_data_ptr()),
                             args[0]->get_shape(),
                             slice->get_lower_bounds(),
                             slice->get_upper_bounds(),
                             slice->get_strides(),
                             out[0]->get_shape());
        }
        else if (node_op == "Sqrt")
        {
            kernel::sqrt<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else if (node_op == "Subtract")
        {
            kernel::subtract<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                                reinterpret_cast<T*>(args[1]->get_data_ptr()),
                                reinterpret_cast<T*>(out[0]->get_data_ptr()),
                                out[0]->get_element_count());
        }
        else if (node_op == "Sum")
        {
            const op::Sum* sum = static_cast<const op::Sum*>(&node);
            kernel::sum<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           args[0]->get_shape(),
                           out[0]->get_shape(),
                           sum->get_reduction_axes());
        }
        else if (node_op == "Tan")
        {
            kernel::tan<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                           reinterpret_cast<T*>(out[0]->get_data_ptr()),
                           out[0]->get_element_count());
        }
        else if (node_op == "Tanh")
        {
            kernel::tanh<T>(reinterpret_cast<T*>(args[0]->get_data_ptr()),
                            reinterpret_cast<T*>(out[0]->get_data_ptr()),
                            out[0]->get_element_count());
        }
        else
        {
            std::stringstream ss;
            ss << "unsupported op " << node_op;
            throw std::runtime_error(ss.str());
        }
    }
};
