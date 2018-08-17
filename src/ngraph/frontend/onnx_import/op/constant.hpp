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

#include "ngraph/node_vector.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace detail
            {
                namespace
                {
                    template <typename T>
                    inline std::shared_ptr<ngraph::op::Constant>
                        __make_ng_constant(const element::Type& type, const Tensor& tensor)
                    {
                        return std::make_shared<ngraph::op::Constant>(
                            type, tensor.get_shape(), tensor.get_data<T>());
                    }
                }

                template <Tensor::Type>
                inline std::shared_ptr<ngraph::op::Constant> make_ng_constant(const Tensor& tensor)
                {
                    throw error::tensor::unsupported_data_type{tensor};
                }

                template <>
                inline std::shared_ptr<ngraph::op::Constant>
                    make_ng_constant<Tensor::Type::float16>(const Tensor& tensor)
                {
                    return __make_ng_constant<float>(element::f32, tensor);
                }

                template <>
                inline std::shared_ptr<ngraph::op::Constant>
                    make_ng_constant<Tensor::Type::float32>(const Tensor& tensor)
                {
                    return __make_ng_constant<float>(element::f32, tensor);
                }

                template <>
                inline std::shared_ptr<ngraph::op::Constant>
                    make_ng_constant<Tensor::Type::float64>(const Tensor& tensor)
                {
                    return __make_ng_constant<double>(element::f64, tensor);
                }

                template <>
                inline std::shared_ptr<ngraph::op::Constant>
                    make_ng_constant<Tensor::Type::int32>(const Tensor& tensor)
                {
                    return __make_ng_constant<int32_t>(element::i32, tensor);
                }

                template <>
                inline std::shared_ptr<ngraph::op::Constant>
                    make_ng_constant<Tensor::Type::uint32>(const Tensor& tensor)
                {
                    return __make_ng_constant<uint32_t>(element::u32, tensor);
                }

                template <>
                inline std::shared_ptr<ngraph::op::Constant>
                    make_ng_constant<Tensor::Type::uint64>(const Tensor& tensor)
                {
                    return __make_ng_constant<uint64_t>(element::u64, tensor);
                }

            } // namespace detail

            inline std::shared_ptr<ngraph::op::Constant> constant(const Tensor& tensor)
            {
#define _make_ng_constant(_data_type)                                                              \
    case _data_type: return detail::make_ng_constant<_data_type>(tensor)

                switch (tensor.get_type())
                {
                    _make_ng_constant(Tensor::Type::float16);
                    _make_ng_constant(Tensor::Type::float32);
                    _make_ng_constant(Tensor::Type::float64);
                    _make_ng_constant(Tensor::Type::int32);
                    _make_ng_constant(Tensor::Type::uint32);
                    _make_ng_constant(Tensor::Type::uint64);
                default: throw error::tensor::invalid_data_type{tensor};
                }
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
